"""
Fine-tune the jersey-number reader Qwen3-VL-2B with LoRA (the 2B base weights stay frozen; small adapters are trained).

  python finetune_qwen_numbers.py                  # train, then compare base vs tuned on the hand-labelled test crops
  python finetune_qwen_numbers.py --eval-only      # only compare (uses weights/qwen_numbers_lora)

Training data (datasets/numbers, built by number_dataset.py):
  train.csv        number crops from our own games, labelled by the pipeline's confirmed numbers
  ebard_train.csv  E-BARD (CC-BY-4.0) NBA crops, hand labels per sequence (a sample is used)
  ebard_none       E-BARD sequences labelled "number not visible" -> answer "none", so the tuned reader keeps saying
                   "none" on unreadable crops (our voting relies on that)
Same prompt and crop size (224 px high) as the live reader, so the adapter can be dropped into jersey_numbers.py.
The test game (Australia vs India, 47 hand-labelled crops) is never used for training.
"""

import argparse
import csv
import os
import random
import re
import time
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image

from jersey_numbers import READER_PROMPT

BASE = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "datasets/numbers"
OUT = "weights/qwen_numbers_lora"
READ_SIZE = 224


def rows_of(name):
    path = os.path.join(DATA, f"{name}.csv")
    return list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []


def ebard_none(zip_path, per_sequence=4):
    """Crops from E-BARD sequences labelled NaN (number not visible) -> label 'none'."""
    out = rows_of("ebard_none")
    if out:
        return out
    z = zipfile.ZipFile(zip_path)
    folders = {}
    for n in z.namelist():
        if n.startswith("annotated_crop/") and n.endswith(".png"):
            folders.setdefault(n.rsplit("/", 1)[0], []).append(n)
    split_dir = os.path.join(DATA, "ebard_none")
    os.makedirs(split_dir, exist_ok=True)
    for line in z.read("all.txt").decode().splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[-1] != "nan":
            continue
        folder = parts[0].replace("annotated", "annotated_crop", 1)
        frames = sorted(folders.get(folder, []))
        for n in frames[:: max(1, len(frames) // per_sequence)][:per_sequence]:
            path = os.path.join(split_dir, f"none_{folder.split('/')[-1]}_{n.rsplit('/', 1)[1]}")
            open(path, "wb").write(z.read(n))
            out.append({"path": path, "label": "none", "run": "ebard", "frame": n, "track": folder})
    with open(os.path.join(DATA, "ebard_none.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "run", "frame", "track"])
        w.writeheader()
        w.writerows(out)
    return out


def load_image(path):
    img = cv2.imread(path)
    # like the live reader (224 px high); very wide crops are capped at 2x that width so a batch fits in 8 GB
    scale = min(READ_SIZE / img.shape[0], 2 * READ_SIZE / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def chat(answer=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": READER_PROMPT}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def batch_inputs(proc, rows, device):
    images = [load_image(r["path"]) for r in rows]
    full = [proc.apply_chat_template(chat(r["label"]), tokenize=False) for r in rows]
    prompt = proc.apply_chat_template(chat(), add_generation_prompt=True, tokenize=False)
    enc = proc(text=full, images=images, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, img in enumerate(images):
        n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels[i, :n_prompt] = -100                       # learn only the answer (+ end token)
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def read(model, proc, rows, device, batch=16):
    proc.tokenizer.padding_side = "left"
    prompt = proc.apply_chat_template(chat(), add_generation_prompt=True, tokenize=False)
    answers = []
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        enc = proc(text=[prompt] * len(part), images=[load_image(r["path"]) for r in part],
                   return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False)
        texts = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        answers += [t.strip() if re.fullmatch(r"\d{1,2}", t.strip()) else None for t in texts]
    return answers


def report(name, answers, rows):
    labels = [r["label"] for r in rows]
    kept = [(a, l) for a, l in zip(answers, labels) if a is not None]
    right = sum(a == l for a, l in kept)
    print(f"  {name:28} answered {len(kept)}/{len(rows)} | correct {right}/{len(kept)} "
          f"({100 * right / max(1, len(kept)):.0f}% of answers, {100 * right / max(1, len(rows)):.0f}% of all)")
    return [(l, a) for a, l in zip(answers, labels) if a != l]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ebard-sample", type=int, default=5000, help="E-BARD number crops to mix in")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch x accum)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--zip", default="datasets/ebard/all.zip")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda"

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16).to(device)
    test = rows_of("test")

    if not args.eval_only:
        ours = rows_of("train")
        ebard = rows_of("ebard_train")
        random.shuffle(ebard)
        none = ebard_none(args.zip)
        data = ours + ebard[: args.ebard_sample] + none
        random.shuffle(data)
        print(f"Training examples: {len(data)} (our games {len(ours)}, E-BARD numbers {min(len(ebard), args.ebard_sample)}, "
              f"E-BARD 'none' {len(none)})")
        config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        steps = int(len(data) * args.epochs / (args.batch * args.accum))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
        proc.tokenizer.padding_side = "right"
        model.train()
        started, losses = time.time(), []
        for step in range(steps):
            total = 0.0
            for a in range(args.accum):
                start = (step * args.accum + a) * args.batch
                part = [data[(start + i) % len(data)] for i in range(args.batch)]
                enc = batch_inputs(proc, part, device)
                loss = model(**enc).loss / args.accum
                loss.backward()
                total += loss.item()
                del enc, loss
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(total)
            if (step + 1) % 50 == 0 or step + 1 == steps:
                print(f"  step {step + 1}/{steps} | loss {np.mean(losses[-50:]):.3f} | {time.time() - started:.0f}s", flush=True)
        model.save_pretrained(OUT)
        print(f"Saved LoRA adapter to {OUT}")
        model.eval()
    else:
        model = PeftModel.from_pretrained(model, OUT).eval()

    print(f"\nTest crops (Australia vs India, hand labels, never trained on): {len(test)}")
    with model.disable_adapter():
        report("Base Qwen3-VL-2B", read(model, proc, test, device), test)
    wrong = report("Fine-tuned (LoRA)", read(model, proc, test, device), test)
    print("  fine-tuned mistakes (true -> read):", ", ".join(f"{l}->{a}" for l, a in wrong))


if __name__ == "__main__":
    main()
