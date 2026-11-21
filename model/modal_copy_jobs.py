"""
Copy the finished analyses from this laptop (server_jobs/) into the Modal Volume used by modal_app.py,
so every account still sees its games after the website switches to Modal.

    ..\\env\\Scripts\\python.exe modal_copy_jobs.py           list what would be copied
    ..\\env\\Scripts\\python.exe modal_copy_jobs.py --copy    copy it

Copied: finished analyses that belong to an account (results, analysis video, court mapping, the player's number).
Left out: the original upload and converted video (not needed to view results), unfinished, cancelled and
ownerless test analyses.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "server_jobs")
SKIP_FILES = ("input.mp4", "log.txt")


def finished_jobs():
    for job_id in sorted(os.listdir(JOBS)):
        path = os.path.join(JOBS, job_id, "job.json")
        if not os.path.exists(path):
            continue
        job = json.load(open(path, encoding="utf-8"))
        if job.get("status") == "done" and job.get("owner"):
            yield job_id, job


def files_of(job_id):
    root = os.path.join(JOBS, job_id)
    for folder, _, files in os.walk(root):
        for name in files:
            if name in SKIP_FILES or name.startswith("upload"):
                continue
            local = os.path.join(folder, name)
            yield local, "/" + os.path.relpath(local, JOBS).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy", action="store_true", help="copy to Modal (without it: only list)")
    args = parser.parse_args()
    plan = [(job_id, job, list(files_of(job_id))) for job_id, job in finished_jobs()]
    total = 0
    for job_id, job, files in plan:
        size = sum(os.path.getsize(local) for local, _ in files)
        total += size
        print(f"{job_id}  {job['title'][:30]:30}  {job.get('role', 'coach'):6}  {len(files):3} files  {size / 1e6:6.1f} MB")
    print(f"{len(plan)} analyses, {total / 1e6:.0f} MB")
    if not args.copy:
        print("Nothing copied (add --copy).")
        return
    import modal
    volume = modal.Volume.from_name("clutchmetrics-jobs", create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for _, _, files in plan:
            for local, remote in files:
                batch.put_file(local, remote)
    print("Copied to the Modal Volume clutchmetrics-jobs.")


if __name__ == "__main__":
    main()
