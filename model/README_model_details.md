# AI-Powered Basketball Video Analyser (Design Practicum)

Run every command from inside this `model` folder (ClutchMetrics_Final/model).

## Folder layout

| Path | What it is |
|---|---|
| `basketball_pipeline.py` | Main pipeline: `detect` → `track` → `ball`. Makes annotated videos and CSVs. |
| `jersey_numbers.py` | Reads jersey numbers (5-class model + Qwen3-VL-2B), votes per track, writes `players.json`, `tracks_named.csv`, `annotated_numbers.mp4`. |
| `label_numbers.py` | Hand-label 150 random player boxes and score the jersey numbers against them (accuracy for the report). |
| `court_mapping.py` | Court mapping: `calibrate` (click keyframes; predicted lines → mostly just Enter) then `apply` → `positions.csv` (metres, speed km/h, distance, off-court flag), `court_check.mp4`. |
| (website) | Court mapping for uploads: `court_mapping.py prepare` picks a few pictures (only wide court views: close-ups, crowd shots, replay fades and near-duplicates removed; at most `COURT_PICTURES`=12, 9 on the PAOK clip). A window opens on the website after the upload; the user clicks 4+ spots per picture on the picture and on a court diagram. The analysis keeps running meanwhile and waits at most `COURT_WAIT_MINUTES` (15) at the court step, then continues without court stats. |
| `match_events.py` | Ball holder, team ball control, passes, interceptions, shots (rim from 5-class model + TV scoreboard read by Qwen3-VL) → `events.json`, `report.json`, `possession.csv`. |
| `render_broadcast.py` | Final broadcast-style video `analysis.mp4`: team stat cards, number badges, speed, ball-holder tag, court mini-map, event feed. |
| `export_dashboard.py` | Writes `data/match.json` + H.264 `data/analysis.mp4` into the ClutchMetrics website (format: website `data/MATCH_FORMAT.md`). |
| `server.py` + `start_server.bat` | Analysis server for website uploads: converts any video (MP4/MOV/AVI/MKV/WebM → H.264 ≤1080p, constant fps), runs the whole pipeline, serves `match.json` + video. Public via ngrok `lizard-makeshift-clock.ngrok-free.dev`. Double-click `start_server.bat`. |
| `court_dataset.py` | Builds a court-landmark training set (33 FIBA landmarks, left/right-correct flips) from calibrated runs. |
| `compare_models.py` | v2 vs 5-class model on human-labelled Knicks + FIBA frames (ball F1, AP50, speed) → `runs/model_comparison/report.json`. |
| `compare_detectors.py` | Compares detectors on sampled frames; ball-labelling tool; scoring (F1). |
| `fiba_dataset.py` | Labelling tool for FIBA frames and builder of the COCO training set. |
| `train_rfdetr.py` | Fine-tunes RF-DETR Nano (576px) on NBA + college + FIBA data. |
| `train_rfdetr_colab.ipynb` | Colab notebook that runs `train_rfdetr.py` on a GPU. |
| `weights/rfdetr_basketball_v2/` | **Current best model** (ball / player / referee) + `test_report.json`. |
| `weights/rfdetr_basketball_576/` | First fine-tuned model (v1), kept for comparison. |
| `weights/rfdetr_5class/checkpoint_best_total.pth` | 5-class RF-DETR Nano (ball, number, player, referee, rim) – used for jersey numbers and rim. |
| `weights/*.pt` | Older YOLO models, only used by `compare_detectors.py`. |
| `datasets/fiba_ausind/` | Our 350 hand-labelled AUS-IND frames (`ball_clicks.csv`) + COCO splits. |
| `datasets/fiba_ausind.zip` | Same dataset, zipped for Google Drive / Colab. |
| `datasets/test_videos/` | Knicks–76ers test video. |
| `runs/aus_ind_14m_to_16m_clip.mp4` | Main demo clip (FIBA U16, AUS vs IND). |
| `runs/pipeline/aus_ind_v2/` | **Latest results**: `annotated_ball.mp4`, `annotated.mp4`, `tracks.csv`, `ball.csv`. |
| `runs/pipeline/aus_ind_ft/`, `aus_ind/`, `aus_trial/` | Older results (v1 model, hosted Roboflow model, 12 s trial). |
| `runs/detector_comparison/` | Detector comparison outputs and our Knicks ball labels. |
| `experiments/fix_referees.py` | Re-applies the referee rule to an existing run without re-tracking (used once on `aus_ind_v2`; original saved as `tracks_before_ref_fix.csv`). |
| `experiments/` | One-off test scripts, check images, jersey-number crops. Not needed to run the pipeline. |

## Run the pipeline on the demo clip

```
python basketball_pipeline.py detect --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind_v2 --start 10 --end 120 --fps 15
python basketball_pipeline.py track  --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind_v2
python basketball_pipeline.py ball   --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind_v2
python jersey_numbers.py --name aus_ind_v2 --teams AUS IND
python court_mapping.py calibrate --name aus_ind_v2
python court_mapping.py apply --name aus_ind_v2
python jersey_numbers.py --name aus_ind_v2 --teams AUS IND --court-link
python match_events.py --name aus_ind_v2 --teams AUS IND --scoreboard 480 945 1440 1005
python render_broadcast.py --name aus_ind_v2
python export_dashboard.py --name aus_ind_v2 --site ../website --team-names Australia India --title "Australia vs India" --competition "FIBA U16 Asian Championship 2023" --date 2023-09-17 --venue "Doha, Qatar"
```
Court-landmark model tested and rejected: `koppolusameer/yolo11n-basketball-court-keypoints` found too few points on the FIBA clip and confused the left and right court halves (also on NBA frames), so its court lines landed metres off. Manual keyframes are used instead.
(`--reuse` on `jersey_numbers.py` redoes only the voting and video, without re-reading numbers.)

## Key results so far

| Test | Score |
|---|---|
| Ball AP50 on unseen FIBA frames (v1 → v2) | 30 → 69 |
| Ball F1 on 195 hand-labelled Knicks frames | Roboflow hosted 0.53 · v1 0.63 · **v2 0.79** |
| v2 vs 5-class (human labels) | Ball F1 Knicks 0.79 vs 0.52 · Ball F1 FIBA 0.84 vs 0.40 · Player AP50 0.98 vs 0.97 · Referee AP50 0.93 vs 0.86 → v2 used for ball/players, 5-class only for numbers + rim |
| AUS-IND ball found per frame (v1 → v2) | detected 29% → 78%, unknown 35% → 6% |
| Jersey-number reading (Qwen3-VL-2B, 70 crops) | ~45 read, ~80–85% of reads correct before voting |
| Jersey numbers on AUS-IND | 73% of player boxes carry a number (was 53%). Numbers found: AUS 2,4,5,6,16,18 · IND 2,7,18,23,36. 11 track swaps split, 5 team-colour errors fixed by number. Accuracy: run `label_numbers.py`. |
| Court mapping (27 clicked keyframes + camera motion) | 94% of frames mapped; player speed median 6.1 km/h, 95th pct 15.2 km/h; 51 bench/staff tracks removed |
| Shots vs TV scoreboard (54–108 s, where it is visible) | all 3 real baskets found (IND 3PT, IND 2PT, AUS 3PT) with correct points; 3 other attempts correctly marked missed |
| Referee fix (decide referee vs player from confident detections only) | 11 referee tracks had been counted as players; interceptions 10 → 8; player-box number coverage 75% |
| Pass fix (holder stays with the current player while the ball is in reach; passer and receiver ≥ 2 m apart) | Flicker between team-mates standing together removed: AUS passes 19 → 16, IND 13 → 14; interceptions 8 → 4 |
| Referee dead-ball fix (a referee holding the ball ≥ 0.33 s breaks the pass chain) | Referee-to-referee throw at 15 s no longer counted as an AUS pass; AUS passes 16 → 15, IND 14 → 15; interceptions 4 → 3 |
| Court-position linking + lineup elimination backed by the track's own reads (`--court-link`) | On-court player boxes with a number 81% → 86%; 7 more track pieces named, 3 wrong candidates rejected after checking by eye |
| Shooter = last player holding the ball ≥ 0.4 s | 56 s shot now credited to IND #7 (3-pointer, confirmed by scoreboard) instead of AUS #18 who only touched it; interceptions 3 → 2 |
| Points by FIBA Official Basketball Rules 2026 (Art. 2.5.4, 16.2.1; Interpretations 2024 16-1..16-10) | FT = 1, release inside the arc or on the line = 2, beyond the 6.75 m arc / 0.90 m corner lines = 3. Scoreboard-confirmed points: AUS +3, IND +5 (= real TV score change 15-2 → 18-7). Position-based value agreed with the scoreboard on the clear 2-pointer; the IND 3-pointer at 56 s measured 6.74 m (within ±0.5 m accuracy) so the scoreboard decided. Clip totals incl. unconfirmed: AUS 6, IND 7 |
| Court-landmark model v1 (YOLO11s-pose, 603 frames of one game) | Seen views: median 0.32 m. **Held-out views (last 25 s): median 5.6 m → not reliable**, so uploads skip court mapping (`AUTO_COURT` off in `server.py`) until more games are calibrated and it passes a held-out test |
| Upload pipeline test (25 s MOV clip, uploaded through the server API) | Converted + analysed in 3 minutes, opens in the coach dashboard via the match picker; court-based sections show a 'not available' note |
| Court-landmark model v2 (1,787 frames from 3 arenas: AUS-IND, Unicaja BCL, India-Qatar; PAOK EuroCup held out) | On the unseen PAOK arena: median 13-16 m error (7-8 m even if left/right swapped); often 0 landmarks found on its dark floor. Not usable: automatic court mapping stays off for uploads. Generalising to new arenas would need many more calibrated arenas. |

## Calibrating more games for the court model

```
python court_mapping.py calibrate --name <game_name> --video path	o\game.mp4 --start 0 --end 120
python court_mapping.py apply --name <game_name>
python court_dataset.py --names aus_ind_v2 <game_name> [<game_name2> ...]
```
Then retrain the court model and check it with `python court_mapping.py auto --name <a game not used for training>`.

## Moving to Modal (cloud GPU, no laptop needed)

`modal_app.py` runs the same server on Modal: a small CPU web server (always reachable, free when idle) and the model
on an L4 GPU that starts for each upload and stops when it finishes (about $0.80 an hour, so roughly $0.30–0.40 per
3-minute video). At most 15 uploads a day. Log in once with `modal token new`.

Run from this folder, once there are funds on the Modal account:

1. Deploy (builds the image once: about 15–20 minutes):
   `..\env\Scripts\python.exe -m modal deploy modal_app.py`
   It prints the web address, e.g. `https://<workspace>--clutchmetrics-web.modal.run`.
   Check it: open `<address>/api/health` in a browser.
2. Copy the finished analyses so every account keeps its games:
   `..\env\Scripts\python.exe modal_copy_jobs.py` (lists them), then `..\env\Scripts\python.exe modal_copy_jobs.py --copy`.
3. Point the website at Modal: in `website/config.js` set `DEFAULT_SERVER` to the address from step 1, then commit and
   push the website.
4. Test one upload from the website (court mapping window, player number, Cancel) and watch it under Apps →
   clutchmetrics in the Modal dashboard.

To go back to the laptop, set `DEFAULT_SERVER` back to `https://lizard-makeshift-clock.ngrok-free.dev` and run
`start_server.bat`. New uploads made on Modal stay on Modal (`modal volume get clutchmetrics-jobs <job id>` downloads one).
