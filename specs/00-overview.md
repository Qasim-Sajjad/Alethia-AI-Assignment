# Video Indexing at Speed — Project Spec

Living document. Update the status table and decision log as work progresses.
Detailed decisions get their own file in `specs/` (e.g. `10-sampling.md`).

## Goal

`python run.py --input <folder of .parquet> --output results.jsonl`

For every second of every video, output which of {person, car, dog, laptop, cell phone}
are present and how many people are visible — as fast as possible while passing the
grader's correctness check against an every-frame YOLO11l reference.

- Deadline: **Sun 22 Nov 2026, 11:55 pm PKT** (last commit graded)
- Grader machine: RTX 4090 24 GB, 32 threads, 64 GB RAM, Ubuntu 22.04, Python 3.11, no internet during timed run, no Docker
- Untimed prep command allowed (internet on) → download weights, build TensorRT engine
- Test set ≈ 13 h, 6 h hard limit; some files may be damaged/unusual

## Hard constraints (must match reference exactly)

- Model `yolo11l.pt`, imgsz 640, conf ≥ 0.4
- Ultralytics defaults: letterbox (keep aspect), class-aware NMS IoU 0.7, BGR input
- Apply rotation tag before inference
- Second `s` = frames with `s ≤ t − t_first < s+1` (t_first = first decoded frame's timestamp)
- Line for every second up to and including the one containing the last frame
- Damaged video → one `error` line; keep lines already written for a partially processed video
- **No fine-tuning, no extra preprocessing** — "correct" means "agrees with stock YOLO11l"

## Pass criteria (grader) and our internal target (safety margin)

| Metric | Grader | Our target |
|---|---|---|
| Object F1 (per video-second-object) | ≥ 0.95 | ≥ 0.97 |
| People exact match | ≥ 85% | ≥ 90% |
| People within ±1 | ≥ 95% | ≥ 97% |

Reference per second: object present if `2·hits ≥ n_frames`; people = median of per-frame
counts, halfway rounded **down**.

## Environments

| Role | Machine |
|---|---|
| Dev / small tests | Local: RTX 3050 8 GB, i3-10100 (4c/8t), 16 GB RAM, Win 11, Python 3.11 venv (uv) |
| Reference build, tuning, full 50-file run | Colab Pro **L4** (Ada, same arch as 4090), results/engine saved to Drive |
| Grading | RTX 4090 (see above) |

HF token lives in `.env` (`hf_fine_video_tk`) locally and in Colab Secrets — never committed.

## Phases

| # | Phase | Status |
|---|---|---|
| 0 | Setup: repo, venv, weights, sample data, data profiling | done |
| 1 | Baseline `run.py`: every frame, exactly correct (= "first working version" timing) | done (local 3050: ~48 fps) |
| 2 | Reference cache (per-frame detections, 66 videos / 5.2 h from files 00000–00001) + `tools/score.py` | done: 66/66 videos, 526,328 frames, 18,806 seconds, 0 errors; 2.8 h at 52 fps on the 3050; baseline scores 100% |
| 3 | Offline simulation of sampling strategies (fixed k/sec, adaptive coarse→refine) | pending |
| 4 | Speed: parallel decode workers, cross-video batching, TensorRT FP16, NVDEC, static-frame reuse | pending |
| 5 | Robustness: damaged files, rotation, VFR, zero-length, resume after disconnect | pending |
| 6 | Full 50-file run on L4 → `results.jsonl.gz`, `HARDWARE.md`, README, requirements, `ai_chats/` | pending |

`WRITEUP.md` is written by the candidate, without AI.

## Key design ideas

1. **Scorer first.** `tools/score.py` reimplements grader rules; baseline every-frame output must score exactly 100%.
2. **Simulate sampling from cache.** Once per-frame detections are cached, any frame-selection strategy can be scored offline in seconds — no GPU.
3. **Adaptive sampling (coarse → refine).** Run odd number (3/5) of evenly spaced frames per second; if samples disagree on any object or on people count (or differ from neighbouring seconds), run more/all frames of that second. Saves GPU inference, not decode.
4. **Pipeline.** N decode processes (one video each, longest first) → shared-memory frame queue → one GPU process batching frames across videos. Decode and inference overlap.
5. **Re-score every speed change** that can alter per-frame detections (FP16/TensorRT, decoder, resize path) by re-running the sample set against the reference.
6. Pick the **fastest** config that meets our target margins, not the highest F1.

## Findings so far

**Data profile** (`tools/profile_videos.py`, files 00000–00001, 66 videos, 5.21 h, ~526k frames):
- All H.264 yuv420p, constant frame rate (no VFR detected), no rotation tags.
- Low resolution: 50/66 are 640×360; the rest are smaller, including portrait (270×480, 202×360, 360×640).
- fps: 30 / 29.97 / 25 / 23.98 / 24. Duration 21 s – 640 s (median 263 s). Keyframe every ~1.5–5.3 s.
- Implications: decoding is cheap per frame (≤ 640×360); Ultralytics letterboxes 640×360 to **640×384**
  (minimal stride-32 padding, not a 640×640 square). A fixed 640×640 TensorRT engine would pad
  differently than the reference → must check its effect on detections.
- Full 50 files ≈ 130 h ≈ 13M frames.

**Decoder check** (`tools/check_decode.py`): PyAV `bgr24` frames are pixel-identical to OpenCV
`VideoCapture` frames, and our rotation handling is pixel-identical to ffmpeg autorotate and OpenCV
for display-rotation tags 90/180/270. Colour-conversion risk is closed.

**Scorer** (`tools/score.py`): baseline vs its own cache = 100% on all metrics. Sanity check on 2 videos
using every k-th frame:

| every k-th frame | F1 (pooled) | F1 (macro) | people exact | ±1 |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 (~6 fps) | 0.975 | 0.911 | 0.966 | 1.000 |
| 15 (~2 fps) | 0.896 | 0.745 | 0.899 | 0.994 |
| 30 (~1 fps) | 0.868 | 0.629 | 0.859 | 0.987 |

Only 2 videos, so rare classes have tiny support — but it already shows fixed sparse sampling hurts
rare classes most (macro F1) and that sampling must always cover the **last frame's second**
(naive striding left 1–2 seconds missing).

**Reference cache class support** (seconds where the object is present, of 18,806):
person 13,678 · laptop 770 · dog 495 · cell phone 376 · car 305. Rare classes have a few hundred
seconds each — enough to measure, but per-class F1 will be noisy.

## Open risks
- Grader's exact F1 definition (pooled vs per-class) unknown → report both, require both.
- Decode is likely the bottleneck on low-core machines; NVDEC availability/behaviour on the grader VM.
- Colab disconnects → `run.py` must append incrementally and skip completed videos.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-20 | Full run on Colab L4; dev on local 3050 | Local i3 too weak for 125 h; L4 same arch as 4090 |
| 2026-11-20 | No fine-tuning / no extra preprocessing | Correctness is measured against stock YOLO11l output |
| 2026-11-20 | Choose fastest config meeting margin targets | Grader only counts speed if correctness passes |
| 2026-11-21 | Decode with PyAV from in-memory bytes | Pixel-identical to OpenCV/ffmpeg, no temp files, exact pts |
| 2026-11-21 | Cache detections down to conf 0.25, count at 0.4 | Same ≥0.4 set after NMS; lets us study borderline confidences |
| 2026-11-21 | Torch 2.7.1 (CUDA 12.6 wheels) | Grader driver supports CUDA 12; PyPI Linux wheels are cu126 |

## How to reproduce (local)

```
python run.py --input finevideo/data --output cache/baseline_2files.jsonl --dump-dir cache/ref
python tools/score.py --ref-dir cache/ref --pred <results.jsonl> --details
python tools/check_decode.py --parquet finevideo/data/train-00000-of-01357.parquet --row 0
```
