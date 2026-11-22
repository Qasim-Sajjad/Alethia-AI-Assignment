# Video indexing at speed

For each second of each FineVideo video: which of **person, car, dog, laptop, cell phone** YOLO11l finds,
and how many people are visible. Output is one JSON line per second (format in the brief, section 4).

## Install and run (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

1. System packages (OpenCV needs these on minimal Ubuntu images):

   ```bash
   sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
   ```

2. Python packages, inside the virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

3. **Preparation command** (run once, needs internet, not timed):

   ```bash
   python prepare.py
   ```

   Downloads the official `yolo11l.pt` into `weights/` and builds a TensorRT FP16 engine
   `weights/yolo11l.engine` for the GPU it runs on (≈ 15–20 min). A TensorRT engine is tied to the GPU
   and TensorRT version, so it has to be built on the grading machine.

4. **Run** (no internet needed):

   ```bash
   python run.py --input <folder of .parquet files> --output results.jsonl
   ```

`run.py` uses `weights/yolo11l.engine` when it exists; if the engine could not be built it falls back
to `weights/yolo11l.pt` with PyTorch FP16 (slower, same logic). Stage timings are written next to the
output as `results.jsonl.timings.json`.

Optional flags: `--workers N` (decode processes, default CPU count − 2), `--batch N` (default 64),
`--sampling all` (run the model on every frame), `--fp32`, `--resume` (continue an interrupted run),
`--dump-dir DIR` (save per-frame detections for checking).

## How it works

- **Exact model settings**: YOLO11l, 640 input, conf ≥ 0.4, Ultralytics letterbox (minimal stride-32
  padding, e.g. 640×360 → 640×384), class-aware NMS IoU 0.7, BGR frames, rotation tags applied.
  Letterbox, preprocessing and NMS reproduce Ultralytics' predictor exactly (checked pixel-for-pixel).
- **Adaptive per-second sampling** (`vidx/adaptive.py`): run 2 frames of a second; if they disagree on any
  object or on the people count, or a detection is within 0.05 of the 0.4 threshold, run 7 more evenly
  spaced frames; if the result is still close to 50/50, run every frame of that second. About 18–19% of
  frames go through the model.
- **Pipeline** (`vidx/pipeline.py`): parallel decode worker processes (longest videos first) decode every
  frame with PyAV, convert and letterbox only the frames the sampler asks for into shared-memory slots;
  one GPU process batches frames from all videos through the TensorRT engine. Damaged videos keep the
  seconds already decoded and get one error line; a crashed worker is replaced.

Design notes, measurements and decisions are in `specs/` (`00-overview.md` first).

## Results and checks

- `results.jsonl.gz`: output for the first 50 FineVideo files (run on a Colab L4, see `HARDWARE.md`).
- `eval/ref_66/`: every-frame reference detections for 66 videos (files 00000–00001), used by
  `tools/score.py` to check results with the brief's rules (object F1 per class, people exact / ±1).
- `tools/simulate.py` (sampling strategies offline), `tools/check_exact.py` (letterbox / detector vs
  Ultralytics), `tools/check_decode.py` (decoder and rotation vs OpenCV / ffmpeg),
  `tools/check_robustness.py` (damaged and unusual inputs), `tools/baseline_run.py` (first version:
  every frame, single process).

| On the 66-video sample | Time | Worst class F1 | People exact |
|---|---|---|---|
| First version, every frame (RTX 3050) | 10,085 s | 1.000 | 1.000 |
| Final: adaptive + TensorRT FP16 (L4) | 362 s | 0.984 | 0.988 |

## Repository layout

```
run.py              entry point (fast pipeline)
prepare.py          preparation step: weights + TensorRT engine
vidx/               pipeline, adaptive sampling, detector, letterbox, decoding, aggregation
tools/              scoring, simulation, checks, baseline, export of annotated videos
eval/ref_66/        every-frame reference cache for 66 videos
colab/L4_run.ipynb  notebook used for the full run on Colab
specs/              design notes and measurements
```
