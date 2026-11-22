# Hardware

## Full run (all 50 Parquet files) — Google Colab Pro, L4 runtime

| | |
|---|---|
| Where | Google Colab Pro (rented cloud VM), L4 GPU runtime |
| GPU | NVIDIA L4, 23,034 MiB (≈ 22.5 GiB) memory, driver 580.82.07 |
| CPU | Intel Xeon @ 2.20 GHz, 6 cores / 12 threads (12 vCPUs) |
| RAM | 53 GB |
| Disk | NVMe persistent disk (SSD, non-rotational), 236 GB; dataset read from local disk |
| OS | Ubuntu 24.04.4 LTS, Python 3.13.16 |
| Software | torch 2.7.1 (CUDA 12.6), ultralytics 8.4.216, TensorRT 10.16.1.11 (FP16 engine) |

**Full run time: 8,033 s (2 h 13 min 53 s)**, measured from start to finish of
`python run.py --input finevideo/data --output results.jsonl` (one uninterrupted run; `run.py`'s own
wall-clock timer, see `results.jsonl.timings.json`).

- 50 files, 1,650 videos, 12,981,615 frames decoded, 2,332,879 frames run through YOLO11l (18.0%),
  461,757 output lines, 0 error lines.
- Not included in that time (both happen before the timed run, as the brief allows):
  the preparation step `python prepare.py` (download `yolo11l.pt` + build the TensorRT engine: 1,060 s)
  and downloading the dataset (77 s).

## Development machine (baseline and tuning)

| | |
|---|---|
| Where | Own computer |
| GPU | NVIDIA GeForce RTX 3050, 8 GB |
| CPU | Intel Core i3-10100 @ 3.60 GHz, 4 cores / 8 threads |
| RAM | 16 GB |
| Disk | HDD (Seagate ST500DM009) for the project and sample data |
| OS | Windows 11 Pro, Python 3.11.15 |

Used for the every-frame reference on 66 videos (files 00000–00001, 5.21 h of video): the first,
every-frame version took 10,085 s there. Timing comparisons on the same 66 videos are in
`specs/20-pipeline.md`.
