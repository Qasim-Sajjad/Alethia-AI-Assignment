# 20 — Fast pipeline design

Status: **built and verified (FP32)** — `run.py` + `vidx/pipeline.py`; baseline kept as `tools/baseline_run.py`
Replaces the single-process baseline in `run.py`. Sampling logic comes from `10-sampling.md`
(`adapt:2:opb0.05:r7x0.15`).

## Why the baseline is slow

One process does decode → BGR → letterbox → inference strictly in turn. CPU waits for the GPU and
the GPU waits for the CPU; one core decodes one video at a time; the model runs on 100% of frames.
Result: 52 fps on the RTX 3050 (10,085 s for the 66-video, 5.21 h sample).

## Architecture

```
                 ┌─ decode worker 1 (video A) ─┐
 main process    ├─ decode worker 2 (video B) ─┤  frames via         ┌──────────────┐
 (task list,  ──►├─ decode worker 3 (video C) ─┼─ shared memory ───► │ GPU process  │
  writer)        │          ...                │  slots              │ batch + YOLO │
                 └─ decode worker N (video Z) ─┘ ◄── small results ──┴──────────────┘
                              │
                              └── per-second lines ──► writer ──► results.jsonl
```

### Main process
- Lists every (parquet file, row) with its mp4 byte size (cheap proxy for duration).
- Sorts **longest first** so one long video does not finish alone at the end.
- Starts N decode workers (default `cpu_count − 2`) and one GPU process.
- Receives finished per-second lines and writes them; one write per completed video.
- Watches workers: if one dies, writes an error line for its in-flight video and starts a new one.

### Decode workers (N processes)
For each video taken from the task queue:
1. Decode **every** frame (H.264 requires it), keeping frames as decoded YUV `VideoFrame`s in a
   private buffer covering the last few seconds.
2. For each second: convert to BGR (+ rotation) and letterbox **only the frames the model needs**,
   write them into shared-memory slots, send requests to the GPU process.
3. Keep decoding the next seconds while waiting for answers (pipelined, no stall).
4. Run the adaptive logic on the answers: accept the 2-frame result, or request the extra 7 frames,
   or all frames of that second — taken from the buffer, never re-decoded.
5. Aggregate each second with the brief's rules; send lines to the main process.
6. On any exception: send the seconds already finished plus one error line, move to the next video.

Letterbox is done here on the CPU with OpenCV, exactly as Ultralytics does (minimal stride-32
padding, value 114, `INTER_LINEAR` resize when scale ≠ 1), so pixels match the reference.
GPU resizing gives slightly different pixels and could flip borderline detections.

### GPU process (1 process, stateless)
- Collects requests from all workers; groups them **by letterboxed shape** (most are 640×384)
  into batches of up to B frames (e.g. 64–128) or a short timeout.
- Reads slots without copying (NumPy view → pinned tensor → GPU), normalises, runs the YOLO11l
  network directly, then Ultralytics' own `non_max_suppression` (conf 0.25 for caching / 0.4 for
  counting, IoU 0.7, class-aware, max_det 300).
- Replies per frame: counts of the 5 classes at conf ≥ 0.4 and min |conf − 0.4| over target
  detections (for the borderline trigger). Frees the slot.
- Batches never mix shapes: mixing would force a common padding and change results.

## Shared-memory frame slots

- Each worker owns a fixed pool of slots (e.g. 32) in shared memory, sized for its current video's
  letterboxed frame; reallocated if a larger video arrives.
- Request message is tiny: `(worker_id, slot, frame_id, h, w)`. Response: `(frame_id, counts, near)`.
- No free slot → the worker waits. This is the **backpressure**: when the GPU is the bottleneck,
  workers cannot run ahead and fill memory.
- Only the ~19% of frames sent to the model cross processes; the decode buffer stays private.

## Output, errors, resume

- Each video's lines are written together when it finishes; an error line follows any partial video.
- Unreadable Parquet file → one error line, continue with the next file.
- `--resume` (for Colab disconnects): skip videos already in the output, trim a half-written last
  line. Off by default so the grader's run starts fresh.
- `--dump-dir`: also save per-frame detections of the frames that were run, for re-scoring and
  re-simulating with the fast inference path.

## Timing instrumentation

Each process accumulates time per stage; the main process prints a summary and writes
`timings.json` at the end.

| Where | Stages |
|---|---|
| Worker | parquet read, decode, BGR + letterbox, wait for free slot, wait for GPU answers |
| GPU process | idle (waiting for frames), host→device copy, inference (CUDA events), NMS, mean batch size |
| Main | write, total wall clock, frames decoded, frames inferred |

Reading the bottleneck: GPU mostly idle → decode-bound (more workers, NVDEC). Workers mostly
waiting on the GPU → inference-bound (TensorRT FP16, batch size). Helper modes: `--decode-only`
(no model) and a synthetic inference benchmark.

## How we measure the improvement

Same 66 videos (5.21 h, 526,328 frames), same GPU, wall clock from start to finish:

| Run | Purpose |
|---|---|
| Baseline (every frame, single process) — 3050: **10,085 s** | reference point |
| Adaptive sampling only, single process | effect of sampling alone |
| Pipeline only, every frame | effect of parallel decode + batching alone |
| Pipeline + adaptive sampling (final structure) | combined |
| + FP16 / TensorRT (speed phase) | effect of inference engine |

Each row isolates one change, which answers write-up Q1 ("which single change saved the most time").
Repeat the key rows on the Colab L4 so first-vs-final is compared on one GPU.

### Measured (local RTX 3050 8 GB, i3-10100 4c/8t, 66 videos / 5.21 h / 526,328 frames)

| Run | Wall time | Speedup | Frames run | Notes |
|---|---|---|---|---|
| Baseline, every frame, single process | 10,085 s | 1.0× | 100% | `tools/baseline_run.py` |
| Pipeline + adaptive, FP32, 3 workers, batch 32 | **1,646 s** | **6.1×** | 19.3% | 320 video-fps; peak GPU mem 6.3 GB |
| Pipeline + adaptive, **FP16** (PyTorch), 3 workers, batch 32 | **970 s** | **10.4×** | 19.3% | 542 video-fps; peak GPU mem 2.2 GB; GPU infer 908 s (94%) |

**FP16 accuracy** (vs FP32 every-frame reference): per-class F1 person 0.9989, car 0.9951, dog 0.9940,
laptop 0.9935, phone 0.9880; people exact 0.9888, ±1 0.9999 (FP32 pipeline: phone 0.9893, exact 0.9898).
Per frame (`tools/compare_dumps.py`, the 101,647 frames it ran): 0.63% of frames have a different
0.4-count than the reference; matched confidences shift by median 0.0004, p99 0.007. The borderline
trigger (±0.05 around 0.4) absorbs these shifts. **Decision: use FP16.**

Stage timings of the pipeline run (`cache/fast_fp32.jsonl.timings.json`):
- GPU process: inference **1,563 s of 1,646 s wall (95%)**, idle only 16 s, gather 25 s, mean batch 26.
  GPU throughput ≈ 65 frames/s (FP32).
- Workers (summed over 3): decode 307 s, BGR + letterbox 233 s, the rest waiting on the GPU.
- Main: Parquet read 13 s, writing < 0.01 s.

→ Locally the pipeline is **inference-bound**: decode + preprocessing need ~540 core-seconds for the
whole sample, the GPU needs 1,563 s. Next lever is the inference engine (FP16 / TensorRT).

### Issue found and fixed: NMS time limit

First attempt used batch 64: GPU memory hit 7.96 / 8 GB, Windows spilled to system memory, NMS took
2.6–3.8 s and hit Ultralytics' time limit (2 s + 0.05 s × batch). On timeout Ultralytics **returns
empty detections for the rest of the batch** — a silent correctness bug. Fixes: default batch 32
(peak 6.3 GB on the 3050) and `max_time_img=10` in our NMS call (a safety cutoff, not a detection
setting; a one-frame reference never reaches it). Rerun had no warnings.

### Measured on Colab L4 (24 GB, 12 vCPU Xeon 2.2 GHz, 53 GB RAM, Python 3.13), same 66 videos

| Run | Wall time | Video-fps | GPU infer / wall | Min class F1 | People exact |
|---|---|---|---|---|---|
| Pipeline + adaptive, PyTorch FP16, 10 workers, batch 64 | 572 s | 920 | 520 / 572 s | 0.988 (phone) | 0.989 |
| Pipeline + adaptive, **TensorRT 10 FP16** engine, 10 workers, batch 64 | **362 s** | **1,453** | 314 / 362 s | **0.984** (phone) | **0.988** |

TensorRT per-class F1: person 0.9988, car 0.9951, dog 0.9899, laptop 0.9942, phone 0.9840; ±1 0.99995.
Per frame: 1.09% of TensorRT frames differ in a 0.4-count from the FP32 reference (PyTorch FP16: 0.63%);
matched confidences shift by median 0.0009, p99 0.011. Still GPU-bound (87% of wall in inference;
decode ≈ 48 s per worker). **Decision: TensorRT FP16 engine is the final configuration.**

TensorRT notes:
- TensorRT **11** removed the FP16 builder flag; Ultralytics then requires NVIDIA ModelOpt "AutoCast"
  mixed precision, which failed to install next to torch 2.7.1 (it pinned `torch==2.7.1+cu126`).
  **TensorRT 10.16.1.11** keeps the plain FP16 flag and is pinned in `requirements.txt`.
- Engine build (`prepare.py`, dynamic shapes, max batch 64): 1,060 s on the L4, untimed by the grader.
  Execution context reserves ~9 GB of GPU memory for the max shape (64×640×640).

## Correctness check

The pipeline dumps per-frame detections for the frames it ran; its `results.jsonl` is scored
against the every-frame reference. With the same FP32 model it must reproduce the simulation
exactly (all 66: min class F1 0.989, people exact 0.990). Any difference is a pipeline bug.

**Result (FP32 pipeline, all 66 videos):** frames run 19.30% (simulation 19.3%); per-class F1 person
0.9989, car 0.9951, dog 0.9940, laptop 0.9948, phone 0.9893; people exact 0.9898, ±1 0.9999;
0 missing / 0 extra seconds. Identical to the simulation → pipeline logic verified.

## Expected gains

| Change | Effect |
|---|---|
| Adaptive sampling | ~5× fewer model runs |
| BGR + letterbox only for sampled frames | ~5× less CPU preprocessing |
| CPU and GPU overlap | no idle time on either side |
| N decode workers | decode scales with cores (grader: 32 threads) |
| Cross-video batches | fuller GPU batches |

Local i3 (4 cores) will likely be decode-bound; L4 (12 vCPU) and the grader (32 threads) should
gain much more. Real numbers go in the table above once measured.
