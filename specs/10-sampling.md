# 10 — Frame sampling strategy

Status: **decided (provisional until re-checked with the fast inference path)**
Tool: `tools/simulate.py` (offline, uses the every-frame cache in `cache/ref`, no GPU)

## Question

Which frames must we run YOLO11l on so that per-second results still match the every-frame
reference? Target (internal, stricter than grader): **each class F1 ≥ 0.97**, people exact ≥ 0.90,
people ±1 ≥ 0.97 — on videos we did not tune on.

## Method

- Reference: all 526,328 frames of 66 videos (files 00000 + 00001, 18,806 seconds).
- **Tune on file 00000 (33 videos), confirm on file 00001 (33 videos)** — no parameter was chosen
  by looking at 00001.
- A strategy picks frames per second; only those frames are aggregated with the brief's rules;
  the scorer compares with the reference. Cost = % of all frames sent to the model.

Strategy notation `adapt:k:TRIGGERS:rKxM`:
- `k` coarse frames per second, evenly spaced and centred.
- Triggers that mark a second as uncertain: `o` samples disagree on an object, `p` sampled people
  counts differ, `b0.05` a sampled detection has conf within 0.05 of 0.4, `n` result differs from a
  neighbouring second.
- Refine: `rK` re-decide with K evenly spaced frames; `xM` if that is still ambiguous (an object's
  frame fraction within M of 0.5, or the people median is held by < half the samples), use **all**
  frames of the second. Without `rK`, an uncertain second goes straight to all frames.

## Results

Tune set (file 00000):

| strategy | frames % | min class F1 | worst | people exact | ±1 |
|---|---|---|---|---|---|
| fixed:3 | 10.8 | 0.889 | dog | 0.956 | 0.998 |
| fixed:5 | 18.0 | 0.933 | dog | 0.970 | 1.000 |
| fixed:10 | 36.0 | 0.957 | dog | 0.974 | 0.999 |
| adapt:2:op (refine → all) | 23.1 | 0.988 | phone | 0.991 | 1.000 |
| adapt:2:op:r7 (no ambiguity check) | 11.5 | 0.953 | phone | 0.971 | 0.999 |
| adapt:2:op:r7x0.15 | 16.7 | 0.985 | phone | 0.985 | 1.000 |
| **adapt:2:opb0.05:r7x0.15** | **18.6** | **0.990** | phone | **0.989** | **1.000** |
| adapt:1:opn:r5x0.15 | 15.9 | 0.987 | car | 0.983 | 1.000 |

Held-out set (file 00001):

| strategy | frames % | min class F1 | worst | people exact | ±1 |
|---|---|---|---|---|---|
| fixed:5 | 17.7 | 0.942 | car | 0.964 | 0.998 |
| fixed:10 | 35.4 | 0.940 | phone | 0.975 | 1.000 |
| adapt:2:op:r7x0.15 | 18.0 | 0.975 | phone | 0.986 | 1.000 |
| **adapt:2:opb0.05:r7x0.15** | **19.9** | **0.988** | phone | **0.990** | **1.000** |
| adapt:1:opn:r5x0.15 | 17.2 | 0.988 | phone | 0.983 | 1.000 |
| adapt:3:op:r9x0.15 | 22.5 | 0.994 | phone | 0.992 | 1.000 |

All 66 videos, chosen strategy: **19.3% of frames, per-class F1 = person 0.999, car 0.995,
dog 0.994, laptop 0.995, phone 0.989; people exact 0.990, ±1 0.9999.**

## Findings

1. **Fixed sampling cannot meet a per-class target.** Even 10 frames/s (≈35% of frames) leaves the
   worst class at 0.94–0.96. Errors concentrate in the few seconds where an object is present in
   ~half the frames, and rare classes have so few seconds that each error costs a lot of F1.
2. **Adaptive coarse → refine works**: spend frames only on seconds whose coarse samples disagree.
   About 18–25% of seconds get refined; the rest are decided from 2 frames.
3. **Partial refinement needs an ambiguity check.** Refining to 7 frames alone drops F1 to 0.95;
   adding "go to all frames if still near 50/50" recovers it at a small extra cost.
4. The borderline-confidence trigger (`b0.05`) buys extra safety for ~2% more frames — worth it,
   because FP16/TensorRT will nudge confidences near 0.4 (see speed phase).
5. Cell phone is consistently the weakest class (small, flickery detections).

## Decision

Use **`adapt:2:opb0.05:r7x0.15`**:
1. Run 2 evenly spaced frames per second.
2. If they disagree on any object or on the people count, or any target-class detection has
   confidence within 0.05 of 0.4 → run 7 evenly spaced frames (reusing the 2).
3. If any object's fraction over those frames is within 0.15 of 50%, or the people median is held by
   fewer than half of them → run every frame of that second.

Expected cost ≈ **19–20% of frames → ~5× less inference** than every-frame. Decoding still touches
every frame (H.264), so decode becomes the next bottleneck.

Why not cheaper options: `adapt:2:op:r7x0.15` (17%) falls to 0.975 on the held-out set — too thin
a margin before FP16 effects. `adapt:1:opn:...` (17%) needs the next second's result before deciding,
which complicates streaming, and relies on a single frame per second.

## Caveats / to re-check

- Only 66 videos; rare classes have 300–800 positive seconds each, so per-class F1 has noise of
  roughly ±0.005. Margin over 0.95 is ~0.04.
- Re-run the simulation once the fast inference path (FP16 / TensorRT) produces its own per-frame
  cache: confidences shift slightly, which may change both triggers and results.
- Implementation needs a per-video buffer of the current second's decoded frames (≤ 30 frames of
  ≤ 640×360 ≈ 20 MB) so refinement can run on frames already decoded.
