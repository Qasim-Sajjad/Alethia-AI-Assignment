# 30 — Robustness

Status: **all checks pass** (local, 2026-11-22)
Tools: `tools/make_robustness_set.py` (builds `cache/robust/`), `tools/check_robustness.py` (runs and checks)

## Test inputs

`robust-00000-of-00003.parquet` (13 rows), `robust-00001` (2 normal videos, checks the run continues),
`robust-00002` (Parquet truncated to half: footer missing, unreadable).

| Row | Case | Result |
|---|---|---|
| 0 | good | 21 lines, no error |
| 1 | truncated (first 40% of bytes) | 9 lines + 1 error line |
| 2 | tail cut (last 30% removed) | 15 lines + 1 error line |
| 3 | random bytes over 10% of the middle | 10 lines + 1 error line |
| 4 | empty bytes | 1 error line |
| 5 | null value | 1 error line |
| 6 | random bytes, not an mp4 | 1 error line |
| 7 | 90° display rotation tag | 21 lines, identical to baseline (which rotates via the verified path) |
| 8 | resolution changes mid-stream (202×360 → 270×480) | 10 lines, identical to baseline |
| 9 | audio only, no video stream | 1 error line |
| 10 | video stream with zero frames | 1 error line |
| 11 | single frame | 1 line |
| 12 | variable frame rate | 8 lines, identical to baseline |
| — | unreadable Parquet file | 1 error line, next file processed normally |

Partial videos always have contiguous seconds 0..N before their error line, as the brief requires.

## Other checks

- **Every-frame pipeline vs baseline** on all valid unusual videos: 0 differing seconds.
- **Worker crash** (test hook `VIDX_TEST_CRASH=<video id>` kills a worker with `os._exit`): the video
  gets a `worker crashed` error line, a replacement worker starts, all other videos complete.
- **Resume**: output cut in the middle of a line, rerun with `--resume` → per-second lines identical to
  the uninterrupted run, no duplicated error lines.

## Bugs found and fixed by these tests

1. **Hang on worker crash.** A worker that dies right after taking a task can lose its "started video X"
   message, so the main process waited forever. Fix: main tracks every video it hands out; videos with
   no result when all workers are gone get a `worker crashed` error line; dead workers are replaced.
2. **Resume could leave a video half-written.** It treated any video with lines as finished. Fix: only
   the last video in the file can be incomplete (each video is written in one go), so it is dropped
   and redone.

## Notes

- With 1 decoder thread (pipeline default), FFmpeg reports truncation as an error at the end of the
  data, so truncated videos get their lines plus an error line. With FFmpeg's automatic threading the
  same files end silently.
- The old baseline script stalled once during these tests (no CPU, no output); per-video decode tests
  with timeouts found no decoder hang, and it did not recur. It is only used for comparison.
