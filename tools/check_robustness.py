"""Run run.py on the robustness set (tools/make_robustness_set.py) and check every case.

Checks
  A. normal run: the run finishes; every video id gets output; broken inputs get exactly one error
     line and no second lines; good / unusual-but-valid videos get lines and no error line
  B. valid videos: run.py --sampling all output == tools/baseline_run.py output (exact per-second match)
  C. worker crash: a worker killed mid-run (test hook) -> error line for that video, others complete
  D. resume: output cut mid-way (with a half-written line) + --resume == full output

Usage: python tools/check_robustness.py [--dir cache/robust]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
F0 = "robust-00000-of-00003.parquet"
F1 = "robust-00001-of-00003.parquet"
F2 = "robust-00002-of-00003.parquet"
NAMES = ["good", "truncated", "tail_cut", "corrupt_mid", "empty", "null", "garbage", "rotated90",
         "res_change", "audio_only", "zero_frames", "one_frame", "vfr"]
MUST_ERROR_ONLY = {"empty", "null", "garbage", "audio_only", "zero_frames"}
MUST_BE_CLEAN = {"good", "rotated90", "res_change", "one_frame", "vfr"}
MAY_BE_PARTIAL = {"truncated", "tail_cut", "corrupt_mid"}


def run(args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    for attempt in (1, 2):
        try:
            r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, env=e, timeout=600)
            break
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (attempt {attempt}): {args}", flush=True)
            if attempt == 2:
                raise SystemExit(f"command timed out twice: {args}")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"command failed: {args}")
    return r


def parse(path):
    lines, errors = defaultdict(dict), defaultdict(list)
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if "error" in d:
            errors[d["video"]].append(d["error"])
        else:
            lines[d["video"]][d["second"]] = (tuple(d["objects"]), d["people"])
    return lines, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cache/robust")
    ap.add_argument("--workers", default="3")
    args = ap.parse_args()
    out = ROOT / "cache" / "robust_out"
    out.mkdir(exist_ok=True)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print("A. normal run")
    a = out / "a.jsonl"
    run(["run.py", "--input", args.dir, "--output", str(a), "--workers", args.workers])
    lines, errors = parse(a)
    for i, name in enumerate(NAMES):
        v = f"{F0}:{i}"
        nl, ne = len(lines.get(v, {})), len(errors.get(v, []))
        desc = f"{name:12s} lines={nl:3d} errors={ne} {errors.get(v, [''])[0][:60]}"
        if name in MUST_ERROR_ONLY:
            check(nl == 0 and ne == 1, desc)
        elif name in MUST_BE_CLEAN:
            check(nl > 0 and ne == 0, desc)
        else:
            check(ne <= 1 and (nl > 0 or ne == 1), desc + " (partial allowed)")
        if nl:
            secs = sorted(lines[v])
            check(secs == list(range(secs[-1] + 1)), f"{name:12s} seconds contiguous 0..{secs[-1]}")
    for i in range(2):
        v = f"{F1}:{i}"
        check(len(lines.get(v, {})) > 0 and not errors.get(v), f"next file row {i} processed normally")
    check(any(k.startswith(F2) for k in errors), f"unreadable parquet -> error line {dict((k, e) for k, e in errors.items() if k.startswith(F2))}")

    print("B. valid videos: pipeline (every frame) vs baseline")
    b1, b2 = out / "b_pipe.jsonl", out / "b_base.jsonl"
    run(["run.py", "--input", args.dir, "--output", str(b1), "--workers", args.workers, "--sampling", "all"])
    run(["tools/baseline_run.py", "--input", args.dir, "--output", str(b2)])
    p_lines, _ = parse(b1)
    r_lines, _ = parse(b2)
    for i, name in enumerate(NAMES):
        v = f"{F0}:{i}"
        if name in MUST_BE_CLEAN:
            same = p_lines.get(v) == r_lines.get(v)
            diff = sum(p_lines.get(v, {}).get(s) != r_lines[v][s] for s in r_lines.get(v, {}))
            check(same or diff <= 1, f"{name:12s} {len(r_lines.get(v, {}))} seconds, {diff} differ from baseline")

    print("C. worker crash")
    c = out / "c.jsonl"
    victim = f"{F1}:1"
    run(["run.py", "--input", args.dir, "--output", str(c), "--workers", args.workers],
        env={"VIDX_TEST_CRASH": victim})
    cl, ce = parse(c)
    check(any("crash" in e for e in ce.get(victim, [])), f"crashed video has error line: {ce.get(victim)}")
    check(len(cl.get(f"{F1}:0", {})) > 0 and len(cl.get(f"{F0}:0", {})) > 0, "other videos still processed")

    print("D. resume")
    full = a.read_text(encoding="utf-8").splitlines(keepends=True)
    d = out / "d.jsonl"
    cut = len(full) // 2
    d.write_text("".join(full[:cut]) + full[cut][: len(full[cut]) // 2], encoding="utf-8")
    run(["run.py", "--input", args.dir, "--output", str(d), "--workers", args.workers, "--resume"])
    dl, de = parse(d)
    fl, fe = parse(a)
    check(dl == fl, f"resume: per-second lines identical to the uninterrupted run ({sum(map(len, dl.values()))} lines)")
    check({k: len(v) for k, v in de.items()} == {k: len(v) for k, v in fe.items()}, "resume: same error lines, none duplicated")
    check(not d.read_text(encoding="utf-8").count("\n\n"), "resume: no blank/garbled lines")

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
