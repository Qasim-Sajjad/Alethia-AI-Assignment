"""Score a results.jsonl against the every-frame reference cache, using the brief's rules.

Reference per second (from cached per-frame detections at conf >= 0.4):
  - object present if found in >= half of the second's frames
  - people = median of per-frame counts, .5 rounded down
Metrics:
  - object F1, pooled over (video, second, object) items  [also per-class F1, macro F1]
  - people exact-match rate and within-1 rate over reference seconds
Missing predicted seconds count as wrong (no objects, people mismatch unless ref is 0... we
count them as mismatches explicitly). Extra predicted seconds add false positives.

Usage: python tools/score.py --ref-dir cache/ref --pred results.jsonl [--details]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx import CLASS_NAMES, CONF  # noqa: E402
from vidx.aggregate import aggregate  # noqa: E402

THRESH = {"f1": 0.95, "exact": 0.85, "within1": 0.95}


def load_ref(ref_dir: Path, conf: float = CONF) -> dict[str, dict[int, tuple[frozenset, int]]]:
    ref = {}
    for npz in sorted(ref_dir.glob("*.npz")):
        z = np.load(npz)
        vid = npz.stem.replace("__", ":")
        if str(z["error"]):
            continue  # damaged in reference build; skip from scoring
        sec = z["sec"]
        counts = counts_from_dets(len(sec), z["det_frame"], z["det_cls"], z["det_conf"], conf)
        ref[vid] = {s: (frozenset(o), p) for s, o, p in aggregate(sec, counts)}
    return ref


def counts_from_dets(n_frames, det_frame, det_cls, det_conf, conf):
    m = det_conf >= conf
    counts = np.zeros((n_frames, len(CLASS_NAMES)), np.int16)
    np.add.at(counts, (det_frame[m], det_cls[m].astype(int)), 1)
    return counts


def load_pred(path: Path) -> tuple[dict, set]:
    pred: dict[str, dict[int, tuple[frozenset, int]]] = defaultdict(dict)
    errors = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
                continue
            pred[d["video"]][int(d["second"])] = (frozenset(d["objects"]), int(d["people"]))
    return pred, errors


def score(ref, pred, videos=None, details=False):
    vids = sorted(ref) if videos is None else sorted(v for v in videos if v in ref)
    tp = fp = fn = 0
    cls_tp = defaultdict(int); cls_fp = defaultdict(int); cls_fn = defaultdict(int)  # noqa: E702
    n = exact = within1 = missing = extra = 0
    per_video = []
    for v in vids:
        r, p = ref[v], pred.get(v, {})
        v_n = v_exact = 0
        v_tp = v_fp = v_fn = 0
        for s, (ro, rp) in r.items():
            n += 1; v_n += 1  # noqa: E702
            if s in p:
                po, pp = p[s]
            else:
                missing += 1
                po, pp = frozenset(), None
            for c in CLASS_NAMES:
                a, b = c in ro, c in po
                if a and b:
                    tp += 1; cls_tp[c] += 1; v_tp += 1  # noqa: E702
                elif b:
                    fp += 1; cls_fp[c] += 1; v_fp += 1  # noqa: E702
                elif a:
                    fn += 1; cls_fn[c] += 1; v_fn += 1  # noqa: E702
            if pp is not None and pp == rp:
                exact += 1; v_exact += 1  # noqa: E702
            if pp is not None and abs(pp - rp) <= 1:
                within1 += 1
        for s, (po, _) in p.items():
            if s not in r:
                extra += 1
                fp += len(po)
                for c in po:
                    cls_fp[c] += 1
        v_f1 = 2 * v_tp / (2 * v_tp + v_fp + v_fn) if (v_tp + v_fp + v_fn) else 1.0
        per_video.append((v, v_n, v_f1, v_exact / max(v_n, 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    per_cls = {
        c: (2 * cls_tp[c] / (2 * cls_tp[c] + cls_fp[c] + cls_fn[c])) if (cls_tp[c] + cls_fp[c] + cls_fn[c]) else None
        for c in CLASS_NAMES
    }
    present = [c for c in CLASS_NAMES if cls_tp[c] + cls_fn[c] > 0]
    macro = float(np.mean([per_cls[c] for c in present])) if present else 1.0
    res = {
        "videos": len(vids), "seconds": n, "f1": f1, "f1_macro": macro,
        "exact": exact / max(n, 1), "within1": within1 / max(n, 1),
        "missing_seconds": missing, "extra_seconds": extra,
        "per_class_f1": per_cls,
        "support": {c: cls_tp[c] + cls_fn[c] for c in CLASS_NAMES},
    }
    res["pass"] = res["f1"] >= THRESH["f1"] and res["exact"] >= THRESH["exact"] and res["within1"] >= THRESH["within1"]
    if details:
        res["worst_videos"] = sorted(per_video, key=lambda x: (x[2] + x[3]))[:10]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()
    ref = load_ref(Path(args.ref_dir))
    pred, errors = load_pred(Path(args.pred))
    res = score(ref, pred, details=args.details)
    res["pred_error_videos"] = len(errors)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
