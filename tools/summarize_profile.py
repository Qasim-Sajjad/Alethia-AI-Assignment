"""Summarize cache/profile.csv produced by profile_videos.py."""
import collections
import csv
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "cache/profile.csv"
r = list(csv.DictReader(open(path)))
ok = [x for x in r if not x.get("error")]
print("videos", len(r), "errors", len(r) - len(ok))
print("total hours", round(sum(float(x["duration_s"]) for x in ok) / 3600, 2))
print("codecs", collections.Counter(x["codec"] for x in ok))
print("pix_fmt", collections.Counter(x["pix_fmt"] for x in ok))
print("res", collections.Counter(f"{x['width']}x{x['height']}" for x in ok).most_common(10))
print("fps", collections.Counter(round(float(x["avg_fps"]), 2) for x in ok).most_common(10))
print("rotation", collections.Counter(x["rotation"] for x in ok))
d = [float(x["duration_s"]) for x in ok]
print("duration s min/med/max", min(d), st.median(d), max(d))
g = [float(x["gop_s"]) for x in ok]
print("gop s min/med/max", min(g), st.median(g), max(g))
v = [float(x["vfr_frac"]) for x in ok]
print("vfr_frac >0.01:", sum(a > 0.01 for a in v), "max", max(v))
print("total frames (approx)", sum(int(x["n_packets"]) for x in ok))
