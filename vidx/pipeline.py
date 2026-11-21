"""Fast pipeline: parallel decode workers + one batching GPU process (design: specs/20-pipeline.md).

main process  : reads Parquet files, feeds (video_id, mp4 bytes) tasks longest-first, writes output,
                restarts dead workers, prints/saves stage timings
decode worker : decodes every frame, runs the adaptive sampling state machine, BGR+letterboxes only
                the frames it needs into its shared-memory slots, aggregates seconds, returns lines
GPU process   : gathers requests from all workers, batches by letterboxed shape, runs YOLO11l + NMS,
                replies with per-frame counts / near-threshold distance / detections
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from . import CLASS_NAMES, CONF
from .adaptive import K1, ambiguous, decide, pick, stage1_uncertain, stage2_frames
from .letterbox import IMGSZ

NC = len(CLASS_NAMES)
SLOT_SHAPE = (IMGSZ, IMGSZ, 3)  # letterboxed frames never exceed 640x640
SLOT_BYTES = IMGSZ * IMGSZ * 3


@dataclass
class Config:
    weights: str = "weights/yolo11l.pt"
    workers: int = max((os.cpu_count() or 4) - 2, 1)
    spare_workers: int = 4
    slots: int = 24
    batch: int = 32
    max_pending_secs: int = 6
    decode_threads: int | None = 1
    sampling: str = "adaptive"  # adaptive | all
    half: bool = False
    dump_dir: str | None = None
    gpu_timeout: float = 120.0
    prefetch_bytes: int = 3 * 2**30


# ----------------------------------------------------------------------------------------- worker

@dataclass
class SecState:
    frames: list
    stage: int = 1
    res: dict = field(default_factory=dict)  # frame offset -> (counts(5,), near)
    outstanding: int = 0
    run_idx: np.ndarray | None = None


class Timer:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, k, dt, n=1):
        self.t[k] += dt
        self.n[k] += n


def worker_main(wid, cfg: Config, task_q, req_q, resp_q, out_q, shm_name, n_total_workers):
    from .io import iter_frames, to_bgr
    from .letterbox import letterbox

    shm = SharedMemory(name=shm_name)
    slots = np.ndarray((n_total_workers, cfg.slots) + SLOT_SHAPE, np.uint8, buffer=shm.buf)[wid]
    free = list(range(cfg.slots))
    token = 0
    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            vid, mp4 = task
            token += 1
            out_q.put(("start", wid, vid))
            if os.environ.get("VIDX_TEST_CRASH") == vid:  # test hook: simulate a hard decoder crash
                os._exit(3)
            tm = Timer()
            lines, error, dump = run_video(vid, mp4, wid, token, cfg, slots, free, req_q, resp_q, tm,
                                           iter_frames, to_bgr, letterbox)
            out_q.put(("video", wid, vid, lines, error, dict(tm.t), dict(tm.n)))
            if dump is not None and cfg.dump_dir:
                np.savez_compressed(Path(cfg.dump_dir) / (vid.replace(":", "__") + ".npz"), **dump)
    finally:
        del slots
        shm.close()


def run_video(vid, mp4, wid, token, cfg, slots, free, req_q, resp_q, tm, iter_frames, to_bgr, letterbox):
    pending: dict[int, SecState] = {}
    results: dict[int, tuple] = {}
    to_submit: deque = deque()
    all_secs: list[int] = []
    dump_frames, dump_cls, dump_conf, dump_ran = [], [], [], []
    sec_start: dict[int, int] = {}  # second -> global index of its first frame (for dumps)

    def enqueue(sec, st, offsets):
        new = [int(o) for o in offsets if int(o) not in st.res]
        st.outstanding += len(new)
        for o in new:
            to_submit.append((sec, o))

    def finish(sec, st, idx):
        c = np.stack([st.res[int(o)][0] for o in idx])
        results[sec] = decide(c)
        del pending[sec]
        st.frames = None

    def advance(sec, st):
        n = len(st.frames)
        if cfg.sampling == "all":
            finish(sec, st, np.arange(n)); return  # noqa: E702
        if st.stage == 1:
            idx = pick(n, K1)
            if len(idx) == n:
                finish(sec, st, idx); return  # noqa: E702
            c = np.stack([st.res[int(o)][0] for o in idx])
            near = np.array([st.res[int(o)][1] for o in idx])
            if not stage1_uncertain(c, near):
                finish(sec, st, idx); return  # noqa: E702
            st.stage = 2
            st.run_idx = stage2_frames(n)
            enqueue(sec, st, st.run_idx)
            if st.outstanding == 0:
                advance(sec, st)
        elif st.stage == 2:
            idx = st.run_idx
            c = np.stack([st.res[int(o)][0] for o in idx])
            if len(idx) < n and ambiguous(c):
                st.stage = 3
                st.run_idx = np.arange(n)
                enqueue(sec, st, st.run_idx)
                if st.outstanding == 0:
                    advance(sec, st)
            else:
                finish(sec, st, idx)
        else:
            finish(sec, st, np.arange(n))

    def handle(msg):
        slot, tok, sec, off, counts, near, cls, conf = msg
        free.append(slot)
        if tok != token or sec not in pending:
            return
        st = pending[sec]
        st.res[off] = (counts, near)
        st.outstanding -= 1
        if cfg.dump_dir:
            g = sec_start[sec] + off
            dump_ran.append(g)
            dump_frames.append(np.full(len(cls), g, np.int32)); dump_cls.append(cls); dump_conf.append(conf)  # noqa: E702
        if st.outstanding == 0:
            advance(sec, st)

    def wait_one(label):
        t = time.perf_counter()
        try:
            msg = resp_q.get(timeout=cfg.gpu_timeout)
        except queue.Empty:
            raise RuntimeError("GPU process timeout")
        tm.add(label, time.perf_counter() - t)
        handle(msg)

    def drain():
        while True:
            try:
                msg = resp_q.get_nowait()
            except queue.Empty:
                break
            handle(msg)

    def submit_all():
        while to_submit:
            sec, off = to_submit.popleft()
            st = pending.get(sec)
            if st is None:
                continue
            while not free:
                wait_one("wait_slot")
            t = time.perf_counter()
            img = letterbox(to_bgr(st.frames[off]))
            h, w = img.shape[:2]
            slot = free.pop()
            slots[slot, :h, :w] = img
            tm.add("convert_letterbox", time.perf_counter() - t)
            req_q.put((wid, slot, token, sec, off, h, w))
            tm.add("frames_sent", 0)

    def close_second(sec, frames):
        st = SecState(frames)
        pending[sec] = st
        n = len(frames)
        enqueue(sec, st, np.arange(n) if cfg.sampling == "all" else pick(n, K1))

    error = None
    cur, buf = None, []
    t_dec = time.perf_counter()
    try:
        gen = iter_frames(mp4, cfg.decode_threads)
        for sec, frame in gen:
            all_secs.append(sec)
            if cur is not None and sec != cur:
                tm.add("decode", time.perf_counter() - t_dec)
                if cur in pending or cur in results:  # non-monotonic timestamps: keep first group only
                    pass
                else:
                    close_second(cur, buf)
                buf = []
                drain()
                submit_all()
                while len(pending) > cfg.max_pending_secs:
                    wait_one("wait_gpu")
                    submit_all()
                t_dec = time.perf_counter()
            if cur is None or sec != cur:
                sec_start.setdefault(sec, len(all_secs) - 1)
            cur = sec
            buf.append(frame)
        tm.add("decode", time.perf_counter() - t_dec)
    except Exception as e:  # noqa: BLE001 - damaged video: keep what decoded
        error = f"decode failed: {type(e).__name__}: {str(e)[:100]}"
    try:
        if buf and cur not in pending and cur not in results:
            close_second(cur, buf)
        buf = []
        submit_all()
        while pending:
            wait_one("wait_gpu")
            submit_all()
    except Exception as e:  # noqa: BLE001
        error = error or f"inference failed: {type(e).__name__}: {str(e)[:100]}"
        traceback.print_exc()
    tm.add("frames_decoded", 0, len(all_secs))

    lines = []
    if results:
        last = max(results)
        for s in range(last + 1):
            objs, people = results.get(s, (frozenset(), 0))
            lines.append(json.dumps({"video": vid, "second": s,
                                     "objects": [c for c in CLASS_NAMES if c in objs], "people": int(people)}))
    if not all_secs and error is None:
        error = "no decodable frames"
    dump = None
    if cfg.dump_dir:
        dump = {
            "sec": np.array(all_secs, np.int32),
            "ran": np.array(sorted(dump_ran), np.int32),
            "det_frame": np.concatenate(dump_frames) if dump_frames else np.zeros(0, np.int32),
            "det_cls": np.concatenate(dump_cls) if dump_cls else np.zeros(0, np.int8),
            "det_conf": np.concatenate(dump_conf) if dump_conf else np.zeros(0, np.float32),
            "error": np.array(error or ""),
        }
    return lines, error, dump


# -------------------------------------------------------------------------------------------- GPU

def gpu_main(cfg: Config, req_q, resp_qs, shm_name, n_total_workers, stat_q, ready_ev):
    import torch

    from .detector import Detector

    torch.set_num_threads(2)
    det = Detector(cfg.weights, half=cfg.half)
    shm = SharedMemory(name=shm_name)
    slots = np.ndarray((n_total_workers, cfg.slots) + SLOT_SHAPE, np.uint8, buffer=shm.buf)
    tm = Timer()
    ready_ev.set()
    stop = False
    while not stop:
        t = time.perf_counter()
        msg = req_q.get()
        tm.add("idle", time.perf_counter() - t)
        if msg is None:
            break
        batch = [msg]
        while len(batch) < cfg.batch:
            try:
                m = req_q.get_nowait()
            except queue.Empty:
                break
            if m is None:
                stop = True
                break
            batch.append(m)
        groups = defaultdict(list)
        for m in batch:
            groups[(m[5], m[6])].append(m)
        for (h, w), ms in groups.items():
            t0 = time.perf_counter()
            arr = np.stack([slots[m[0], m[1], :h, :w] for m in ms])
            t1 = time.perf_counter()
            dets = det.infer(arr)
            t2 = time.perf_counter()
            for m, (cls, conf) in zip(ms, dets):
                keep = conf >= CONF
                counts = np.bincount(cls[keep], minlength=NC).astype(np.int16)
                near = float(np.abs(conf - CONF).min()) if len(conf) else float("inf")
                resp_qs[m[0]].put((m[1], m[2], m[3], m[4], counts, near, cls, conf))
            t3 = time.perf_counter()
            tm.add("gather", t1 - t0)
            tm.add("infer", t2 - t1, len(ms))
            tm.add("reply", t3 - t2)
            tm.add("batches", 0)
    tm.t["peak_mem_gb"] = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    tm.t["model_fp16"] = float(det.fp16)
    stat_q.put(("gpu", dict(tm.t), dict(tm.n)))
    del slots
    shm.close()


# ------------------------------------------------------------------------------------------- main

def list_tasks_from_file(path: Path):
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["mp4"])
    col = tbl.column(0)
    out = []
    for i in range(len(col)):
        v = col[i].as_py()
        out.append((f"{path.name}:{i}", v))
    return out


def reader_thread(files, task_q, out_q, n_workers, info, cfg):
    """Load Parquet files in groups of ~prefetch_bytes, push rows longest-first."""
    t0 = time.perf_counter()
    group, group_bytes = [], 0

    def flush():
        group.sort(key=lambda x: -(len(x[1]) if x[1] else 0))
        for vid, mp4 in group:
            if not mp4:
                out_q.put(("error_line", vid, "empty mp4 bytes"))
                continue
            if task_q.put((vid, mp4)):
                info["pushed"] += 1
                info["vids"].append(vid)
        group.clear()

    for f in files:
        t = time.perf_counter()
        try:
            rows = list_tasks_from_file(f)
        except Exception as e:  # noqa: BLE001
            rows = salvage_rows(f, out_q, e)
        info["read_s"] += time.perf_counter() - t
        for vid, mp4 in rows:
            group.append((vid, mp4))
            group_bytes += len(mp4) if mp4 else 0
        if group_bytes >= cfg.prefetch_bytes:
            flush()
            group_bytes = 0
    flush()
    info["reader_total_s"] = time.perf_counter() - t0
    info["reader_done"] = True
    for _ in range(n_workers + cfg.spare_workers):
        task_q.put(None)


def salvage_rows(path: Path, out_q, err):
    """Unreadable as a whole: read row by row until it breaks."""
    import pyarrow.parquet as pq

    rows = []
    try:
        f = pq.ParquetFile(path)
        idx = 0
        for b in f.iter_batches(batch_size=1, columns=["mp4"]):
            for i in range(b.num_rows):
                rows.append((f"{path.name}:{idx}", b.column(0)[i].as_py()))
                idx += 1
        return rows
    except Exception as e2:  # noqa: BLE001
        n = len(rows)
        out_q.put(("error_line", f"{path.name}:{n}", f"unreadable parquet: {str(e2 or err)[:100]}"))
        return rows


def run(input_dir: str, output: str, cfg: Config, resume: bool = False) -> dict:
    from .io import list_parquet

    ctx = get_context("spawn")
    t_start = time.perf_counter()
    files = list_parquet(input_dir)
    if cfg.dump_dir:
        Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)

    done_vids = set()
    mode = "w"
    if resume and Path(output).exists():
        done_vids = resume_scan(output)
        mode = "a"

    n_total = cfg.workers + cfg.spare_workers
    shm = SharedMemory(create=True, size=n_total * cfg.slots * SLOT_BYTES)
    task_q = ctx.Queue(maxsize=max(2 * cfg.workers, 4))
    req_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(n_total)]
    out_q = ctx.Queue()
    stat_q = ctx.Queue()
    ready = ctx.Event()

    gpu = ctx.Process(target=gpu_main, args=(cfg, req_q, resp_qs, shm.name, n_total, stat_q, ready), daemon=True)
    gpu.start()
    if not ready.wait(timeout=600):
        raise RuntimeError("GPU process failed to start")

    workers: dict[int, object] = {}
    next_wid = 0

    def spawn():
        nonlocal next_wid
        if next_wid >= n_total:
            return
        p = ctx.Process(target=worker_main,
                        args=(next_wid, cfg, task_q, req_q, resp_qs[next_wid], out_q, shm.name, n_total),
                        daemon=True)
        p.start()
        workers[next_wid] = p
        next_wid += 1

    for _ in range(cfg.workers):
        spawn()

    info = {"pushed": 0, "vids": [], "read_s": 0.0, "reader_done": False}
    filt_q = _FilterQueue(task_q, done_vids)
    rt = threading.Thread(target=reader_thread, args=(files, filt_q, out_q, cfg.workers, info, cfg), daemon=True)
    rt.start()

    # Every pushed video is tracked here, independently of worker messages: a worker that dies hard may
    # never report which video it held. Videos left without a result at the end get a crash error line.
    current: dict[int, str] = {}
    finished: set[int] = set()
    accounted: set[str] = set()
    tsum, nsum = defaultdict(float), defaultdict(int)
    n_videos = n_errors = 0
    t_write = 0.0

    def write_error(fh, vid, text):
        fh.write(json.dumps({"video": vid, "error": text}) + "\n")
        fh.flush()

    with open(output, mode, encoding="utf-8") as fh:
        while True:
            if info["reader_done"] and len(accounted) >= info["pushed"]:
                break
            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                for wid, p in list(workers.items()):
                    if wid not in finished and not p.is_alive():
                        finished.add(wid)
                        if p.exitcode != 0:
                            vid = current.pop(wid, None)
                            if vid is not None and vid not in accounted:
                                write_error(fh, vid, f"worker crashed (exit {p.exitcode})")
                                accounted.add(vid)
                                n_videos += 1
                                n_errors += 1
                            spawn()
                alive = [p for w, p in workers.items() if p.is_alive()]
                if info["reader_done"] and not alive:
                    break  # all workers gone; unreported videos handled below
                if not gpu.is_alive():
                    raise RuntimeError("GPU process died")
                continue
            kind = msg[0]
            if kind == "start":
                current[msg[1]] = msg[2]
            elif kind == "error_line":
                if msg[1] in done_vids:  # already written before a resume
                    continue
                t = time.perf_counter()
                fh.write(json.dumps({"video": msg[1], "error": msg[2]}) + "\n")
                fh.flush()
                t_write += time.perf_counter() - t
                n_errors += 1
            elif kind == "video":
                _, wid, vid, lines, error, tt, nn = msg
                current.pop(wid, None)
                if vid in accounted:  # already reported as crashed
                    continue
                accounted.add(vid)
                t = time.perf_counter()
                if lines:
                    fh.write("\n".join(lines) + "\n")
                if error:
                    fh.write(json.dumps({"video": vid, "error": error}) + "\n")
                    n_errors += 1
                fh.flush()
                t_write += time.perf_counter() - t
                n_videos += 1
                for k, v in tt.items():
                    tsum[k] += v
                for k, v in nn.items():
                    nsum[k] += v
        for vid in info["vids"]:
            if vid not in accounted:
                write_error(fh, vid, "worker crashed")
                accounted.add(vid)
                n_videos += 1
                n_errors += 1

    for p in workers.values():
        p.join(timeout=30)
    req_q.put(None)
    gstats = stat_q.get(timeout=60)
    gpu.join(timeout=30)
    shm.close()
    shm.unlink()

    wall = time.perf_counter() - t_start
    frames_dec = nsum.get("frames_decoded", 0)
    frames_inf = int(gstats[2].get("infer", 0))
    report = {
        "wall_s": wall,
        "videos": n_videos,
        "error_lines": n_errors,
        "frames_decoded": frames_dec,
        "frames_inferred": frames_inf,
        "inferred_pct": 100 * frames_inf / max(frames_dec, 1),
        "video_fps": frames_dec / max(wall, 1e-9),
        "workers": cfg.workers,
        "batch": cfg.batch,
        "sampling": cfg.sampling,
        "half": cfg.half,
        "main": {"parquet_read_s": info["read_s"], "write_s": t_write},
        "workers_sum_s": {k: v for k, v in tsum.items()},
        "gpu_s": gstats[1],
        "gpu_mean_batch": frames_inf / max(gstats[2].get("batches", 1), 1),
    }
    return report


class _FilterQueue:
    """Wraps the task queue to skip already-completed videos when resuming."""

    def __init__(self, q, skip):
        self.q, self.skip, self.skipped = q, skip, 0

    def put(self, item) -> bool:
        if item is not None and item[0] in self.skip:
            self.skipped += 1
            return False
        self.q.put(item)
        return True


def resume_scan(output: str) -> set:
    """Videos already written. Each video's lines are written in one go, so only the LAST video in the
    file can be incomplete: its lines (and any half-written line) are removed and it is redone."""
    p = Path(output)
    data = p.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
    lines = data.decode("utf-8").splitlines()
    vids = []
    for line in lines:
        try:
            vids.append(json.loads(line)["video"])
        except Exception:  # noqa: BLE001
            vids.append(None)
    if vids:
        last = vids[-1]
        keep = [ln for ln, v in zip(lines, vids) if v is not None and v != last]
        p.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
        return {v for v in vids if v is not None and v != last}
    return set()
