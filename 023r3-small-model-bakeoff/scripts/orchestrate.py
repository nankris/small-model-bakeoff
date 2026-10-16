#!/usr/bin/env python3
"""Allocate a Colab GPU, run Qwen in resumable batches, download after each batch, stop.

Session name is r3023. This script never stops any other session.
CPU Colab, then a note to the caller, is the fallback if L4, T4 and A100 all fail.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SESSION = "r3023"
PY = sys.executable
REMOTE = ROOT / "scripts" / "remote_infer.py"
CU_CAP = 4.6
FALLBACK_RATE = {"L4": 1.54, "T4": 0.6, "A100": 4.5, "CPU": 0.2}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " " + msg
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "orchestrate.log").open("a") as f:
        f.write(line + "\n")


def run(cmd, timeout=120):
    log("CMD " + " ".join(cmd))
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        log(f"TIMEOUT {cmd[1] if len(cmd)>1 else cmd} {err[-300:]}")
        return subprocess.CompletedProcess(cmd, 124, out, err or "TIMEOUT")
    text = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    if text:
        log(text[-2000:])
    log(f"RC {p.returncode}")
    return p


def parse_usage(text):
    bal = re.search(r"balance:\s*([0-9.]+)", text or "")
    rate = re.search(r"rate:\s*([0-9.]+)", text or "")
    active = re.search(r"assignments:\s*([0-9]+)", text or "")
    return {
        "balance": float(bal.group(1)) if bal else None,
        "rate": float(rate.group(1)) if rate else None,
        "active": int(active.group(1)) if active else None,
        "raw": text or "",
        "ts": time.time(),
    }


def usage():
    p = run(["colab", "usage"], timeout=90)
    snap = parse_usage((p.stdout or "") + "\n" + (p.stderr or ""))
    with (RESULTS / "usage_log.jsonl").open("a") as f:
        f.write(json.dumps({k: snap[k] for k in ("balance", "rate", "active", "ts")}) + "\n")
    return snap


def load_ids(path):
    ids = {}
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("id") and rec.get("text"):
            ids[rec["id"]] = rec
    return ids


def merge_into(dest, extra_paths):
    rows = load_ids(dest)
    order = list(rows)
    for path in extra_paths:
        for iid, rec in load_ids(path).items():
            if iid not in rows:
                order.append(iid)
            rows[iid] = rec
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for iid in order:
            f.write(json.dumps(rows[iid], ensure_ascii=False) + "\n")
    return rows


def stream_exec(env_pairs, timeout):
    """Run remote_infer. Parse RESULT/META lines into local files as they arrive."""
    cmd = ["colab", "exec", "-s", SESSION, "-f", str(REMOTE), "--timeout", str(timeout)]
    for kv in env_pairs:
        cmd.extend(["--env", kv])
    log("CMD " + " ".join(cmd))
    live = RESULTS / "qwen_live.jsonl"
    meta_path = RESULTS / "qwen_meta.json"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    metas = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            with (RESULTS / "remote_stdout.log").open("a") as f:
                f.write(line)
            if "RESULT " in line:
                raw = line[line.find("RESULT ") + 7:].strip()
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with live.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if "META " in line:
                raw = line[line.find("META ") + 5:].strip()
                try:
                    metas.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = 124
        log("exec wrapper timeout")
    if metas:
        meta_path.write_text(json.dumps(metas[-1], indent=2) + "\n")
    return rc


def session_alive():
    p = run(["colab", "sessions"], timeout=90)
    blob = (p.stdout or "") + "\n" + (p.stderr or "")
    return f"[{SESSION}]" in blob or f"{SESSION} " in blob


def allocate():
    plan = [("L4", 15), ("T4", 5), ("A100", 5)]
    for gpu, tries in plan:
        for i in range(tries):
            snap = usage()
            if snap["balance"] is not None and snap["balance"] < 50:
                raise SystemExit(f"balance {snap['balance']} < 50 CU, not starting a session")
            active = snap["active"] if snap["active"] is not None else 0
            if active >= 2:
                log(f"{gpu} attempt {i+1}/{tries}: {active} assignments already active, waiting 300s")
                if i < tries - 1:
                    time.sleep(300)
                continue
            p = run(["colab", "new", "--gpu", gpu, "-s", SESSION], timeout=600)
            blob = ((p.stdout or "") + "\n" + (p.stderr or "")).lower()
            refused = "refused" in blob or "precondition failed" in blob
            if (p.returncode == 0 and not refused) or (session_alive() and not refused):
                log(f"allocated {gpu}")
                return gpu
            log(f"{gpu} attempt {i+1}/{tries} failed rc={p.returncode}")
            # A half-created session still burns CU. Stop only our name before waiting.
            run(["colab", "stop", "-s", SESSION], timeout=120)
            if i < tries - 1:
                log("sleeping 300s before retry")
                time.sleep(300)
        log(f"{gpu} exhausted")
    log("GPU allocation failed, trying CPU runtime")
    p = run(["colab", "new", "-s", SESSION], timeout=600)
    if p.returncode == 0 or session_alive():
        return "CPU"
    raise SystemExit("no Colab runtime could be allocated")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT / "items.json"))
    args = ap.parse_args()
    items_path = pathlib.Path(args.items)
    if not items_path.is_absolute():
        items_path = ROOT / items_path
    items = json.loads(items_path.read_text())
    wanted = [it["id"] for it in items]
    RESULTS.mkdir(parents=True, exist_ok=True)
    before = usage()
    (RESULTS / "usage_before.json").write_text(json.dumps(before, indent=2) + "\n")
    log(f"before balance={before['balance']} rate={before['rate']} active={before['active']}")
    started = False
    gpu = None
    t_alloc = None
    incremental = None
    reason = "ok"
    try:
        gpu = allocate()
        started = True
        t_alloc = time.time()
        after = usage()
        base = before["rate"] or 0
        now_rate = after["rate"] if after["rate"] is not None else base
        incremental = now_rate - base
        if incremental < 0.15:
            incremental = FALLBACK_RATE.get(gpu, 1.54)
            rate_source = "fallback_table"
        else:
            rate_source = "usage_delta"
        log(f"gpu={gpu} incremental_cu_per_h={incremental:.3f} source={rate_source} balance={after['balance']}")
        deadline = t_alloc + (CU_CAP / incremental) * 3600
        # Keep judgement time. Hard cap the runtime at 150 minutes as well.
        deadline = min(deadline, t_alloc + 150 * 60)
        run(["colab", "upload", "-s", SESSION, str(items_path), "/content/items.json"], timeout=180)
        dest = RESULTS / "qwen.jsonl"
        batch_i = 0
        while True:
            have = load_ids(dest)
            missing = [i for i in wanted if i not in have]
            if not missing:
                log(f"all {len(wanted)} qwen answers present")
                reason = "complete"
                break
            if time.time() > deadline:
                reason = "cu_or_time_budget"
                log(f"budget stop with {len(missing)} missing")
                break
            elapsed_h = (time.time() - t_alloc) / 3600
            est = elapsed_h * incremental
            log(f"batch {batch_i} missing {len(missing)} est_cu {est:.3f}")
            if est > CU_CAP:
                reason = "cu_or_time_budget"
                log("estimated CU over cap before batch")
                break
            if dest.exists() and dest.stat().st_size > 0:
                run(["colab", "upload", "-s", SESSION, str(dest), "/content/answers.jsonl"], timeout=180)
            batch_size = "20" if batch_i == 0 else "50"
            # First batch includes model download, so allow a long timeout.
            timeout = 1800 if batch_i == 0 else 1200
            rc = stream_exec(
                [
                    "ITEMS_PATH=/content/items.json",
                    "ANSWERS_PATH=/content/answers.jsonl",
                    f"BATCH_SIZE={batch_size}",
                    "PYTHONUNBUFFERED=1",
                    "MODEL_REPO=Qwen/Qwen3-4B",
                ],
                timeout,
            )
            dl = RESULTS / "qwen_download.jsonl"
            dlp = run(["colab", "download", "-s", SESSION, "/content/answers.jsonl", str(dl)], timeout=180)
            if dlp.returncode != 0:
                log("download failed, relying on streamed RESULT lines")
            merge_into(dest, [RESULTS / "qwen_live.jsonl", dl])
            have = load_ids(dest)
            new_missing = [i for i in wanted if i not in have]
            log(f"after batch {batch_i} rc={rc} have {len(have)} still_missing {len(new_missing)}")
            if batch_i == 0:
                texts = [have[i]["text"] for i in wanted if i in have]
                if texts and (sum(len(t) for t in texts) / len(texts)) < 2:
                    reason = "empty_generations"
                    log("first batch texts are empty, stopping")
                    break
                for rec in list(have.values())[:3]:
                    log(f"SAMPLE {rec['id']} {rec.get('model')} {rec['text'][:180]!r}")
            if len(new_missing) >= len(missing):
                log("batch made no progress")
                if batch_i > 2:
                    reason = "no_progress"
                    break
            batch_i += 1
            if batch_i > 20:
                reason = "too_many_batches"
                break
            # Drop a cached model before the next load so a persistent kernel cannot OOM.
            run(["colab", "restart-kernel", "-s", SESSION], timeout=180)
    except BaseException as e:
        if reason == "ok":
            reason = f"exception {type(e).__name__}: {e}"
        raise
    finally:
        log("stopping session " + SESSION)
        run(["colab", "stop", "-s", SESSION], timeout=180)
        time.sleep(3)
        run(["colab", "sessions"], timeout=60)
        after_stop = usage()
        elapsed = None if t_alloc is None else round(time.time() - t_alloc, 1)
        cu = None
        if elapsed is not None and incremental is not None:
            cu = round((elapsed / 3600) * incremental, 4)
        record = {
            "session": SESSION,
            "gpu_requested_order": ["L4", "T4", "A100", "CPU"],
            "gpu_allocated": gpu,
            "started": started,
            "elapsed_s": elapsed,
            "cu_rate_per_h": incremental,
            "cu_estimated": cu,
            "cu_cap": CU_CAP,
            "reason": reason,
            "balance_before": before.get("balance"),
            "balance_after": after_stop.get("balance"),
            "rate_before": before.get("rate"),
            "rate_after": after_stop.get("rate"),
            "active_after": after_stop.get("active"),
            "ts": time.time(),
        }
        with (RESULTS / "sessions.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        # Aggregate every session this experiment started.
        total = 0.0
        rows = []
        for line in (RESULTS / "sessions.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
                if rows[-1].get("cu_estimated"):
                    total += rows[-1]["cu_estimated"]
        meta = {}
        if (RESULTS / "qwen_meta.json").exists():
            meta = json.loads((RESULTS / "qwen_meta.json").read_text())
        summary = {
            "sessions": rows,
            "cu_estimated_total": round(total, 4),
            "gpu_allocated": gpu,
            "qwen_meta": meta,
            "balance_before_first": rows[0].get("balance_before") if rows else None,
            "balance_after_last": after_stop.get("balance"),
            "reason": reason,
        }
        (RESULTS / "compute.json").write_text(json.dumps(summary, indent=2) + "\n")
        log(f"compute {json.dumps(summary)}")
    have = load_ids(RESULTS / "qwen.jsonl")
    missing = [i for i in wanted if i not in have]
    log(f"final have {len(have)} missing {len(missing)} reason {reason}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
