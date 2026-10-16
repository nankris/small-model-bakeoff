#!/usr/bin/env python3
"""Resumable Grok answers. One JSON line per item, flushed immediately.

Usage: grok_answers.py [items.json]
Skips ids already present in results/grok.jsonl.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "grok.jsonl"
WORKERS = 4
TIMEOUT = 180
RETRIES = 3

CMD_TAIL = [
    "--output-format", "json",
    "--disable-web-search",
    "--no-subagents",
    "--max-turns", "1",
    "--reasoning-effort", "low",
    "--system-prompt-override", "You are a helpful assistant. Follow the output format exactly.",
    "--tools", "",
]

LOCK = threading.Lock()


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("text") and rec.get("exit", 0) == 0:
                done[rec["id"]] = rec
    return done


def parse_grok(stdout):
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i = text.find("{")
        if i < 0:
            return {}
        return json.loads(text[i:])


def append(rec):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with LOCK:
        with OUT.open("a") as f:
            f.write(line)
            f.flush()


def call_one(item):
    last = None
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="grok023r3-") as wd:
            cmd = ["grok", "--no-auto-update", "-p", item["prompt"], *CMD_TAIL]
            try:
                p = subprocess.run(
                    cmd, cwd=wd, capture_output=True, text=True, timeout=TIMEOUT,
                )
                data = {}
                err = (p.stderr or "")[-400:]
                try:
                    data = parse_grok(p.stdout or "")
                except Exception as e:
                    err = f"parse {e}; " + err
                text = (data.get("text") or "").strip()
                usage = data.get("usage") or {}
                rec = dict(
                    model="grok",
                    id=item["id"],
                    task=item["task"],
                    text=text,
                    exit=p.returncode,
                    wall_s=round(time.time() - t0, 3),
                    attempt=attempt,
                    cost_usd=data.get("total_cost_usd"),
                    output_tokens=usage.get("output_tokens"),
                    reasoning_tokens=usage.get("reasoning_tokens"),
                    input_tokens=usage.get("input_tokens"),
                    model_name=",".join((data.get("modelUsage") or {}).keys()),
                )
                if p.returncode != 0 or not text:
                    rec["stderr"] = err
                    last = rec
                    time.sleep(min(60, 5 * attempt))
                    continue
                return rec
            except subprocess.TimeoutExpired:
                last = dict(
                    model="grok", id=item["id"], task=item["task"], text="",
                    exit=-9, wall_s=round(time.time() - t0, 3), attempt=attempt,
                    stderr="TIMEOUT",
                )
                time.sleep(min(60, 5 * attempt))
    return last


def main():
    items_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "items.json"
    if not items_path.is_absolute():
        items_path = ROOT / items_path
    items = json.loads(items_path.read_text())
    done = load_done(OUT)
    todo = [it for it in items if it["id"] not in done]
    print(f"grok: {len(done)} done, {len(todo)} todo from {items_path.name}", flush=True)
    if not todo:
        return 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ok = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(call_one, it): it["id"] for it in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            append(rec)
            ok += int(bool(rec.get("text")) and rec.get("exit", 1) == 0)
            print(f"grok {rec['id']} exit={rec['exit']} chars={len(rec.get('text') or '')} ${rec.get('cost_usd')}", flush=True)
    print(f"grok batch finished, new_ok={ok}", flush=True)
    # Return nonzero if any requested id is still missing a good answer.
    done = load_done(OUT)
    missing = [it["id"] for it in items if it["id"] not in done]
    if missing:
        print(f"grok missing {len(missing)}: {missing[:8]}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
