#!/usr/bin/env python3
"""Run stage 1 (60/task). If the preregistered rule is inconclusive and CU remains, run stage 2."""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = ROOT / "results"


def launch(script, args, log_name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    logf = open(RESULTS / log_name, "a")
    logf.write(f"\n--- start {time.time()} {script} {args}\n")
    logf.flush()
    return subprocess.Popen([PY, str(ROOT / "scripts" / script), *args], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT), logf


def run_blocking(script, args, log_name):
    proc, logf = launch(script, args, log_name)
    rc = proc.wait()
    logf.write(f"\n--- end rc={rc}\n")
    logf.close()
    return rc


def cu_so_far():
    path = RESULTS / "compute.json"
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text())
    return float(data.get("cu_estimated_total") or 0)


def verdict():
    path = RESULTS / "summary.json"
    if not path.exists():
        return {"verdict": "STILL INCONCLUSIVE", "rule_applied": False}
    return json.loads(path.read_text())


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "run_started.txt").write_text(str(time.time()) + "\n")
    # Stage 1 answers in parallel.
    grok, glog = launch("grok_answers.py", ["items.json"], "grok_run.log")
    orch, olog = launch("orchestrate.py", ["--items", "items.json"], "orch_run.log")
    grc = grok.wait()
    orc = orch.wait()
    glog.close()
    olog.close()
    print(f"stage1 answers grok_rc={grc} orch_rc={orc} cu={cu_so_far()}", flush=True)
    if grc != 0:
        print("retrying grok once", flush=True)
        grc = run_blocking("grok_answers.py", ["items.json"], "grok_run.log")
        print(f"grok retry rc={grc}", flush=True)
    jrc = run_blocking("judge.py", ["items.json"], "judge_run.log")
    if jrc != 0:
        print("retrying judge once", flush=True)
        jrc = run_blocking("judge.py", ["items.json"], "judge_run.log")
    arc = run_blocking("analyse.py", ["items.json"], "analyse_run.log")
    summary = verdict()
    print("stage1", summary.get("verdict"), summary.get("rule_applied"), "cu", cu_so_far(), flush=True)
    started = float((RESULTS / "run_started.txt").read_text().strip())
    elapsed_h = (time.time() - started) / 3600
    need_ext = (
        summary.get("verdict") == "STILL INCONCLUSIVE"
        and summary.get("rule_applied") is True
        and cu_so_far() < 3.0
        and elapsed_h < 2.5
    )
    if not need_ext:
        print("stop after stage 1", flush=True)
        return 0 if arc == 0 else arc
    print("stage 2 extension items 60..89", flush=True)
    grc = run_blocking("grok_answers.py", ["items_all.json"], "grok_run.log")
    if grc != 0:
        grc = run_blocking("grok_answers.py", ["items_all.json"], "grok_run.log")
    orc = run_blocking("orchestrate.py", ["--items", "items_all.json"], "orch_run.log")
    print(f"stage2 answers grok_rc={grc} orch_rc={orc} cu={cu_so_far()}", flush=True)
    jrc = run_blocking("judge.py", ["items_all.json"], "judge_run.log")
    if jrc != 0:
        jrc = run_blocking("judge.py", ["items_all.json"], "judge_run.log")
    arc = run_blocking("analyse.py", ["items_all.json"], "analyse_run.log")
    summary = verdict()
    print("stage2", summary.get("verdict"), "cu", cu_so_far(), flush=True)
    return 0 if arc == 0 else arc


if __name__ == "__main__":
    sys.exit(main())
