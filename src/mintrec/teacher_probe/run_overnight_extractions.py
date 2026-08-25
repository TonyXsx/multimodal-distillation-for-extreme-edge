"""
Runs the local extractions back to back overnight. The 6.4 GB GPU only fits one
at a time.

Every variant is resumable, so a non-zero exit just gets retried and it picks up
from the last shard. If one keeps failing it gets given up on and the next
starts, so a bad run can't block the rest. Per-sample failures are already
skipped inside the extractor.

Cheapest first, so at least one variant is definitely finished early:
    1. ta_plain    audio+transcript, no video       ~5 h
    2. tva_aware   text+video+audio, aware prompt   ~8-10 h
    3. tva_plain   text+video+audio, plain prompt   ~8-10 h

overnight.log gets one line per attempt, each variant gets its own log with the
full output.
"""

import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "src" / "mintrec" / "teacher_probe" / "extract_features_local.py"
LOGDIR = ROOT / "outputs" / "mintrec"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "overnight.log"

VARIANTS = [
    ("ta_plain",  ["--split", "all", "--modalities", "ta"]),
    ("tva_aware", ["--split", "all", "--prompt", "aware"]),
    ("tva_plain", ["--split", "all"]),
]
MAX_RETRIES = 8
RETRY_WAIT  = 60


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(MASTER, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log(f"=== overnight runner start | python={sys.executable} ===")
    for name, args in VARIANTS:
        vlog = LOGDIR / f"{name}.log"
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            log(f"{name}: attempt {attempt}/{MAX_RETRIES} -> {' '.join(args)}")
            with open(vlog, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n===== attempt {attempt} @ {datetime.datetime.now()} =====\n")
                f.flush()
                rc = subprocess.run([sys.executable, str(SCRIPT), *args],
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                log(f"{name}: DONE (rc=0)")
                ok = True
                break
            log(f"{name}: exit rc={rc}; retry in {RETRY_WAIT}s (resume-safe)")
            time.sleep(RETRY_WAIT)
        if not ok:
            log(f"{name}: GAVE UP after {MAX_RETRIES} attempts — moving on")
    log("=== overnight runner finished: all variants processed ===")


if __name__ == "__main__":
    main()
