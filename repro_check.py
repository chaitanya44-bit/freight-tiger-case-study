"""Runs the whole pipeline 3 times (each with a different PYTHONHASHSEED) and compares file hashes."""
import hashlib
import os
import subprocess
import sys
import tempfile

FILES = ["flagged_output.csv", "weekly_metrics.csv"]
hashes = []
for i, seed in enumerate(["1", "42", "999"], 1):
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, "run.py", "--out", out], check=True, capture_output=True,
                   env={**os.environ, "PYTHONHASHSEED": seed})
    hashes.append({f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in FILES})
    print(f"run {i} (PYTHONHASHSEED={seed}): " + ", ".join(f"{f[:12]}..={h[:12]}" for f, h in hashes[-1].items()))

same = all(h == hashes[0] for h in hashes)
msg = "IDENTICAL: 3 runs produced byte-identical flagged_output.csv and weekly_metrics.csv" if same else "MISMATCH between runs!"
print(msg)
with open("outputs/repro_log.txt", "w") as f:
    for i, h in enumerate(hashes, 1):
        f.write(f"run {i}: " + " ".join(f"{k}={v}" for k, v in h.items()) + "\n")
    f.write(msg + "\n")
sys.exit(0 if same else 1)
