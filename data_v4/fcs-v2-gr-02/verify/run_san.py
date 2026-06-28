#!/usr/bin/env python3
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen.py")
SAN = "/tmp/fcs_san"
bad = 0
for seed in range(1, 401):
    g = subprocess.run([sys.executable, GEN, str(seed)], capture_output=True, text=True)
    p = subprocess.run([SAN], input=g.stdout, capture_output=True, text=True)
    if p.returncode != 0 or "runtime error" in p.stderr or "ERROR" in p.stderr:
        print("SANITIZER ISSUE seed", seed, "rc", p.returncode)
        print(p.stderr[:600])
        print("INPUT:\n", g.stdout)
        bad += 1
        if bad >= 3: break
print("sanitizer done, issues=", bad)
