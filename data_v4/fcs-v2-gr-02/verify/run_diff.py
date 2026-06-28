#!/usr/bin/env python3
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = "/tmp/fcs-v2-gr-02_x"
GEN = os.path.join(HERE, "gen.py")
BRUTE = os.path.join(HERE, "brute.py")

def run_sol(inp):
    p = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"sol crashed rc={p.returncode} stderr={p.stderr[:200]}"
    return p.stdout.strip(), None

def run_brute(inp):
    p = subprocess.run([sys.executable, BRUTE], input=inp, capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"brute crashed rc={p.returncode} stderr={p.stderr[:200]}"
    return p.stdout.strip(), None

def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    mism = 0
    for seed in range(lo, hi + 1):
        g = subprocess.run([sys.executable, GEN, str(seed)], capture_output=True, text=True)
        inp = g.stdout
        a, ea = run_sol(inp)
        b, eb = run_brute(inp)
        if ea or eb:
            print(f"ERROR seed={seed} solerr={ea} bruteerr={eb}")
            print("INPUT:\n" + inp)
            mism += 1
            if mism >= 6: break
            continue
        if a != b:
            print(f"MISMATCH seed={seed}")
            print("INPUT:\n" + inp)
            print("SOL  :", a)
            print("BRUTE:", b)
            mism += 1
            if mism >= 6: break
    print(f"DONE range [{lo},{hi}] mismatches/errors={mism}")

if __name__ == "__main__":
    main()
