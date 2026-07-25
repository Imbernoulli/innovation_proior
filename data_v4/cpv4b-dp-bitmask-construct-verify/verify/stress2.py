#!/usr/bin/env python3
# Heavier stress: wider w, more cases, including w up to 8 (brute limited by n).
import subprocess, sys, random
SOL = "/tmp/cpv4b-dp-bitmask-construct-verify_sol"
HERE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-bitmask-construct-verify/verify"
sys.path.insert(0, HERE)
import brute

def run_sol(inp):
    r = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return r.stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 12345)
    mism = 0; ex = []
    feasible = 0; infeasible = 0
    for it in range(N):
        w = random.randint(1, 8)
        n = random.randint(0, 16)
        a = random.randint(0, w); b = random.randint(0, w)
        lo, hi = min(a, b), max(a, b)
        inp = f"{n} {w} {lo} {hi}\n"
        got = run_sol(inp)
        exp = brute.solve(n, w, lo, hi)
        if exp == "-1": infeasible += 1
        else: feasible += 1
        if got != exp:
            mism += 1
            if len(ex) < 20: ex.append(((n,w,lo,hi), got, exp))
    print(f"TOTAL={N} MISMATCH={mism} feasible={feasible} infeasible={infeasible}")
    for e in ex: print("MISMATCH", e)

if __name__ == "__main__":
    main()
