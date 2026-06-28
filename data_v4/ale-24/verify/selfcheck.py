#!/usr/bin/env python3
import os, subprocess, sys, statistics, time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen.py")
SCORE = os.path.join(HERE, "score.py")
SOL = os.path.join(HERE, "sol")
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)

def run(seeds):
    sol_scores = []
    rr_scores = []          # round-robin trivial baseline scored by score.py
    all_feasible = True
    for sd in seeds:
        inst = os.path.join(TMP, f"in_{sd}.txt")
        with open(inst, "w") as f:
            subprocess.run([sys.executable, GEN, str(sd)], stdout=f, check=True)
        # solver
        out = os.path.join(TMP, f"sol_{sd}.txt")
        t0 = time.time()
        with open(inst) as fi, open(out, "w") as fo:
            subprocess.run([SOL], stdin=fi, stdout=fo, check=True)
        dt = time.time() - t0
        sc = int(subprocess.run([sys.executable, SCORE, inst, out],
                                capture_output=True, text=True, check=True).stdout.strip())
        if sc <= 0:
            all_feasible = False
        sol_scores.append(sc)
        # trivial baseline: round-robin schedule (j -> machine j%M, input order)
        with open(inst) as f:
            toks = f.read().split()
        n = int(toks[0]); Mm = int(toks[1])
        machines = [[] for _ in range(Mm)]
        for j in range(n):
            machines[j % Mm].append(j)
        rrout = os.path.join(TMP, f"rr_{sd}.txt")
        with open(rrout, "w") as f:
            for mm in machines:
                f.write(str(len(mm)) + ("".join(" " + str(j) for j in mm)) + "\n")
        rrsc = int(subprocess.run([sys.executable, SCORE, inst, rrout],
                                  capture_output=True, text=True, check=True).stdout.strip())
        rr_scores.append(rrsc)
        print(f"seed {sd:3d}: n={n:3d} M={Mm}  sol={sc:8d}  rr_baseline={rrsc:8d}  "
              f"feasible={'Y' if sc>0 else 'N'}  t={dt:.2f}s")
    print("-" * 70)
    print(f"solver mean   = {statistics.mean(sol_scores):.1f}")
    print(f"baseline mean = {statistics.mean(rr_scores):.1f}")
    print(f"all feasible  = {all_feasible}")
    print(f"beats baseline= {statistics.mean(sol_scores) > statistics.mean(rr_scores)}")
    print(f"min sol score = {min(sol_scores)}")
    return all_feasible, statistics.mean(sol_scores), statistics.mean(rr_scores)

if __name__ == "__main__":
    seeds = list(range(1, 21))
    run(seeds)
