#!/usr/bin/env python3
"""Self-verification harness for ale-47 (QKP).

For seeds 1..20: generate instance, run sol, score it, and compare against
two trivial baselines:
  * EMPTY: the empty subset -> objective 0 -> score 0.
  * RATIO-GREEDY: this is exactly the scorer's reference G -> score 1_000_000
    for a feasible run (so "beating the trivial baseline" means mean score
    strictly > 1_000_000, i.e. the solver collects more synergy than the
    synergy-blind greedy).
Reports per-seed feasibility (score > 0 and parses) and the solver vs baseline
means, and asserts the solver strictly beats the baseline.
"""
import subprocess, sys, os, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen.py")
SCORE = os.path.join(HERE, "score.py")
SOL = os.path.join(HERE, "sol")


def run(seeds):
    sol_scores = []
    empty_scores = []
    greedy_scores = []
    all_feasible = True
    for s in seeds:
        inst = subprocess.run([sys.executable, GEN, str(s)], capture_output=True, text=True).stdout
        with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as fi:
            fi.write(inst); inst_path = fi.name
        # solver
        out = subprocess.run([SOL], input=inst, capture_output=True, text=True).stdout
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fo:
            fo.write(out); out_path = fo.name
        sc = int(subprocess.run([sys.executable, SCORE, inst_path, out_path],
                                capture_output=True, text=True).stdout.strip())
        # empty baseline
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fe:
            fe.write("0\n"); empty_path = fe.name
        ec = int(subprocess.run([sys.executable, SCORE, inst_path, empty_path],
                                capture_output=True, text=True).stdout.strip())
        # greedy baseline == reference G == 1_000_000 by construction; recompute
        # by feeding the scorer's own greedy as a "solution" is not necessary:
        # G yields 1_000_000 by definition. We just record the canonical value.
        gc = 1_000_000
        feas = sc > 0
        if not feas:
            all_feasible = False
        sol_scores.append(sc); empty_scores.append(ec); greedy_scores.append(gc)
        print(f"seed {s:2d}: sol={sc:>9d}  empty={ec:>2d}  greedy={gc:>9d}  feasible={feas}")
        os.unlink(inst_path); os.unlink(out_path); os.unlink(empty_path)

    n = len(seeds)
    msol = sum(sol_scores) / n
    mempty = sum(empty_scores) / n
    mgreedy = sum(greedy_scores) / n
    print("-" * 60)
    print(f"mean sol   = {msol:.1f}")
    print(f"mean empty = {mempty:.1f}")
    print(f"mean greedy(ref G) = {mgreedy:.1f}")
    print(f"all_feasible = {all_feasible}")
    beats_empty = msol > mempty
    beats_greedy = msol > mgreedy
    print(f"beats EMPTY baseline   : {beats_empty}")
    print(f"beats RATIO-GREEDY (G) : {beats_greedy}")
    ok = all_feasible and beats_empty and beats_greedy
    print(f"OVERALL OK = {ok}")
    return ok


if __name__ == "__main__":
    seeds = list(range(1, 21))
    ok = run(seeds)
    sys.exit(0 if ok else 1)
