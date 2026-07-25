#!/usr/bin/env python3
import subprocess, sys, random
SOL = "/tmp/cpv4b-dp-bitmask-construct-verify_sol"
HERE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-bitmask-construct-verify/verify"
sys.path.insert(0, HERE)
import brute

def gen(seed):
    random.seed(seed)
    L = random.randint(1, 5)
    n = random.randint(0, 16)
    universe = 1 << L
    m = random.randint(0, universe)
    pats = random.sample(range(universe), m)
    forb = [format(p, '0' + str(L) + 'b') for p in pats]
    return n, L, forb

def run_sol(inp):
    r = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return r.stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mism = 0; ex = []; feas = 0; infe = 0
    for seed in range(base + 1, base + N + 1):
        n, L, forb = gen(seed)
        inp = f"{n} {L} {len(forb)}\n" + "".join(f + "\n" for f in forb)
        got = run_sol(inp)
        exp = brute.solve(n, L, set(forb))
        if exp == "-1": infe += 1
        else: feas += 1
        if got != exp:
            mism += 1
            if len(ex) < 20: ex.append((seed, (n, L, forb), got, exp))
    print(f"TOTAL={N} MISMATCH={mism} feasible={feas} infeasible={infe}")
    for e in ex: print("MISMATCH", e)

if __name__ == "__main__":
    main()
