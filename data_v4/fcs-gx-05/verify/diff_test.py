#!/usr/bin/env python3
# Differential tester: compare sol (C++) vs brute (Kuhn matching) on many cases.
# - Feasibility verdict (YES/NO) must match.
# - When sol says YES, validate sol's assignment: every request gets a distinct
#   slot inside its interval.
import sys, subprocess, random

SOL = "/tmp/fcs-gx-05_x"
BRUTE = "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-gx-05/verify/brute.py"
GEN = "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-gx-05/verify/gen.py"

def parse_input(inp):
    t = inp.split()
    n = int(t[0]); T = int(t[1])
    iv = []
    k = 2
    for i in range(n):
        l = int(t[k]); r = int(t[k+1]); k += 2
        iv.append((l, r))
    return n, T, iv

def validate_assignment(n, T, iv, line):
    toks = line.split()
    if len(toks) != n:
        return False, f"assignment has {len(toks)} tokens, expected {n}"
    used = set()
    for i, tk in enumerate(toks):
        s = int(tk)
        l, r = iv[i]
        if not (l <= s <= r):
            return False, f"request {i} slot {s} not in [{l},{r}]"
        if s in used:
            return False, f"slot {s} used twice"
        used.add(s)
    return True, ""

def run_case(inp):
    n, T, iv = parse_input(inp)
    sol = subprocess.run([SOL], input=inp, capture_output=True, text=True, timeout=20).stdout
    bru = subprocess.run(["python3", BRUTE], input=inp, capture_output=True, text=True, timeout=60).stdout
    sol_lines = sol.split("\n")
    bru_lines = bru.split("\n")
    sol_verdict = sol_lines[0].strip() if sol_lines else ""
    bru_verdict = bru_lines[0].strip() if bru_lines else ""
    if sol_verdict != bru_verdict:
        return False, f"verdict mismatch: sol={sol_verdict} brute={bru_verdict}"
    if sol_verdict == "YES":
        asg = sol_lines[1] if len(sol_lines) > 1 else ""
        ok, msg = validate_assignment(n, T, iv, asg)
        if not ok:
            return False, f"invalid sol assignment: {msg}"
    elif sol_verdict not in ("YES", "NO"):
        return False, f"bad verdict token: {sol_verdict!r}"
    return True, ""

def main():
    ncases = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    fails = 0
    for seed in range(ncases):
        inp = subprocess.run(["python3", GEN, str(seed)], capture_output=True, text=True).stdout
        ok, msg = run_case(inp)
        if not ok:
            fails += 1
            print(f"MISMATCH seed={seed}: {msg}")
            print("INPUT:")
            print(inp)
            if fails >= 10:
                print("too many fails, stopping")
                break
    print(f"done: {ncases} cases, {fails} mismatches")

if __name__ == "__main__":
    main()
