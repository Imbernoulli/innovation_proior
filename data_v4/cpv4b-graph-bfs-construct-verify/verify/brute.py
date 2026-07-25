#!/usr/bin/env python3
"""
Independent brute-force ORACLE for the 2-shift (2-coloring) construction problem.

Problem: given an undirected graph on n vertices (1..n) and m edges, assign each
vertex a shift label in {1,2} so that every edge joins two DIFFERENT shifts.
Output the n labels, or -1 if no such assignment exists (graph not bipartite).

This oracle decides FEASIBILITY independently by brute force:
  - small n: try all 2^n labelings, check every edge.
  - it does NOT reproduce a particular coloring; instead it is used by the
    checker as ground truth for "feasible?" and to validate the candidate's
    own coloring against the edges.

Usage:
  python3 brute.py < input        -> prints "POSSIBLE" or "IMPOSSIBLE"
  python3 brute.py --check input candidate_output
                                  -> prints "OK" or an error, exit code 0/1
"""
import sys


def read_graph(text):
    toks = text.split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    edges = []
    for _ in range(m):
        u = int(toks[idx]); idx += 1
        v = int(toks[idx]); idx += 1
        edges.append((u, v))
    return n, m, edges


def feasible_bruteforce(n, edges):
    # try every labeling in {0,1}^n
    for mask in range(1 << n):
        good = True
        for (u, v) in edges:
            cu = (mask >> (u - 1)) & 1
            cv = (mask >> (v - 1)) & 1
            if cu == cv:
                good = False
                break
        if good:
            return True
    return False


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        with open(sys.argv[2]) as f:
            inp = f.read()
        with open(sys.argv[3]) as f:
            outp = f.read().split()
        n, m, edges = read_graph(inp)
        feas = feasible_bruteforce(n, edges)
        if len(outp) == 1 and outp[0] == "-1":
            if feas:
                print("ERROR: said -1 but a valid assignment exists")
                sys.exit(1)
            print("OK")
            sys.exit(0)
        # candidate claims a coloring
        if not feas:
            print("ERROR: gave a coloring but problem is infeasible")
            sys.exit(1)
        if len(outp) != n:
            print(f"ERROR: expected {n} labels, got {len(outp)}")
            sys.exit(1)
        labels = []
        for t in outp:
            if t not in ("1", "2"):
                print(f"ERROR: label {t} not in {{1,2}}")
                sys.exit(1)
            labels.append(int(t))
        for (u, v) in edges:
            if labels[u - 1] == labels[v - 1]:
                print(f"ERROR: edge ({u},{v}) monochromatic")
                sys.exit(1)
        print("OK")
        sys.exit(0)
    else:
        inp = sys.stdin.read()
        n, m, edges = read_graph(inp)
        print("POSSIBLE" if feasible_bruteforce(n, edges) else "IMPOSSIBLE")


if __name__ == "__main__":
    main()
