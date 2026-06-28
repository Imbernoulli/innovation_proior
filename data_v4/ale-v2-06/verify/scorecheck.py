#!/usr/bin/env python3
"""Independent checks of the scorer feasibility floor + baseline normalization."""
import sys, os, subprocess, tempfile
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import score

PY = sys.executable
inst = os.path.join(tempfile.mkdtemp(), "inst.txt")
with open(inst, "w") as f:
    subprocess.run([PY, os.path.join(D, "gen.py"), "1"], stdout=f, check=True)

n, m, w, adj, edges = score.read_instance(inst)


def run_score(sol_text):
    p = os.path.join(tempfile.mkdtemp(), "s.txt")
    with open(p, "w") as f:
        f.write(sol_text)
    out = subprocess.run([PY, os.path.join(D, "score.py"), inst, p],
                         capture_output=True, text=True)
    return int(out.stdout.strip())


# 1) the scorer's own greedy set must score exactly 1_000_000
avail = [True] * n
order = sorted(range(n), key=lambda i: (-w[i], i))
sel = []
for v in order:
    if not avail[v]:
        continue
    sel.append(v); avail[v] = False
    for u in adj[v]:
        avail[u] = False
greedy_txt = str(len(sel)) + "\n" + "\n".join(map(str, sel)) + "\n"
print("greedy set size", len(sel), "weight", sum(w[v] for v in sel))
print("greedy-set score (must be 1000000):", run_score(greedy_txt))

# 2) an actual edge as the chosen set -> not independent -> 0
a, b = edges[0]
print("planted adjacent pair", a, b)
print("adjacent-pair score (must be 0):", run_score("2\n%d\n%d\n" % (a, b)))

# 3) header mismatch -> 0
print("header-mismatch score (must be 0):", run_score("3\n0\n1\n"))
# 4) out-of-range id -> 0
print("out-of-range score (must be 0):", run_score("1\n999999999\n"))
# 5) duplicate id -> 0
print("duplicate score (must be 0):", run_score("2\n5\n5\n"))
# 6) empty set -> 0 (feasible, weight 0)
print("empty score (must be 0):", run_score("0\n"))
# 7) a single isolated low-id vertex that is independent trivially -> >0 if w>0
print("single-vertex-0 score (should be > 0 if 0 independent alone):",
      run_score("1\n0\n"))
