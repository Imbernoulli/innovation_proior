#!/usr/bin/env python3
"""Max-scale stress: n=2e5, many queries with sum|S| ~ 2e5."""
import sys, random
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
n = 200000
out = [str(n)]
# mix: deep-ish random tree
for v in range(2, n + 1):
    u = rng.randint(max(1, v - rng.randint(1, 50)), v - 1)
    out.append(f"{u} {v}")
# queries: total sum|S| ~ 2e5
total = 200000
q = 100000
out2 = []
remaining = total
qs = []
for i in range(q):
    if remaining <= 0:
        break
    k = rng.randint(1, min(10, remaining))
    remaining -= k
    verts = rng.sample(range(1, n + 1), k)
    qs.append((k, verts))
out.append(str(len(qs)))
for k, verts in qs:
    out.append(str(k) + " " + " ".join(map(str, verts)))
sys.stdout.write("\n".join(out) + "\n")
