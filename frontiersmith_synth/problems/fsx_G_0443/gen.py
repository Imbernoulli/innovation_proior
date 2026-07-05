#!/usr/bin/env python3
"""Generate ONE tensor-network contraction instance (quantum-sim flavored).

python3 gen.py <testId>   -> prints one instance to stdout.
testId 1..10 is a difficulty ladder (more tensors / higher connectivity).
Randomness is seeded ONLY by testId => fully deterministic.

Instance schema (stdin for the solver, <in> for the checker):
  line 1:  m k               # m tensors, k indices (bonds+open legs)
  line 2:  d_0 d_1 ... d_{k-1}   # dimension of each index
  next m lines: deg  i1 i2 ... i_deg   # the index ids carried by tensor u

An index that appears on exactly 2 tensors is an internal bond (summed on
contraction); an index that appears on exactly 1 tensor is an open leg (kept).
"""
import sys
import random


def main():
    tid = int(sys.argv[1])
    rng = random.Random(9000 + tid * 7919)

    sizes = [0, 8, 9, 10, 12, 13, 15, 16, 18, 20, 22]
    m = sizes[tid] if 0 < tid < len(sizes) else 22

    tensors = [[] for _ in range(m)]
    dims = []

    def new_index(d):
        dims.append(d)
        return len(dims) - 1

    # --- spanning tree guarantees a connected network ---
    order = list(range(m))
    rng.shuffle(order)
    for a in range(1, m):
        u = order[a]
        v = order[rng.randrange(a)]
        d = rng.choice([2, 2, 3, 4])
        idx = new_index(d)
        tensors[u].append(idx)
        tensors[v].append(idx)

    # --- extra bonds create cycles (=> contraction is genuinely hard) ---
    extra = int(round(0.9 * m))
    made = 0
    guard = 0
    while made < extra and guard < 50 * m:
        guard += 1
        u = rng.randrange(m)
        v = rng.randrange(m)
        if u == v:
            continue
        d = rng.choice([2, 2, 3, 4])
        idx = new_index(d)
        tensors[u].append(idx)
        tensors[v].append(idx)
        made += 1

    # --- some open legs (external indices, never summed) ---
    for u in range(m):
        if rng.random() < 0.4:
            d = rng.choice([2, 2, 3])
            idx = new_index(d)
            tensors[u].append(idx)

    k = len(dims)
    out = ["%d %d" % (m, k), " ".join(map(str, dims))]
    for u in range(m):
        row = [str(len(tensors[u]))] + [str(i) for i in tensors[u]]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
