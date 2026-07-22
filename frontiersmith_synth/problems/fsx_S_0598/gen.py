#!/usr/bin/env python3
"""Generator for fsx_S_0598 -- Most pairwise-inequivalent words under rewriting.

`python3 gen.py <testId>` prints ONE instance to stdout. testId 1..10 is a
difficulty ladder. Everything is seeded from testId only (deterministic).

Instance format (stdin the solver receives):
    k L Nmax
    m
    l_1 r_1
    ...
    l_m r_m
Alphabet = digits '0'..str(k-1). Every rule l->r is length-preserving
(|l| == |r|) and shortlex-decreasing, so single-word normalization terminates.
Nmax = number of congruence classes of words of length in [1,L] (the output cap).
"""
import sys, itertools, random

# ---------- shared rewriting core ----------
def all_words(k, L):
    al = [str(i) for i in range(k)]
    W = []
    for ln in range(1, L + 1):
        for t in itertools.product(al, repeat=ln):
            W.append("".join(t))
    return W

class DSU:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[ra] = rb

def apply_all(w, l, r):
    out = []; i = 0
    while True:
        j = w.find(l, i)
        if j < 0: break
        out.append(w[:j] + r + w[j + len(l):]); i = j + 1
    return out

def num_classes(W, rules, L):
    idx = {w: i for i, w in enumerate(W)}
    dsu = DSU(len(W))
    for w in W:
        for (l, r) in rules:
            for w2 in apply_all(w, l, r):
                if 1 <= len(w2) <= L and w2 in idx:
                    dsu.union(idx[w], idx[w2])
    return len(set(dsu.find(idx[w]) for w in W))

# ---------- rule sampler (identical rng sequence per (k,comm,npres,seed)) ----------
def slt(a, b):
    return (len(a), a) < (len(b), b)

def gen_rules(rng, k, comm_pairs, n_pres):
    al = [str(i) for i in range(k)]
    rules = []; lhs = set()
    def add(l, r):
        if l == r or not slt(r, l) or l in lhs:
            return False
        lhs.add(l); rules.append((l, r)); return True
    pairs = [(a, b) for i, a in enumerate(al) for b in al[i + 1:]]
    rng.shuffle(pairs)
    for (a, b) in pairs[:comm_pairs]:
        add(b + a, a + b)                 # commutation ba -> ab
    tries = added = 0
    while added < n_pres and tries < 3000:
        tries += 1
        ln = rng.choice([2, 3])
        l = "".join(rng.choice(al) for _ in range(ln))
        r = "".join(rng.choice(al) for _ in range(ln))
        if slt(r, l) and l not in lhs:
            if add(l, r):
                added += 1
    return rules

# testId -> (k, L, comm_pairs, npres, seed_offset)  [chosen so every case is in-spec]
LADDER = {
    1:  (3, 5, 1, 3, 3),
    2:  (3, 6, 1, 3, 1),
    3:  (3, 6, 2, 3, 0),
    4:  (3, 6, 1, 4, 4),
    5:  (3, 7, 2, 3, 7),
    6:  (4, 5, 2, 5, 2),
    7:  (4, 5, 1, 5, 0),
    8:  (3, 7, 1, 4, 22),
    9:  (4, 5, 2, 4, 1),
    10: (4, 5, 1, 4, 0),
}

def build(tid):
    k, L, comm, npres, off = LADDER[tid]
    seed = 100000 + tid * 10007 + off * 131
    rng = random.Random(seed)
    comm = min(comm, k * (k - 1) // 2)
    rules = gen_rules(rng, k, comm, npres)
    W = all_words(k, L)
    Nmax = num_classes(W, rules, L)
    return k, L, Nmax, rules

def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    k, L, Nmax, rules = build(tid)
    out = [f"{k} {L} {Nmax}", str(len(rules))]
    for (l, r) in rules:
        out.append(f"{l} {r}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
