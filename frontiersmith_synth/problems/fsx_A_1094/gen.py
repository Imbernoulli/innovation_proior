#!/usr/bin/env python3
"""gen.py <testId> -- generator for fsx_A_1094 (Ember Kilns).

Deterministic: seed depends on testId only. Each instance is produced by a
seeded template sampler and then EXACTLY verified (full reachable-class BFS +
simulated myopic greedy) against acceptance predicates, so the shipped case
provably has the trap / ladder properties the statement implies.
"""
import sys, random
from collections import deque, Counter

# ---------------------------------------------------------------- engine ----
def moves_from(state, N):
    """All legal moves from `state` (sorted tuple of token cells, 1-indexed).
    Yields (move, new_state) with move = ('s', i) scatter or ('g', i) gather."""
    cnt = Counter(state)
    occ = set(state)
    res = []
    for i in range(2, N):
        if cnt.get(i, 0) >= 2:                      # scatter at i
            lst = list(state)
            lst.remove(i); lst.remove(i)
            lst.append(i - 1); lst.append(i + 1)
            res.append((('s', i), tuple(sorted(lst))))
        if (i - 1) in occ and (i + 1) in occ:       # gather at i
            lst = list(state)
            lst.remove(i - 1); lst.remove(i + 1)
            lst.append(i); lst.append(i)
            res.append((('g', i), tuple(sorted(lst))))
    return res

def coverage(state, w):
    return sum(w[c - 1] for c in set(state))

def component(init, N, cap=60000):
    """BFS the whole reachable class; returns dict state -> (parent, move)."""
    par = {init: (None, None)}
    dq = deque([init])
    while dq:
        s = dq.popleft()
        for mv, ns in moves_from(s, N):
            if ns not in par:
                par[ns] = (s, mv)
                dq.append(ns)
                if len(par) > cap:
                    return None
    return par

def greedy_final(init, N, w):
    """Myopic best-improvement walk (the canonical first approach)."""
    s = init
    cur = coverage(s, w)
    while True:
        best, bestgain, beststate = None, 0, None
        for mv, ns in moves_from(s, N):
            g = coverage(ns, w) - cur
            if g > bestgain:
                best, bestgain, beststate = mv, g, ns
            elif g == bestgain and g > 0 and best is not None and mv < best:
                best, beststate = mv, ns
        if best is None:
            return cur
        s = beststate
        cur += bestgain

# ------------------------------------------------------------ candidates ----
def gen_candidate(rng, N, M, kind):
    w = [rng.choice([0, 0, 1, 1, 2]) for _ in range(N)]
    c = rng.randint(3, N - 2)                       # 1-indexed centre
    base = rng.randint(2, 5)
    w[c - 1] = base
    if kind == "trap":
        w[c - 2] = rng.randint(2, 5)                # decoys at c-1, c+1
        w[c]     = rng.randint(2, 5)
        w[c - 3] = rng.randint(8, 14)               # treasures at c-2, c+2
        w[c + 1] = rng.randint(8, 14)
        init = tuple([c] * M)                       # full stack on the centre
    else:
        for off, lo, hi in ((-2, 6, 12), (2, 6, 12), (-1, 2, 5), (1, 2, 5)):
            p = c + off
            if 1 <= p <= N:
                w[p - 1] = rng.randint(lo, hi)
        k = rng.randint(2, M - 1)
        init = tuple(sorted([c] * k + [rng.choice([c - 1, c, c + 1])
                                       for _ in range(M - k)]))
    # bait: far-away heavy cells that must end up UNREACHABLE (verified below)
    far = [p for p in range(1, N + 1) if abs(p - c) >= 4]
    rng.shuffle(far)
    baits = []
    for p in far[:rng.randint(1, 2)]:
        w[p - 1] = rng.randint(15, 25)
        baits.append(p)
    return w, init, baits

def evaluate(w, init, baits, N):
    B = coverage(init, w)
    if B < 3:
        return None
    par = component(init, N)
    if par is None or len(par) < 6:
        return None
    best_cov, best_depth = -1, None
    depth = {init: 0}
    # recompute depths from parent chain (component is small)
    for s in par:
        d, x = 0, s
        while par[x][0] is not None:
            x = par[x][0]; d += 1
        depth[s] = d
    covered_union = set()
    for s in par:
        covered_union |= set(s)
        cv = coverage(s, w)
        if cv > best_cov or (cv == best_cov and (best_depth is None or depth[s] < best_depth)):
            best_cov, best_depth = cv, depth[s]
    Fstar = best_cov
    Fg = greedy_final(init, N, w)
    bait_ok = all(b not in covered_union for b in baits) and len(baits) >= 1
    return dict(B=B, Fg=Fg, Fstar=Fstar, comp=len(par),
                depth=best_depth, bait_ok=bait_ok)

def accept(ev, kind):
    if ev is None or not ev["bait_ok"]:
        return False
    B, Fg, Fstar = ev["B"], ev["Fg"], ev["Fstar"]
    if Fstar > 8.0 * B:            # keeps strong ratio <= 0.8 (headroom)
        return False
    if kind == "trap":
        return (Fstar - Fg >= 1.5 * B and Fg <= 4.0 * B
                and ev["comp"] >= 6 and ev["depth"] >= 3)
    else:
        return (Fg >= 1.8 * B and Fstar >= 1.3 * Fg
                and ev["comp"] >= 6 and ev["depth"] >= 2)

# ------------------------------------------------------------------ main ----
CASES = {                           # testId: (N, M, kind)
    1:  (8, 4, "easy"),
    2:  (9, 4, "easy"),
    3:  (9, 4, "trap"),
    4:  (10, 5, "easy"),
    5:  (10, 5, "trap"),
    6:  (11, 5, "trap"),
    7:  (11, 5, "easy"),
    8:  (12, 6, "trap"),
    9:  (13, 6, "trap"),
    10: (13, 6, "easy"),
}

def main():
    test_id = int(sys.argv[1])
    N, M, kind = CASES[test_id]
    rng = random.Random(1000003 * test_id + 77)
    for _ in range(200000):
        w, init, baits = gen_candidate(rng, N, M, kind)
        ev = evaluate(w, init, baits, N)
        if accept(ev, kind):
            print(N, M)
            print(" ".join(map(str, w)))
            print(" ".join(map(str, init)))
            return
    raise SystemExit(f"gen {test_id}: no acceptable instance found")

if __name__ == "__main__":
    main()
