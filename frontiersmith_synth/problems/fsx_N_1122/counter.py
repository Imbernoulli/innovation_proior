#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>

Scores a submitted activation schedule for decaying-cascade-timing.

<in>  : the instance (see gen.py / statement.md for the exact format).
<out> : participant artifact -- an activation schedule:
            line 1:      K
            next K lines: node time   (one external activation event each)
<ans> : unused placeholder.

Simulation rule (deterministic, integer-only):
  For step t = 0 .. T-1:
    1) apply every scheduled external event at time t to its node's
       accumulator (only if that node is not yet active);
    2) every node that became active in step t-1 (one-shot emission)
       adds its outgoing edge weight to each not-yet-active neighbour's
       accumulator;
    3) any node whose accumulator now meets its threshold becomes
       permanently active (recorded, so it emits in the NEXT step);
    4) every node still inactive decays: acc = floor(acc * num / den).

Feasibility: after T steps every node must be active, else Ratio: 0.0.
Objective (minimize): K, the number of external activation events used.
Baseline B: the trivial construction that externally activates every
single node once (always feasible) -> B = N.
  score = min(1.0, B / (10 * K))   (fewer events => higher score)
"""
import sys


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt())
    m = int(nxt())
    decay_num = int(nxt())
    decay_den = int(nxt())
    ext_boost = int(nxt())
    horizon = int(nxt())
    theta = [int(nxt()) for _ in range(n)]
    edges = []
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(nxt())
        v = int(nxt())
        w = int(nxt())
        edges.append((u, v, w))
        adj[u].append((v, w))
    return {
        "n": n, "m": m, "decay_num": decay_num, "decay_den": decay_den,
        "ext_boost": ext_boost, "horizon": horizon, "theta": theta, "adj": adj,
    }


def read_schedule(path, n, horizon):
    with open(path, "r") as f:
        text = f.read()
    toks = text.split()
    if len(toks) == 0:
        fail("empty output")

    MAX_K = 50 * max(n, 1) + 1000
    try:
        k = int(toks[0])
    except ValueError:
        fail("first token is not an integer K")
    if k < 0:
        fail("K is negative")
    if k > MAX_K:
        fail(f"K={k} exceeds sane cap {MAX_K}")
    if len(toks) != 1 + 2 * k:
        fail(f"expected {1 + 2 * k} tokens for K={k}, got {len(toks)}")

    events = {}
    p = 1
    for _ in range(k):
        try:
            node = int(toks[p])
            time = int(toks[p + 1])
        except (ValueError, IndexError):
            fail("non-integer or missing node/time token")
        p += 2
        if not (0 <= node < n):
            fail(f"node {node} out of range [0,{n})")
        if not (0 <= time < horizon):
            fail(f"time {time} out of range [0,{horizon})")
        events.setdefault(time, []).append(node)
    return k, events


def simulate(inst, events):
    n = inst["n"]
    theta = inst["theta"]
    adj = inst["adj"]
    decay_num, decay_den = inst["decay_num"], inst["decay_den"]
    ext_boost = inst["ext_boost"]
    horizon = inst["horizon"]

    acc = [0] * n
    active = [False] * n
    just_activated = [False] * n

    for t in range(horizon):
        for node in events.get(t, []):
            if not active[node]:
                acc[node] += ext_boost

        contrib = [0] * n
        for u in range(n):
            if just_activated[u]:
                for (v, w) in adj[u]:
                    if not active[v]:
                        contrib[v] += w
        for v in range(n):
            if contrib[v] and not active[v]:
                acc[v] += contrib[v]

        newly = []
        for v in range(n):
            if not active[v] and acc[v] >= theta[v]:
                newly.append(v)
        just_activated = [False] * n
        for v in newly:
            active[v] = True
            just_activated[v] = True

        for v in range(n):
            if not active[v]:
                acc[v] = (acc[v] * decay_num) // decay_den

    return active


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        inst = read_instance(in_path)
        n, horizon = inst["n"], inst["horizon"]

        k, events = read_schedule(out_path, n, horizon)

        active = simulate(inst, events)
        if not all(active):
            n_inactive = sum(1 for x in active if not x)
            fail(f"{n_inactive} node(s) never activated within horizon {horizon}")

        if k <= 0:
            fail("zero external activations cannot be feasible (no source has in-edges)")

        B = float(n)
        F = float(k)
        sc = min(1000.0, 100.0 * B / max(1e-9, F))
        ratio = sc / 1000.0
        print(f"OK: N={n} K={k} B={n}")
        print("Ratio: %.6f" % ratio)
    except SystemExit:
        raise
    except Exception as e:
        print(f"INFEASIBLE: unexpected error {e}")
        print("Ratio: 0.0")
        sys.exit(0)


if __name__ == "__main__":
    main()
