#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE biogas-digester instance to stdout.

Instance: T days, K feedstock types (4..6). Every day, K "regular"
delivery amounts arrive (a sparse ROTATING-FOCUS pattern: a random subset
of ~K/2 types is well-supplied for a block of 2-4 days, the rest are
thin, then the focus rotates -- so the achievable best mix genuinely
changes over time, not just noise). Separately, 1-2 "spike" events
deliver a large one-off consignment of a designated type `ks`, carrying
its OWN short shelf life (its everyday deliveries keep the normal,
longer shelf life -- only the rare glut is fragile).

All randomness is seeded deterministically from testId only.
"""
import sys
import random

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import simcore as sc

SIZES = [8, 9, 10, 10, 11, 12, 13, 14, 15, 16]
SPIKE_SHELF = 2


def build(test_id):
    rng = random.Random(1000003 * test_id + 7)
    T = SIZES[(test_id - 1) % len(SIZES)]
    K = 4 + (test_id - 1) % 3

    alpha_milli = rng.randint(150, 300)
    switch_cost_i100 = rng.randint(650, 850)          # switch_cost = /100.0
    cap_mult_i100 = rng.randint(200, 260)              # cap_mult = /100.0

    c = [rng.randint(4, 9) for _ in range(K)]
    s = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            v = rng.randint(-3, 4)
            s[i][j] = v
            s[j][i] = v
    thr = [rng.randint(350, 550) for _ in range(K)]
    pen = [rng.randint(3, 6) for _ in range(K)]
    shelf = [rng.randint(5, 7) for _ in range(K)]
    ks = rng.randrange(K)

    cap = max(6, round((cap_mult_i100 / 100.0) * K))

    # sparse rotating-focus regular arrivals
    focus_size = max(2, K // 2)
    arr = [[0] * K for _ in range(T)]
    t = 0
    while t < T:
        blen = rng.randint(2, 4)
        pool = list(range(K))
        rng.shuffle(pool)
        focus = set(pool[:focus_size])
        for tt in range(t, min(T, t + blen)):
            for k in range(K):
                arr[tt][k] = rng.randint(6, 12) if k in focus else rng.randint(0, 2)
        t += blen

    # rare fast-spoiling bulk consignments of the designated type ks
    n_spikes = 1 if T < 12 else 2
    spike_days = sorted(rng.sample(range(1, T - 1), n_spikes)) if T > 3 else [0]
    spikes = []  # list of (day, type, amount, shelf)
    for d in spike_days:
        amt = rng.randint(45, 75)
        spikes.append((d, ks, amt, SPIKE_SHELF))

    rem = 1000 - (1000 // K) * K
    M0 = [1000 // K] * K
    for i in range(rem):
        M0[i] += 1

    return dict(T=T, K=K, alpha_milli=alpha_milli, switch_cost_i100=switch_cost_i100,
                cap=cap, c=c, s=s, thr=thr, pen=pen, shelf=shelf, M0=M0, arr=arr,
                spikes=spikes)


def emit(inst):
    T, K = inst["T"], inst["K"]
    out = [f"{T} {K}"]
    out.append(f"{inst['alpha_milli']} {inst['switch_cost_i100']} {inst['cap']}")
    out.append(" ".join(str(v) for v in inst["c"]))
    s = inst["s"]
    flat = []
    for i in range(K):
        for j in range(i + 1, K):
            flat.append(str(s[i][j]))
    out.append(" ".join(flat) if flat else "")
    out.append(" ".join(str(v) for v in inst["thr"]))
    out.append(" ".join(str(v) for v in inst["pen"]))
    out.append(" ".join(str(v) for v in inst["shelf"]))
    out.append(" ".join(str(v) for v in inst["M0"]))
    for row in inst["arr"]:
        out.append(" ".join(str(v) for v in row))
    out.append(str(len(inst["spikes"])))
    for (d, typ, amt, sh) in inst["spikes"]:
        out.append(f"{d} {typ} {amt} {sh}")
    return "\n".join(out) + "\n"


def main():
    test_id = int(sys.argv[1])
    inst = build(test_id)
    sys.stdout.write(emit(inst))


if __name__ == "__main__":
    main()
