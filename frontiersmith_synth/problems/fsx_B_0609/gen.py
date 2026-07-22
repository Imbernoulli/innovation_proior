#!/usr/bin/env python3
# gen.py <testId> : print ONE instance of the gap-seeded packed-array problem.
# Deterministic: everything seeded from testId only.
#
# Instance = an initial sorted key set + a fully-visible future op trace
# (inserts clustered around drifting hotspots, interleaved with range scans).
import sys, random, bisect


def build(t):
    rng = random.Random(9173 + t * 7919)
    N = 200 + 90 * t                 # initial keys
    M = int(1.0 * N)                 # inserts
    S = int(1.2 * N)                 # scans
    Q = M + S
    C = 2 * (N + M)                  # physical capacity
    VMAX = 10 ** 7

    # --- initial keys: distinct, sorted ---
    ks = set()
    while len(ks) < N:
        ks.add(rng.randrange(0, VMAX))
    keys = sorted(ks)
    spacing = VMAX / N

    # --- hotspot centers in the left-middle of the value order (ranks ~0.12..0.62) ---
    H = 4
    centers = []
    for h in range(H):
        frac = 0.05 + 0.35 * (h + 0.5) / H
        centers.append(float(keys[int(frac * N)]))
    bandw = spacing * 3.0

    # --- inserts: clustered near a drifting hotspot ---
    existing = set(keys)
    inserts = []
    drift = [0.0] * H
    for _ in range(M):
        h = rng.randrange(H)
        drift[h] += rng.uniform(-0.20, 0.35) * spacing
        c = centers[h] + drift[h]
        for _try in range(64):
            v = int(c + rng.gauss(0.0, bandw))
            if 0 <= v < VMAX and v not in existing:
                existing.add(v)
                inserts.append(v)
                break
        else:
            # fallback: any free value
            v = rng.randrange(0, VMAX)
            while v in existing:
                v = rng.randrange(0, VMAX)
            existing.add(v)
            inserts.append(v)

    # --- scans: 70% WIDE cold scans (many keys, want packed), 30% narrow hotspot scans ---
    scans = []
    for _ in range(S):
        if rng.random() < 0.70:
            w = rng.randint(int(0.12 * N), max(int(0.12 * N) + 1, int(0.28 * N)))
            a = rng.randint(0, max(0, N - 1 - w))
            b = min(N - 1, a + w)
            scans.append((keys[a], keys[b]))
        else:
            h = rng.randrange(H)
            c = centers[h]
            lo = int(c - 2.0 * bandw)
            hi = int(c + 2.0 * bandw)
            if lo > hi:
                lo, hi = hi, lo
            scans.append((max(0, lo), min(VMAX - 1, hi)))

    # --- interleave inserts and scans into a fixed schedule ---
    schedule = ['I'] * M + ['S'] * S
    rng.shuffle(schedule)
    ii = 0
    si = 0
    ops = []
    for typ in schedule:
        if typ == 'I':
            ops.append(('I', inserts[ii]))
            ii += 1
        else:
            l, r = scans[si]
            si += 1
            ops.append(('S', l, r))

    return N, M, Q, C, VMAX, keys, ops


def main():
    t = int(sys.argv[1])
    N, M, Q, C, VMAX, keys, ops = build(t)
    out = []
    out.append("%d %d %d %d %d" % (N, M, Q, C, VMAX))
    out.append(" ".join(map(str, keys)))
    for op in ops:
        if op[0] == 'I':
            out.append("I %d" % op[1])
        else:
            out.append("S %d %d" % (op[1], op[2]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
