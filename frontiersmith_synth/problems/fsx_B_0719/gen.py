#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of the Spice Rack Relay to stdout.

Instance shape (deterministic in testId only):
  n items ("spice jars"), m customer requests, a paid-swap budget K,
  and P correlated pairs whose accesses are concentrated inside their
  own dedicated time epoch of the request sequence (a "burst train"),
  while filler ("single") items are requested at a roughly steady
  background rate for the whole day.

Output (stdout):
  n m K P
  P lines: "a_i b_i"      (1-indexed jar ids, the i-th correlated pair)
  1 line : m integers     (the request sequence s_1 .. s_m, 1-indexed jar ids)
"""
import sys
import random


def scale_for(t):
    # (n, m, P) ladder: small/sane -> large/adversarial
    table = {
        1: (16, 150, 3),
        2: (24, 260, 4),
        3: (32, 420, 5),
        4: (50, 700, 8),
        5: (70, 1100, 10),
        6: (100, 1700, 14),
        7: (140, 2500, 18),
        8: (180, 3600, 24),
        9: (240, 5200, 30),
        10: (300, 7000, 38),
    }
    return table[t]


def build(t):
    n, m, P = scale_for(t)
    rnd = random.Random(20260719 + 97 * t)

    ids = list(range(1, n + 1))
    rnd.shuffle(ids)
    pair_items = ids[: 2 * P]
    filler_items = ids[2 * P :]
    if not filler_items:
        # extremely small instances: guarantee at least one filler item
        filler_items = [pair_items.pop()]

    pairs = [(pair_items[2 * i], pair_items[2 * i + 1]) for i in range(P)]

    # Budget: enough for several well-timed round-trip pair rescues, but
    # short of covering every pair train in full -- forces a genuine
    # value/cost selection, not "rescue everything".
    K = max(n, round(9.0 * n))

    # Skewed ("Zipf-like") filler popularity: a few singles are genuinely
    # hot, most are cold. Without this skew, frequency-sorting barely
    # beats a random order (nearly-uniform demand makes position
    # irrelevant), so the ladder needs this to be discriminative at all.
    filler_count = len(filler_items)
    filler_rank = filler_items[:]
    rnd.shuffle(filler_rank)
    weights = [1.0 / ((r + 1) ** 0.95) for r in range(filler_count)]

    elen = m // P
    seq = []
    for i in range(P):
        start = i * elen
        end = (i + 1) * elen if i < P - 1 else m
        L = end - start
        a, b = pairs[i]

        # burst slots occupy a MAJORITY of this pair's own epoch: while
        # active, the pair dominates local traffic and earns a top-band
        # static rank overall -- but a static order can only ever crown
        # ONE global top band shared by all P pairs, so during any given
        # epoch the other (P-1) dormant pairs still squat on front slots
        # that rightfully belong to the pair that is actually firing.
        nb = int(max(2, min(L // 2 - 1, round(0.30 * L))))

        local = [None] * L
        anchors = []
        for k in range(nb):
            anc = int(round((k + 0.5) * L / nb))
            anc = min(anc, L - 2)
            anc = max(anc, 0)
            anchors.append(anc)
        used = set()
        for anc in anchors:
            p0, p1 = anc, anc + 1
            if p0 in used or p1 in used or p1 >= L:
                continue
            order_ab = (a, b) if rnd.random() < 0.5 else (b, a)
            local[p0], local[p1] = order_ab
            used.add(p0)
            used.add(p1)
        picks = rnd.choices(filler_rank, weights=weights, k=L - len(used))
        pit = iter(picks)
        for j in range(L):
            if local[j] is None:
                local[j] = next(pit)
        seq.extend(local)

    assert len(seq) == m
    return n, m, K, P, pairs, seq


def main():
    t = int(sys.argv[1])
    n, m, K, P, pairs, seq = build(t)
    out = []
    out.append(f"{n} {m} {K} {P}")
    for a, b in pairs:
        out.append(f"{a} {b}")
    out.append(" ".join(str(x) for x in seq))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
