#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1023 (cursor-biased-rope) to stdout.

Format:
  line 1: N M F
  next M lines: p_t   (1 <= p_t <= N), the scroll position touched at step t

The trace is built from a sequence of "epochs": a burst of touches clustered
tightly around a moving center, separated by a JUMP (a center change that is
always much larger than any within-epoch jitter, so real epoch boundaries
are a genuine signal in the data) to a new -- sometimes REVISITED -- center
elsewhere on the scroll. Everything is derived from a fixed per-testId RNG
seed -- fully deterministic, no wall clock.

testId 1-2   : BENIGN   -- epoch count equals F and epoch lengths are close
                           to equal, so a chronological equal-F-window split
                           roughly lines up with the real epoch boundaries.
testId 3-10  : TRAP     -- epoch lengths are heavily skewed (one or two huge
                           epochs dominate the trace while dozens of tiny
                           epochs of 2-6 touches are scattered between big
                           jumps) and/or the cursor repeatedly REVISITS a
                           handful of earlier hubs. A strategy that
                           partitions the trace into F equal-sized
                           chronological windows (ignoring where the real
                           epoch boundaries and revisits are) misaligns
                           badly here: a single window straddles several
                           unrelated tiny epochs (its "center" lands nowhere
                           real) while the one huge epoch gets needlessly
                           sliced across several windows.
"""
import sys, random

# (N, M, F, mode, extra) -- extra tunes epoch-count / skew / revisit strength
TABLE = {
    1:  (2000,   400,  5, "benign", {}),
    2:  (4000,   700,  6, "benign", {}),
    3:  (8000,   1200, 6, "skewed", {"n_tiny": 20, "big_frac": 0.42}),
    4:  (10000,  1500, 8, "skewed", {"n_tiny": 8, "big_frac": 0.26}),
    5:  (16000,  2200, 8, "skewed", {"n_tiny": 12, "big_frac": 0.32}),
    6:  (22000,  3000, 8, "skewed", {"n_tiny": 30, "big_frac": 0.50}),
    7:  (30000,  4200, 10, "revisit", {"n_epochs": 34, "n_hubs": 4, "revisit_p": 0.65}),
    8:  (50000,  6000, 10, "skewed", {"n_tiny": 38, "big_frac": 0.50}),
    9:  (80000,  9000, 10, "skewed", {"n_tiny": 44, "big_frac": 0.50}),
    10: (150000, 13000, 12, "skewed", {"n_tiny": 22, "big_frac": 0.42}),
}


def jump_delta(rng, N):
    """A center-to-center jump, always clearly larger than any within-epoch
    jitter (spreads are capped at 25 everywhere below)."""
    mag = rng.randint(N // 12 + 60, N // 4 + 60)
    return mag if rng.random() < 0.5 else -mag


def build_benign(rng, N, M, F):
    n_epochs = F
    base = M // n_epochs
    lens = [max(10, base + rng.randint(-base // 5, base // 5)) for _ in range(n_epochs - 1)]
    lens.append(max(10, M - sum(lens)))
    spreads = [rng.randint(4, 10) for _ in lens]
    centers = []
    cur = rng.randint(N // 4, 3 * N // 4)
    for _ in lens:
        cur = max(1, min(N, cur + jump_delta(rng, N)))
        centers.append(cur)
    return lens, spreads, centers


def build_skewed(rng, N, M, n_tiny, big_frac):
    n_big = 1 if rng.random() < 0.6 else 2
    big_total = int(M * big_frac)
    big_lens = []
    rem = big_total
    for j in range(n_big):
        L = rem if j == n_big - 1 else rem // 2
        big_lens.append(max(25, L))
    tiny_total = max(2 * n_tiny, M - sum(big_lens))
    tiny_lens = [2 for _ in range(n_tiny)]
    diff = tiny_total - sum(tiny_lens)
    for k in range(diff):
        tiny_lens[k % n_tiny] += 1

    lens = []
    slots = ["big"] * len(big_lens) + ["tiny"] * n_tiny
    rng.shuffle(slots)
    bi = ti = 0
    for s in slots:
        if s == "big":
            lens.append(big_lens[bi]); bi += 1
        else:
            lens.append(tiny_lens[ti]); ti += 1

    spreads = []
    centers = []
    cur = rng.randint(N // 4, 3 * N // 4)
    for L in lens:
        cur = max(1, min(N, cur + jump_delta(rng, N)))
        centers.append(cur)
        # big epochs are tight relative to their length (real locality); tiny
        # epochs are also tight -- both are genuine local clusters, just very
        # different DURATIONS, which is exactly what breaks uniform windowing.
        spreads.append(rng.randint(10, 25) if L >= 25 else rng.randint(3, 8))
    return lens, spreads, centers


def build_revisit(rng, N, M, n_epochs, n_hubs, revisit_p):
    base = M // n_epochs
    lens = []
    for _ in range(n_epochs - 1):
        L = max(4, int(base * rng.uniform(0.15, 2.2)))
        lens.append(L)
    lens.append(max(4, M - sum(lens)))
    hubs = [rng.randint(1, N) for _ in range(n_hubs)]
    centers = []
    spreads = []
    cur = rng.randint(N // 4, 3 * N // 4)
    for L in lens:
        if centers and rng.random() < revisit_p:
            cur = max(1, min(N, hubs[rng.randrange(n_hubs)] + rng.randint(-5, 5)))
        else:
            cur = max(1, min(N, cur + jump_delta(rng, N)))
        centers.append(cur)
        spreads.append(rng.randint(6, 15) if L < base else rng.randint(12, 25))
    return lens, spreads, centers


def emit(N, M, F, lens, spreads, centers, rng):
    out = [f"{N} {M} {F}"]
    total = 0
    for L, sp, c in zip(lens, spreads, centers):
        for _ in range(L):
            if total >= M:
                break
            off = rng.randint(-sp, sp) if sp > 0 else 0
            p = max(1, min(N, c + off))
            out.append(str(p))
            total += 1
        if total >= M:
            break
    # pad/truncate to exactly M lines (rounding of epoch lengths can drift by a few)
    while len(out) - 1 < M:
        out.append(str(centers[-1] if centers else 1))
    out = out[:M + 1]
    sys.stdout.write("\n".join(out) + "\n")


def main():
    t = int(sys.argv[1])
    N, M, F, mode, extra = TABLE[t]
    rng = random.Random(900000 + 37 * t)
    if mode == "benign":
        lens, spreads, centers = build_benign(rng, N, M, F)
    elif mode == "skewed":
        lens, spreads, centers = build_skewed(rng, N, M, extra["n_tiny"], extra["big_frac"])
    else:
        lens, spreads, centers = build_revisit(rng, N, M, extra["n_epochs"], extra["n_hubs"], extra["revisit_p"])
    emit(N, M, F, lens, spreads, centers, rng)


if __name__ == "__main__":
    main()
