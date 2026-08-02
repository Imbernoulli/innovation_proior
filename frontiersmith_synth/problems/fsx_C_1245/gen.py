#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the block-codec instance to stdout.

Instance:
    N H C
    A_1 A_2 ... A_N

N integers A_i (0 <= A_i < 2**30), and cost constants H (per-block header bits)
and C (per-transition branch-misprediction tax). testId 1..10 is a difficulty /
regime ladder: 1-2 are gentle multi-level control cases where a fixed-size,
per-block-minimal-width chunker (the obvious "greedy" approach) already does
fine; 3-7, 9-10 are high-variance / bursty streams engineered so a fixed
chunk size straddling runs of very different magnitude, OR long homogeneous
runs separated by widely-spaced regime switches, make a per-block-minimal
width chunker rack up many width transitions (or waste header/bit budget) --
the "trap" the family spec calls for. All randomness is seeded from testId
only, so the same testId always reproduces byte-identical output.
"""
import sys
import random

VMAX = (1 << 30) - 1
WMAX = 30


def clip(v):
    return max(0, min(VMAX, v))


def gen_case(test_id: int):
    rnd = random.Random(1000 + test_id * 7919)

    if test_id == 1:
        # control: 3 well-separated, long magnitude segments -- plain fixed
        # chunking already captures nearly all the available bit savings.
        N, H, C = 64, 8, 40
        levels = [(0, 15), (200, 800), (1 << 14, 1 << 16)]
        seg = N // len(levels)
        A = []
        for (lo, hi) in levels:
            A.extend(rnd.randint(lo, hi) for _ in range(seg))
        while len(A) < N:
            A.append(rnd.randint(*levels[-1]))

    elif test_id == 2:
        # control: 4 broad, gently increasing segments, each much longer
        # than any reasonable chunk size.
        N, H, C = 128, 10, 60
        levels = [(0, 63), (1 << 8, 1 << 10), (1 << 14, 1 << 16), (1 << 20, 1 << 22)]
        seg = N // len(levels)
        A = []
        for (lo, hi) in levels:
            A.extend(rnd.randint(lo, hi) for _ in range(seg))
        while len(A) < N:
            A.append(rnd.randint(*levels[-1]))

    elif test_id == 3:
        N, H, C = 256, 12, 300
        L = round(N ** 0.5)
        period = 6 * L
        A = []
        small = True
        while len(A) < N:
            for _ in range(period):
                if len(A) >= N:
                    break
                A.append(rnd.randint(0, 15) if small else rnd.randint(1 << 26, VMAX))
            small = not small

    elif test_id == 4:
        N, H, C = 300, 14, 350
        L = round(N ** 0.5)
        period = 5 * L
        A = []
        small = True
        while len(A) < N:
            for _ in range(period):
                if len(A) >= N:
                    break
                A.append(rnd.randint(0, 7) if small else rnd.randint(1 << 25, VMAX))
            small = not small

    elif test_id == 5:
        N, H, C = 512, 16, 250
        L = round(N ** 0.5)
        period = 5 * L
        levels = [(0, 15), (1 << 10, 1 << 14), (1 << 26, VMAX)]
        A = []
        li = 0
        while len(A) < N:
            lo, hi = levels[li % len(levels)]
            for _ in range(period):
                if len(A) >= N:
                    break
                A.append(rnd.randint(lo, hi))
            li += 1

    elif test_id == 6:
        # long homogeneous small run, punctuated by rare isolated huge spikes:
        # nearly all bits are cheap, but each spike forces two width flips.
        N, H, C = 500, 10, 500
        L = round(N ** 0.5)
        A = [rnd.randint(0, 20) for _ in range(N)]
        nspikes = max(1, N // (4 * L))
        for _ in range(nspikes):
            A[rnd.randrange(N)] = rnd.randint(1 << 27, VMAX)

    elif test_id == 7:
        N, H, C = 800, 12, 300
        L = round(N ** 0.5)
        A = []
        small = True
        while len(A) < N:
            run = rnd.randint(3 * L, 5 * L) if small else rnd.randint(2 * L, 3 * L)
            for _ in range(run):
                if len(A) >= N:
                    break
                A.append(rnd.randint(0, 31) if small else rnd.randint(1 << 24, VMAX))
            small = not small

    elif test_id == 8:
        # control: smooth, slowly-drifting magnitude -- no adversarial regime
        # changes, exercises that strong isn't worse than greedy for free.
        N, H, C = 1000, 10, 100
        base = 1
        A = []
        for _ in range(N):
            base = clip(int(base * 1.006) + 1)
            A.append(clip(base + rnd.randint(0, 3)))

    elif test_id == 9:
        N, H, C = 1500, 18, 400
        L = round(N ** 0.5)
        classes = [(0, 3), (1 << 6, 1 << 8), (1 << 14, 1 << 16),
                   (1 << 20, 1 << 22), (1 << 26, 1 << 28), (1 << 29, VMAX)]
        A = []
        while len(A) < N:
            run = rnd.randint(2 * L, 4 * L)
            lo, hi = classes[rnd.randrange(len(classes))]
            for _ in range(run):
                if len(A) >= N:
                    break
                A.append(rnd.randint(lo, hi))

    else:  # test_id == 10 -- largest, hardest: long alternating runs + scattered spikes
        N, H, C = 2000, 20, 450
        L = round(N ** 0.5)
        A = []
        while len(A) < N:
            run = rnd.randint(4 * L, 8 * L)
            lo, hi = rnd.choice([(0, 7), (1 << 27, VMAX)])
            for _ in range(run):
                if len(A) >= N:
                    break
                A.append(rnd.randint(lo, hi))
        for _ in range(N // 150):
            A[rnd.randrange(N)] = rnd.randint(1 << 29, VMAX)

    A = A[:N]
    assert len(A) == N
    for v in A:
        assert 0 <= v <= VMAX
    return N, H, C, A


def main():
    test_id = int(sys.argv[1])
    N, H, C, A = gen_case(test_id)
    out = [f"{N} {H} {C}", " ".join(map(str, A))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
