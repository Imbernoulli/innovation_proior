import sys, random

# ---------------------------------------------------------------------------
# single-track-meetpass-rhythm : instance generator
#
# A single line of S stations (0..S-1) joined by S-1 single-track blocks.
# Trains run in two directions (0 = east 0->S-1, 1 = west S-1->0) and have
# HETEROGENEOUS speeds: train j takes h_j ticks per block.  Intermediate
# stations have small passing-loop (siding) capacities; the two terminals are
# yards (huge capacity).
#
# Planted trap (cases >= 4): a few SLOW trains are given early releases and
# TIGHT due dates (so earliest-due-date dispatch sends them first) but LOW
# tardiness weight (so an insightful plan should actually hold them and let the
# fast opposing waves flow).  Greedy ASAP dispatch of a slow blocker cascades
# into long one-sided waits; the strong solution imposes a global meeting
# pattern (fast waves pipelined, slow trains slotted between them).
# ---------------------------------------------------------------------------

def emit(S, cap, trains):
    N = len(trains)
    hmax = max(t[5] for t in trains)
    span = (N + 2) * S * hmax
    maxr = max(t[1] for t in trains)
    maxd = max(t[2] for t in trains)
    TMAX = 4 * span + maxr + maxd + 10 * S * hmax
    out = ["%d %d" % (S, TMAX),
           " ".join(str(c) for c in cap),
           str(N)]
    for (di, r, d, w, v, h) in trains:
        out.append("%d %d %d %d %d %d" % (di, r, d, w, v, h))
    sys.stdout.write("\n".join(out) + "\n")


def planted(i, rng):
    S = rng.choice([7, 8, 9])
    # SCARCE passing loops: sidings only at "every other" station; the stations
    # in between have NO siding (cap 0) so a train cannot stop there and opposing
    # trains must fully clear the multi-block segment before another enters.
    cap = [10 ** 9] * S
    for k in range(1, S - 1):
        cap[k] = 0
    start = rng.choice([1, 2])
    for k in range(start, S - 1, 2):
        cap[k] = 1 if rng.random() < 0.75 else 2
    FAST, SLOW = 2, 4

    bump = (i - 4)                        # 0..6
    nE = rng.randint(3, 4) + bump // 3
    nW = rng.randint(3, 4) + bump // 3
    n_block = 1 + (bump // 3)             # 1..3 slow blockers

    trains = []
    # fast waves, both directions, released as tight bursts on a coarse grid
    g = rng.choice([2, 3])
    ph = rng.choice([0, 1, 2])
    for j in range(nE):
        r = j * g + rng.randint(0, 1)
        h = FAST
        d = r + (S - 1) * h + rng.choice([2, 3, 4])
        v = rng.choice([2, 3, 4, 5])
        trains.append((0, r, d, 1, v, h))
    for j in range(nW):
        r = ph + j * g + rng.randint(0, 1)
        h = FAST
        d = r + (S - 1) * h + rng.choice([2, 3, 4])
        v = rng.choice([2, 3, 4, 5])
        trains.append((1, r, d, 1, v, h))
    # slow blockers: early release + a misleadingly TIGHT (unmeetable) deadline
    # so earliest-due-date dispatch ranks them FIRST and shoves them onto the
    # line; but they are slow AND cheap-to-be-late (v=1), so an insightful plan
    # holds them and lets the fast, high-value waves flow first.
    for b in range(n_block):
        di = rng.randint(0, 1)
        r = rng.randint(0, 2)
        h = SLOW
        d = r + (S - 1) * FAST // 2       # unmeetably tight -> EDD ranks it first
        v = 1                             # but cheap to be late -> should be held
        trains.append((di, r, d, 1, v, h))
    rng.shuffle(trains)
    return S, cap, trains


def easy(i, rng):
    S = rng.choice([5, 6])
    cap = [10 ** 9] * S
    for k in range(1, S - 1):
        cap[k] = rng.choice([1, 2, 2, 3])
    nE = rng.randint(2, 3)
    nW = rng.randint(2, 3)
    trains = []
    for _ in range(nE + nW):
        pass
    idx = 0
    for j in range(nE):
        h = rng.choice([2, 3])
        r = rng.randint(0, 10)
        d = r + (S - 1) * h + rng.randint(3, 8)
        trains.append((0, r, d, 1, rng.choice([1, 2]), h))
    for j in range(nW):
        h = rng.choice([2, 3])
        r = rng.randint(0, 10)
        d = r + (S - 1) * h + rng.randint(3, 8)
        trains.append((1, r, d, 1, rng.choice([1, 2]), h))
    rng.shuffle(trains)
    return S, cap, trains


def main():
    i = int(sys.argv[1])
    rng = random.Random(60800 + 97 * i)
    if i <= 3:
        S, cap, trains = easy(i, rng)
    else:
        S, cap, trains = planted(i, rng)
    emit(S, cap, trains)


if __name__ == "__main__":
    main()
