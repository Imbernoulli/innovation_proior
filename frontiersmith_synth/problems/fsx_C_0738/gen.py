import sys, random

# ---- lock-chamber parity-batching instance generator ----------------------
# Boats have a direction (0=up, 1=down), an arrival time, a deadline, and a
# lateness weight. A lockage batches up to K same-direction boats; its water
# cost is W_same if it repeats the previous lockage's direction, W_diff if it
# alternates (W_diff << W_same). Deadlines add a lateness term on top.
#
# TRAP testIds get a heavily skewed direction mix (~85/15). With capacity K
# forcing several lockages per direction, a "batch by direction to minimize
# lockage count" solver (the obvious recipe) runs one long block per
# direction and pays W_same on almost every lockage inside each block. The
# true minimum number of forced same-direction repeats, given m0 and m1
# lockages worth of boats per direction, is max(0, |m0-m1|-1) -- achievable
# only by interleaving directions at the rate set by the arrival imbalance,
# never by grouping. That gap is what the strong solution exploits.


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260713 + 131 * t)

    N = 10 + t                     # 11 .. 20
    K = [3, 4, 5, 6][t % 4]
    L = 5
    W_same = 100
    W_diff = 10
    s0 = rng.randint(0, 1)

    TRAP_IDS = {3, 6, 9}
    skewed = t in TRAP_IDS

    dirs = []
    if skewed:
        majority = rng.randint(0, 1)
        for _ in range(N):
            dirs.append(majority if rng.random() < 0.85 else 1 - majority)
    else:
        for _ in range(N):
            dirs.append(rng.randint(0, 1))

    horizon = 6 + t
    # deadline slack scaled to the instance's own minimal achievable
    # schedule span S = ceil(N/K)*L, so that even a well-batched,
    # deadline-aware plan keeps real, unavoidable lateness pressure on a
    # sizeable fraction of boats regardless of N -- this keeps the gap
    # between a naive reference and any competent plan from blowing up
    # purely because larger instances happen to carry proportionally more
    # slack.
    S = max(L, -(-N // K) * L)
    lines = []
    for i in range(N):
        dirn = dirs[i]
        a = rng.randint(0, horizon)
        slack_kind = rng.random()
        if slack_kind < 0.45:
            slack = rng.randint(L, max(L + 1, S))                 # tight
        elif slack_kind < 0.8:
            slack = rng.randint(max(L + 1, S), max(L + 2, 2 * S))  # moderate
        else:
            slack = rng.randint(max(L + 2, 2 * S), max(L + 3, 3 * S))  # loose
        d = a + slack
        w = rng.randint(1, 10)
        lines.append(f"{dirn} {a} {d} {w}")

    out = [f"{N} {K} {L} {W_same} {W_diff} {s0}"]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
