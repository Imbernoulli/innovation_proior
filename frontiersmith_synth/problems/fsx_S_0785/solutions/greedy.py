# TIER: greedy
# The obvious "textbook stride prefetcher" move: scan the whole training log,
# average every INTERIOR per-tick delta (skip the epoch-boundary tick, since
# a real stride detector would just call that an outlier / re-sync event) to
# get one stationary stride estimate S_hat, then predict next = h1 + S_hat.
# This denoises the local jitter far better than a raw 2-point delta, so it
# comfortably beats the trivial baseline on most ticks -- but it silently
# assumes the epoch anchor never moves.  It has no idea the anchor drifts
# with the epoch counter, so it is blind exactly at epoch-boundary queries,
# which is where every held-out query re-enters after an unseen epoch.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("SLOT1 h1")
        print("SLOT2 NONE")
        return
    N, P, tid = int(data[0]), int(data[1]), int(data[2])
    addrs = [int(x) for x in data[3:3 + N]]

    deltas = []
    for t in range(N - 1):
        if (t + 1) % P == 0:
            continue
        deltas.append(addrs[t + 1] - addrs[t])
    S_hat = round(sum(deltas) / len(deltas)) if deltas else 0

    print("SLOT1 h1 + %d" % S_hat)
    print("SLOT2 NONE")


if __name__ == "__main__":
    main()
