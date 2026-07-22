# TIER: greedy
"""The obvious single-pass policy: watch the box count as it streams in and,
the moment more than THRESH boxes are alive, fully compact everything into
one box. This is exactly the classic size-tiered compaction trigger -- it
reacts only to how many boxes exist, never to what (or when) anyone is about
to look up. It ignores box sizes and key ranges entirely."""
import sys

THRESH = 6


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    for _ in range(N):
        nx(); nx(); nx()  # size lo hi -- unused by this policy
    T = int(nx())
    events = []
    for _ in range(T):
        e = nx()
        if e == "I":
            events.append(("I", None))
        else:
            events.append(("L", int(nx())))

    merges = []
    arrived = 0
    block_count = 0
    for t in range(1, T + 1):
        kind, _ = events[t - 1]
        if kind == "I":
            arrived += 1
            block_count += 1
            if block_count > THRESH:
                merges.append((t, 1, arrived))
                block_count = 1

    out = [str(len(merges))]
    for (g, f, l) in merges:
        out.append(f"{g} {f} {l}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
