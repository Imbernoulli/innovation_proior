# TIER: greedy
# Textbook packed-memory-array reflex: spread the N keys UNIFORMLY across the
# whole capacity so every insert has a gap nearby (classic amortized-worst-case
# hedge). Blind to the fully-visible trace -> it dilutes every scan and wastes
# slack on cold regions where no insert will ever land.
import sys


def main():
    d = sys.stdin.buffer.read().split()
    N = int(d[0]); C = int(d[3])
    pos = [(i * C) // N for i in range(N)]
    # ensure strictly increasing (guaranteed since C >= N, but be safe)
    for i in range(1, N):
        if pos[i] <= pos[i - 1]:
            pos[i] = pos[i - 1] + 1
    out = ["%d 0" % N, " ".join(map(str, pos))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
