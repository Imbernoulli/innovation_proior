# TIER: trivial
# Spends zero splits: every channel is reconstructed by a single root leaf whose value is
# the channel's own weighted mean. This is exactly the checker's internal baseline.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    C = int(next(it)); S = int(next(it)); D = int(next(it))
    means = []
    for _ in range(C):
        P = int(next(it))
        pts = []
        for _ in range(P):
            v = float(next(it)); w = int(next(it))
            pts.append((v, w))
        sw = sum(w for _, w in pts)
        mean = (sum(v * w for v, w in pts) / sw) if sw > 0 else 0.5
        means.append(mean)

    out = ["0"]           # K = 0 splits
    out.append(str(C))    # L = C leaf declarations (root of every channel)
    for c in range(C):
        out.append("%d 0 0 %.9f" % (c, means[c]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
