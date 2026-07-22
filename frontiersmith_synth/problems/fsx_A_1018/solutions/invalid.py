# TIER: invalid
"""Emits a structurally plausible but INFEASIBLE artifact: every feature
station's return/ring doors are set to zero (below their mandatory floors),
with all mass dumped on the cashier door. Must score Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def take(k):
        nonlocal p
        v = data[p:p + k]
        p += k
        return v

    Nc, K = int(take(1)[0]), int(take(1)[0])
    take(4)
    lo = [float(x) for x in take(Nc)]
    take(K)
    take(Nc)
    take(K)

    out = []
    for i in range(Nc):
        out.append("%.9f %.9f %.9f" % (0.9 - lo[i], 0.1, lo[i]))
    for k in range(K):
        # violates BOTH the return-door and ring-door floors
        out.append("0.0 0.0 1.0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
