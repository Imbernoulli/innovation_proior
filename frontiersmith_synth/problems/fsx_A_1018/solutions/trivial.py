# TIER: trivial
"""Reproduces the checker's own internal reference construction: mostly dump
discretionary flow onto the shortcut/return doors toward the (high-cashier)
feature stations, keeping only a small trickle continuing each loop."""
import sys

ALPHA = 0.08  # must match verify.py's BASE_ALPHA


def main():
    data = sys.stdin.read().split()
    p = 0

    def take(k):
        nonlocal p
        v = data[p:p + k]
        p += k
        return v

    Nc, K = int(take(1)[0]), int(take(1)[0])
    f_ring, f_short, f_hret, f_hring = (float(x) for x in take(4))
    lo = [float(x) for x in take(Nc)]
    hi = [float(x) for x in take(K)]
    take(Nc)  # shortcut targets (unused by this strategy)
    take(K)   # return targets (unused by this strategy)

    out = []
    for i in range(Nc):
        remaining = 1.0 - f_ring - f_short - lo[i]
        p_loop = f_ring + ALPHA * remaining
        p_short = f_short + (1.0 - ALPHA) * remaining
        out.append("%.9f %.9f %.9f" % (p_loop, p_short, lo[i]))
    for k in range(K):
        remaining = 1.0 - f_hret - f_hring - hi[k]
        p_ret = f_hret + ALPHA * remaining
        p_ring = f_hring + (1.0 - ALPHA) * remaining
        out.append("%.9f %.9f %.9f" % (p_ret, p_ring, hi[k]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
