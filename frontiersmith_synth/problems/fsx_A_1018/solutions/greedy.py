# TIER: greedy
"""The obvious first-instinct fix: don't waste any probability above the
mandatory cashier floor (that's clearly bad), but with no view of the whole
cycle structure, split the remaining discretionary flow EVENLY between a
pit/station's two non-cashier doors. Locally sensible, globally naive."""
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
    f_ring, f_short, f_hret, f_hring = (float(x) for x in take(4))
    lo = [float(x) for x in take(Nc)]
    hi = [float(x) for x in take(K)]
    take(Nc)
    take(K)

    out = []
    for i in range(Nc):
        remaining = 1.0 - f_ring - f_short - lo[i]
        p_loop = f_ring + remaining / 2.0
        p_short = f_short + remaining / 2.0
        out.append("%.9f %.9f %.9f" % (p_loop, p_short, lo[i]))
    for k in range(K):
        remaining = 1.0 - f_hret - f_hring - hi[k]
        p_ret = f_hret + remaining / 2.0
        p_ring = f_hring + remaining / 2.0
        out.append("%.9f %.9f %.9f" % (p_ret, p_ring, hi[k]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
