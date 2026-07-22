# TIER: strong
"""Exploits the innovation hook: the Perron (quasi-stationary survival) value
is a CYCLE geometric mean, not a per-pit sum. A shortcut door into a feature
station only pays off if the whole corridor->feature->corridor round trip has
a high geometric mean -- and every feature station is forced to leak a LARGE
mandatory cashier fraction, which tanks any cycle passing through it. So:
commit ALL discretionary flow to the low-cashier corridor loop (never touch
the shortcut beyond its bare floor); the dominant interior cycle then never
has to pass through a high-cashier feature station at all. Feature-station
rows don't matter for the resulting spectral radius (they're off the winning
cycle), so they're filled with any feasible split."""
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
        p_loop = f_ring + remaining          # all discretionary flow stays on the loop
        p_short = f_short                    # shortcut held at its bare floor
        out.append("%.9f %.9f %.9f" % (p_loop, p_short, lo[i]))
    for k in range(K):
        remaining = 1.0 - f_hret - f_hring - hi[k]
        p_ret = f_hret + remaining / 2.0     # off the winning cycle -- any feasible split
        p_ring = f_hring + remaining / 2.0
        out.append("%.9f %.9f %.9f" % (p_ret, p_ring, hi[k]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
