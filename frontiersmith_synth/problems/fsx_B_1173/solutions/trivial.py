# TIER: trivial
"""Ignore the TDOA data entirely: report the receiver array's centroid X_ref for
every held-out emitter. This is exactly the checker's own internal baseline
predictor, so this solution scores ~0.1 on every case by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); R = int(next(it)); c = float(next(it))
    receivers = []
    for _ in range(R):
        x = float(next(it)); y = float(next(it))
        receivers.append((x, y))
    x_ref = (sum(p[0] for p in receivers) / R, sum(p[1] for p in receivers) / R)

    k_cal = int(next(it))
    for _ in range(k_cal):
        for _ in range(2 + (R - 1)):
            next(it)

    k_test = int(next(it))
    out = []
    for _ in range(k_test):
        for _ in range(R - 1):
            next(it)
        out.append("%.6f %.6f" % (x_ref[0], x_ref[1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
