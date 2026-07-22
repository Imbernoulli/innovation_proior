# TIER: greedy
# The obvious first move: model the slowdown as a function of CROWD SIZE, the
# way a lot of congestion/throughput models do it -- more nearby agents means
# more interference, full stop. This solution completely ignores the actual
# neighbor DISTANCES the log hands it (KERNEL is a flat constant, so every
# sensed neighbor counts the same regardless of how close it is) and fits a
# single saturating scale q against neighbor COUNT alone, via a 1-D grid
# search minimizing training error: predicted = v / (1 + q * n_neighbors).
#
# On the logged 3-8-drone flights this is a perfectly serviceable recipe --
# count and true interference happen to correlate well enough there. But the
# held-out swarms are not just BIGGER, they are DENSER: the same neighbor
# count now sits much closer together, so the true pairwise kernel value per
# neighbor is far larger than what training calibrated. A model that never
# separated "how many" from "how close" has no way to see that coming, and
# systematically UNDER-predicts the slowdown on the dense regime.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("KERNEL 1"); print("OUT v"); return
    n_rows = int(data[0])
    idx = 2
    rows = []
    for _ in range(n_rows):
        k = int(data[idx]); v = float(data[idx + 1]); y = float(data[idx + 2])
        idx += 3 + k
        rows.append((k, v, y))

    best_q, best_e = 0.0, None
    q = 0.0
    while q <= 3.0 + 1e-9:
        e = 0.0
        for k, v, y in rows:
            pred = v / (1.0 + q * k)
            e += (pred - y) ** 2
        if best_e is None or e < best_e:
            best_e, best_q = e, q
        q += 0.01

    print("KERNEL 1")
    print("OUT v / ( 1 + %.6f * S )" % best_q)


if __name__ == "__main__":
    main()
