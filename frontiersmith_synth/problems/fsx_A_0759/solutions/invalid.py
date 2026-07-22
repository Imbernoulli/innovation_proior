# TIER: invalid
# Emits a graph whose per-node incident weight sums badly violate the
# feasibility cap (junctions cannot lose more heat per step than they hold) --
# must be rejected with Ratio: 0.0.
import sys


def main():
    sys.stdin.read()
    n = 30
    edges = []
    # pile many oversized edges onto node 0 and a few others: row-sum >> 1
    for j in range(1, 12):
        edges.append((0, j, 0.35))
    for j in range(12, 20):
        edges.append((1, j, 0.3))
    out = [str(len(edges))]
    for i, j, w in edges:
        out.append("%d %d %.8f" % (i, j, w))
    print("\n".join(out))


if __name__ == "__main__":
    main()
