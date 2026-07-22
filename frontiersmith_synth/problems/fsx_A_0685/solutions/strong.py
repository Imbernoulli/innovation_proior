# TIER: strong
# Insight: Z_n (n=k*k) has a UNIQUE subgroup H of order k, namely the multiples of k.
# H's cosets are exactly the residue classes mod k. Any perfect tiling of Z_n by a
# size-k tile and n/k offsets must (by the classical uniqueness of factorizations of
# a cyclic p-group) take the cross-section form: one representative per residue class
# mod k, translated by H itself. This turns "search over all size-k subsets of A" into
# "independently pick one element per class" -- a tiny algebraic family instead of an
# exponential search -- and, since the paving is then automatically perfect (defect 0)
# for ANY choice of representatives, the leftover freedom can be spent minimizing the
# quarry fee: pick the CHEAPEST approved cell in each class.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    allowed = [int(x) for x in data[idx:idx + n]]; idx += n
    cost = [int(x) for x in data[idx:idx + n]]; idx += n

    M = n // k
    classes = [[] for _ in range(k)]
    for i in range(n):
        if allowed[i]:
            classes[i % k].append(i)

    B = [min(c, key=lambda i: cost[i]) for c in classes]
    T = [t * k for t in range(M)]  # the unique order-k subgroup H

    print(" ".join(map(str, B)))
    print(" ".join(map(str, T)))


if __name__ == "__main__":
    main()
