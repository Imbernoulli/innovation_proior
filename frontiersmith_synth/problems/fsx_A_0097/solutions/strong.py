# TIER: strong
# Randomized Sidon-core + marginal-gain extension.
#   1) build a Sidon set (all pairwise sums distinct) by scanning offsets from
#      several seeded starting positions, capped at k;
#   2) if under budget, fill up to k fittings by repeatedly adding the offset
#      that creates the MOST new distinct reaches (allowing sum collisions);
#   3) keep the construction with the largest |A+A| over all restarts.
import sys


def sidon_core(order, k, V):
    A = []
    sums = set()
    for x in order:
        if len(A) >= k:
            break
        ok = True
        new = []
        for a in A:
            s = x + a
            if s in sums:
                ok = False
                break
            new.append(s)
        if not ok:
            continue
        s2 = 2 * x
        if s2 in sums:
            continue
        A.append(x)
        for s in new:
            sums.add(s)
        sums.add(s2)
    return A, sums


def extend(A, sums, k, V):
    inA = set(A)
    while len(A) < k:
        best_x = -1
        best_gain = -1
        for x in range(0, V + 1):
            if x in inA:
                continue
            gain = 0
            for a in A:
                if (x + a) not in sums:
                    gain += 1
            if (2 * x) not in sums:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x < 0:
            break
        # commit best_x
        for a in A:
            sums.add(x_plus := best_x + a)
        sums.add(2 * best_x)
        A.append(best_x)
        inA.add(best_x)
    return A, sums


def main():
    d = sys.stdin.read().split()
    k = int(d[0]); V = int(d[1])

    # seeded starting offsets for the Sidon core scan
    starts = [0, 1, V // 3, V // 2, (2 * V) // 3, V // 5]
    seen_starts = []
    for o in starts:
        o = max(0, min(V, o))
        if o not in seen_starts:
            seen_starts.append(o)

    best_A = None
    best_F = -1
    for o in seen_starts:
        order = list(range(o, V + 1)) + list(range(0, o))
        A, sums = sidon_core(order, k, V)
        A, sums = extend(A, sums, k, V)
        F = len(sums)
        if F > best_F:
            best_F = F
            best_A = A[:]

    print(len(best_A))
    for x in best_A:
        print(x)


main()
