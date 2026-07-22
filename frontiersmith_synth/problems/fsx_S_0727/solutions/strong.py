# TIER: strong
# The insight: look at the WHOLE target set before writing a single op.
#   1) Mine the base that is shared by the most targets (via pairwise gcds,
#      tallying divisibility support) -- this discovers the true "Steiner"
#      shared sub-sum instead of a coincidental adjacent pair.
#   2) Partition targets into "shared" (multiples of that base) and the rest.
#   3) Build the non-shared targets FIRST. This is real, necessary work --
#      not padding -- but it also has the side effect of ADVANCING the
#      program position, so that when the shared base's doubling ladder is
#      finally built, it is born at a much later position and therefore
#      inherits a much larger ratchet budget than an eagerly-built ladder
#      would (a "temporal commitment" strategy: don't commit a
#      widely-reused resource until the ratchet has matured).
#   4) Serve the shared targets off that ladder; if a rung's budget ever
#      runs dry, rebuild a fresh ladder right there (now at an even LATER,
#      even higher-budget position) and keep going, instead of giving up on
#      sharing for the remainder like the greedy strategy does.
import sys
from math import gcd

BIGINF = 10 ** 9


def build_plain(v, ops, budget, K, B0):
    if v == 1:
        return 0
    bits = bin(v)[2:]
    cur = 0
    for b in bits[1:]:
        ops.append((cur, cur))
        k = len(ops)
        budget.append(B0 + k // K)
        cur = k
        if b == '1':
            ops.append((cur, 0))
            k = len(ops)
            budget.append(B0 + k // K)
            cur = k
    return cur


def build_ladder(unit_val, ops, budget, K, B0, max_bits):
    uidx = build_plain(unit_val, ops, budget, K, B0)
    ladder = [uidx]
    cur = uidx
    for _ in range(1, max_bits):
        if budget[cur] < 2:
            break
        budget[cur] -= 2
        ops.append((cur, cur))
        k = len(ops)
        budget.append(B0 + k // K)
        cur = k
        ladder.append(cur)
    return ladder


def combine_ladder(kk, ladder, ops, budget, K, B0):
    bits = [b for b in range(kk.bit_length()) if (kk >> b) & 1]
    if not bits or max(bits) >= len(ladder):
        return None
    if len(bits) == 1:
        return ladder[bits[0]]
    for b in bits:
        if budget[ladder[b]] < 1:
            return None
    acc = ladder[bits[0]]
    for b in bits[1:]:
        r = ladder[b]
        budget[acc] -= 1
        budget[r] -= 1
        ops.append((acc, r))
        k = len(ops)
        budget.append(B0 + k // K)
        acc = k
    return acc


def popcount(v):
    return bin(v).count("1")


def solve(N, K, B0, targets):
    n = len(targets)
    cand = {}
    for a in range(n):
        for b in range(a + 1, n):
            g = gcd(targets[a], targets[b])
            if g > 1:
                cand[g] = cand.get(g, 0) + 1

    best_g, best_support = None, 0
    for g in cand:
        support = sum(1 for v in targets if v % g == 0)
        if support > best_support:
            best_support = support
            best_g = g

    ops = []
    budget = [BIGINF]

    if best_g is None or best_support < 3:
        for v in targets:
            build_plain(v, ops, budget, K, B0)
        return ops

    shared = [v for v in targets if v % best_g == 0]
    rest = [v for v in targets if v % best_g != 0]

    # 3) build the free filler (non-shared) targets first
    for v in rest:
        build_plain(v, ops, budget, K, B0)

    max_bits = max((v // best_g).bit_length() for v in shared) + 1
    shared_sorted = sorted(shared, key=lambda v: -popcount(v // best_g))

    # 4) NOW commit to the shared base -- it inherits a much bigger ratchet
    # budget than it would have if built first.
    ladder = build_ladder(best_g, ops, budget, K, B0, max_bits)
    for v in shared_sorted:
        kk = v // best_g
        r = combine_ladder(kk, ladder, ops, budget, K, B0)
        if r is None:
            ladder = build_ladder(best_g, ops, budget, K, B0, max_bits)
            r = combine_ladder(kk, ladder, ops, budget, K, B0)
        if r is None:
            build_plain(v, ops, budget, K, B0)
    return ops


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); B0 = int(next(it))
    targets = [int(next(it)) for _ in range(N)]

    ops = solve(N, K, B0, targets)
    out = [str(len(ops))]
    out.extend("%d %d" % (i, j) for (i, j) in ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
