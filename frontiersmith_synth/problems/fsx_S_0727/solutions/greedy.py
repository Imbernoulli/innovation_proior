# TIER: greedy
# The "obvious" first attempt: scan targets in the GIVEN input order. As soon
# as two ADJACENT targets share a common factor (a local, no-lookahead check
# via gcd), build that factor as a reusable "unit" register RIGHT AWAY and
# keep reusing it (via a shared power-of-two doubling ladder combined per
# target) until its ratchet budget runs out -- with no replanning and no
# reordering. Still strictly beats doing nothing (every target reused before
# exhaustion is cheaper than an independent chain), but committing to the
# shared unit immediately means it is born with a tiny ratchet budget, so it
# is exhausted after only a handful of targets on the (long) shared block.
import sys
from math import gcd

BIGINF = 10 ** 9
MAXBITS = 7


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


def solve(N, K, B0, targets):
    ops = []
    budget = [BIGINF]
    ladder = None
    unit_val = None
    prev = None
    for v in targets:
        if v == 1:
            prev = v
            continue
        used = False
        if ladder is not None and v % unit_val == 0:
            kk = v // unit_val
            if combine_ladder(kk, ladder, ops, budget, K, B0) is not None:
                used = True
        if not used and ladder is None and prev is not None:
            g = gcd(v, prev)
            if g > 1:
                unit_val = g
                ladder = build_ladder(unit_val, ops, budget, K, B0, MAXBITS)
                kk = v // g
                if combine_ladder(kk, ladder, ops, budget, K, B0) is not None:
                    used = True
        if not used:
            build_plain(v, ops, budget, K, B0)
        prev = v
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
