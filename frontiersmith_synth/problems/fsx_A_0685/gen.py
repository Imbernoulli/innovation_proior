#!/usr/bin/env python3
"""
gen.py <testId> -- deterministic instance generator for fsx_A_0685
(coset-constrained-tiling: paving a circular plaza with quarry-approved stones).

Instance:
  n = p*p  (p a prime chosen from a fixed ladder by testId)
  k = p    (tile size; also n/k = p = number of translation offsets)
  A = allowed mask over Z_n (0/1, n entries)   -- "quarry-approved" positions
  c = cost array over Z_n (n nonneg ints)       -- quarry fee, meaningful only where A=1

Planted structure: partition Z_n into k residue classes mod k (the cosets of the
unique order-k subgroup H = {0,k,2k,...,(k-1)k} of Z_n). Every class is guaranteed
>=1 allowed position (so a defect-0 tiling always exists via "one representative per
class, translate by H"). Allowed positions in each class are split into a LOW index
zone (a single "primary") and a HIGH index zone ("secondaries", extra candidates with
independent random cost, some cheaper than the primary) so cost-optimization has
genuine headroom. No length-k window of consecutive residues is ever fully allowed
(forbids the naive contiguous-arc tile).

Trap cases (testId in TRAP_SET): two "victim" classes get NO low-zone representative
at all (only high-zone secondaries), while two "donor" classes get an EXTRA low-zone
representative. A naive "grab the first k approved stones in index order" selection
then draws entirely from k-2 classes (missing both victims, double-covering both
donors) -- a large, structural defect that only a class-aware (coset-cross-section)
construction avoids.
"""
import sys

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
TRAP_SET = {3, 6, 9}
CMAX = 97


class RNG:
    """Tiny deterministic LCG so the instance never depends on Python's hash/random
    implementation details across machines."""
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFF

    def nxt(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFF
        return self.s

    def randint(self, lo, hi):  # inclusive
        if lo >= hi:
            return lo
        span = hi - lo + 1
        return lo + (self.nxt() >> 16) % span

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]


def build(test_id):
    p = PRIMES[(test_id - 1) % len(PRIMES)]
    n = p * p
    k = p
    rng = RNG(1000003 * test_id + 7919)

    lo_zone = list(range(0, k // 2))          # t-values reserved for primaries
    hi_zone = list(range(k // 2, k))          # t-values reserved for secondaries
    is_trap = test_id in TRAP_SET

    victims, donors = set(), set()
    if is_trap and k >= 6:
        order = list(range(k))
        rng.shuffle(order)
        victims = {order[0], order[1]}
        donors = {order[2], order[3]}

    allowed = [0] * n
    cost = [0] * n

    for j in range(k):
        # ---- low zone (primary / extra-primary) allowed slots ----
        lo_t = list(lo_zone)
        rng.shuffle(lo_t)
        if j in victims:
            n_lo = 0
        elif j in donors:
            n_lo = min(2, len(lo_t))
        else:
            n_lo = 1 if lo_t else 0
        for t in lo_t[:n_lo]:
            pos = j + t * k
            allowed[pos] = 1
            cost[pos] = rng.randint(1, CMAX)

        # ---- high zone (secondary) allowed slots ----
        hi_t = list(hi_zone)
        rng.shuffle(hi_t)
        max_extra = max(1, len(hi_t))
        n_hi = rng.randint(1, min(4, max_extra))
        if j in victims:
            n_hi = max(2, n_hi)  # victims rely solely on the high zone
        for t in hi_t[:n_hi]:
            pos = j + t * k
            allowed[pos] = 1
            cost[pos] = rng.randint(1, CMAX)

    # safety: guarantee every class has >=1 allowed slot no matter what
    for j in range(k):
        if not any(allowed[j + t * k] for t in range(k)):
            t = rng.randint(0, k - 1)
            pos = j + t * k
            allowed[pos] = 1
            cost[pos] = rng.randint(1, CMAX)

    # forbid any fully-allowed length-k window of *consecutive* residues (no naive
    # contiguous-arc tile). Break any such window by clearing one non-primary cell.
    changed = True
    guard = 0
    while changed and guard < 4 * n:
        changed = False
        guard += 1
        for start in range(n):
            idxs = [(start + d) % n for d in range(k)]
            if all(allowed[i] for i in idxs):
                # prefer to clear a high-zone (t >= k//2) cell, keep >=1 allowed/class
                victim_pos = None
                for i in idxs:
                    j, t = i % k, i // k
                    if t >= k // 2:
                        cls_count = sum(allowed[j + tt * k] for tt in range(k))
                        if cls_count > 1:
                            victim_pos = i
                            break
                if victim_pos is None:
                    for i in idxs:
                        j = i % k
                        cls_count = sum(allowed[j + tt * k] for tt in range(k))
                        if cls_count > 1:
                            victim_pos = i
                            break
                if victim_pos is not None:
                    allowed[victim_pos] = 0
                    changed = True

    return n, k, allowed, cost


def main():
    test_id = int(sys.argv[1])
    n, k, allowed, cost = build(test_id)
    out = [f"{n} {k}"]
    out.append(" ".join(str(x) for x in allowed))
    out.append(" ".join(str(x) for x in cost))
    print("\n".join(out))


if __name__ == "__main__":
    main()
