# TIER: strong
# Multi-start shuffled greedy with local repair: try replacing one or two blockers by a
# rejected candidate, then refill the family greedily from the same pool.
import sys

MASK64 = (1 << 64) - 1


class XorShift64:
    def __init__(self, seed):
        self.s = (seed ^ 0x9E3779B97F4A7C15) & MASK64
        if self.s == 0:
            self.s = 0x123456789ABCDEF

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= (x >> 7)
        x ^= (x << 17) & MASK64
        self.s = x & MASK64
        return self.s

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.next() % (i + 1)
            arr[i], arr[j] = arr[j], arr[i]


def sample_mask(rng, n, w):
    arr = list(range(n))
    for i in range(w):
        j = i + (rng.next() % (n - i))
        arr[i], arr[j] = arr[j], arr[i]
    mask = 0
    for i in range(w):
        mask |= 1 << arr[i]
    return mask


def can_add(code, cand, lam):
    m = len(code)
    for row in code:
        if (cand & row).bit_count() > lam:
            return False
    for i in range(m):
        rem = cand & ~code[i]
        for j in range(i + 1, m):
            if (rem & ~code[j]) == 0:
                return False
    for i in range(m):
        rem = code[i] & ~cand
        for j in range(m):
            if i != j and (rem & ~code[j]) == 0:
                return False
    return True


def conflict_indices(code, cand, lam, limit):
    bad = set()
    m = len(code)

    def add(idx):
        bad.add(idx)
        return len(bad) > limit

    for i, row in enumerate(code):
        if (cand & row).bit_count() > lam and add(i):
            return bad

    for i in range(m):
        rem = cand & ~code[i]
        for j in range(i + 1, m):
            if (rem & ~code[j]) == 0:
                if add(i) or add(j):
                    return bad

    for i in range(m):
        rem = code[i] & ~cand
        for j in range(m):
            if i != j and (rem & ~code[j]) == 0:
                if add(i) or add(j):
                    return bad
    return bad


def build_from_order(order, lam, cap):
    code = []
    rejected = []
    for cand in order:
        if len(code) >= cap:
            break
        if can_add(code, cand, lam):
            code.append(cand)
        else:
            rejected.append(cand)

    for _ in range(2):
        changed = False
        for cand in rejected[:350]:
            bad = conflict_indices(code, cand, lam, 2)
            if not bad or len(bad) > 2:
                continue
            trial = [row for i, row in enumerate(code) if i not in bad]
            if not can_add(trial, cand, lam):
                continue
            trial.append(cand)
            present = set(trial)
            for row in order:
                if len(trial) >= cap:
                    break
                if row in present:
                    continue
                if can_add(trial, row, lam):
                    trial.append(row)
                    present.add(row)
            if len(trial) > len(code):
                code = trial
                changed = True
                break
        if not changed:
            break
    return code


def mask_to_bits(mask, n):
    return "".join("1" if ((mask >> j) & 1) else "0" for j in range(n))


def main():
    toks = sys.stdin.read().split()
    n, w, lam, _d, cap, salt = map(int, toks[:6])
    rng = XorShift64(0xD00 + 149 * salt)
    seen = set()
    pool = []
    for _ in range(1500 + 20 * n):
        cand = sample_mask(rng, n, w)
        if cand not in seen:
            seen.add(cand)
            pool.append(cand)

    best = []
    for _ in range(5):
        order = list(pool)
        rng.shuffle(order)
        code = build_from_order(order, lam, cap)
        if len(code) > len(best):
            best = code

    print(len(best))
    for row in best:
        print(mask_to_bits(row, n))


if __name__ == "__main__":
    main()
