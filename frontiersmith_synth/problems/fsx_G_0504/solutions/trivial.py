# TIER: trivial
# Reproduces the checker-internal first-fit reference construction.
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


def mask_to_bits(mask, n):
    return "".join("1" if ((mask >> j) & 1) else "0" for j in range(n))


def main():
    toks = sys.stdin.read().split()
    n, w, lam, _d, cap, salt = map(int, toks[:6])
    rng = XorShift64(0xB00 + 131 * salt)
    seen = set()
    candidates = []
    for _ in range(100 + 3 * n):
        cand = sample_mask(rng, n, w)
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)
    code = []
    for cand in candidates:
        if len(code) >= cap:
            break
        if can_add(code, cand, lam):
            code.append(cand)
    print(len(code))
    for row in code:
        print(mask_to_bits(row, n))


if __name__ == "__main__":
    main()
