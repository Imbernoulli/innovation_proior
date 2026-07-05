# TIER: trivial
# Reproduces the checker's internal baseline: the no-carry base-3 set on a
# short register (< (p/2)/9). Valid (progression-free mod p) but small -> ~0.1.
import sys


def base3_nocarry(bound):
    if bound <= 0:
        return []
    pows = []
    v = 1
    while v < bound:
        pows.append(v)
        v *= 3
    out = []
    for mask in range(1 << len(pows)):
        s = 0
        m, i = mask, 0
        while m:
            if m & 1:
                s += pows[i]
            m >>= 1
            i += 1
        if s < bound:
            out.append(s)
    return sorted(set(out))


def main():
    p = int(sys.stdin.read().split()[0])
    half = (p + 1) // 2
    S = base3_nocarry(max(2, half // 9))
    print(" ".join(map(str, S)))


if __name__ == "__main__":
    main()
