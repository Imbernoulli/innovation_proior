#!/usr/bin/env python3
# gen.py <testId>  ->  prints ONE instance to stdout.
# testId 1..10 = difficulty ladder (small -> large / adversarial).
#
# Instance: a folding puzzle for an H/P chain on the 2D square lattice.
#   line 1: L
#   line 2: an H/P string of length L  (H = binding residue, P = filler)
#   line 3: L integers, the binding strength w[i] (0 exactly where the string is P,
#           a positive charge where it is H)
#
# The generator plants, on top of a moderate random H background, a "resonant"
# structure: a set of interior lattice columns that become solid H-stacks *only*
# when the serpentine fold width equals a hidden value w* (a prime, chosen far
# from round(sqrt(L)) and far from any power of two).  A coarse width search
# (the obvious approach) skips w*, so it never lights up those stacks.
import sys, random


def column_of(n, w):
    """Column (x-coord) of chain index n in a width-w serpentine fold."""
    r = n // w
    off = n - r * w
    return off if (r % 2 == 0) else (w - 1 - off)


def pick_wstar(L, rng):
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37]
    root = int(round(L ** 0.5))
    pow2 = {2, 4, 8, 16, 32, 64, 128}
    cand = []
    for p in primes:
        if p >= L - 2:
            continue
        if abs(p - root) < 3:
            continue
        if any(abs(p - q) < 3 for q in pow2):
            continue
        cand.append(p)
    if not cand:                      # degenerate small L fallback
        cand = [p for p in primes if 2 <= p < max(3, L - 2)] or [3]
    return rng.choice(cand)


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20250552 + 9973 * tid)

    Ls = [48, 90, 150, 240, 360, 500, 660, 840, 1020, 1200]
    L = Ls[(tid - 1) % len(Ls)]

    wstar = pick_wstar(L, rng)
    # interior columns of the w* fold that we turn into solid H-stacks.
    # avoid column 0 and w*-1 (their vertical bonds are the sequential turns).
    inner = list(range(2, wstar - 2))
    rng.shuffle(inner)
    k = max(2, min(len(inner), wstar // 3))
    hot_cols = set(inner[:k])

    # planted range: a large contiguous middle band of the chain.
    p0 = L // 6
    p1 = L - L // 6

    is_h = [False] * L
    wt = [0] * L

    # 1) random H background (moderate density) so every fold captures contacts
    #    (keeps the checker baseline positive and non-saturating).
    for i in range(L):
        if rng.random() < 0.40:
            is_h[i] = True
            wt[i] = 1

    # 2) resonant plant: inside the band, force the w* hot columns to H with a
    #    higher charge.  These align into tall vertical stacks ONLY at width w*.
    for n in range(p0, p1):
        if column_of(n, wstar) in hot_cols:
            is_h[n] = True
            wt[n] = 3

    # 3) guarantee the hairpin (2-row) baseline has some contacts, so the
    #    normaliser is well-defined.  Wh = ceil(L/2); pair i0 with 2*Wh-1-i0.
    Wh = (L + 1) // 2
    for i0 in (Wh // 4, Wh // 2, (3 * Wh) // 4):
        j0 = 2 * Wh - 1 - i0
        if 0 <= i0 < L and 0 <= j0 < L and abs(i0 - j0) >= 3:
            for idx in (i0, j0):
                if not is_h[idx]:
                    is_h[idx] = True
                    wt[idx] = 1

    s = "".join("H" if is_h[i] else "P" for i in range(L))
    out = []
    out.append(str(L))
    out.append(s)
    out.append(" ".join(str(x) for x in wt))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
