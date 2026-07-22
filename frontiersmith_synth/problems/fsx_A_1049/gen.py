#!/usr/bin/env python3
"""gen.py <testId> -- ambigram-matrix-cipher instance generator.

Instance = a dictionary of L 7x5 binary glyphs (35 bits each, flattened
row-major so that rot180 == reverse the 35-length string), plus the number
of grid slots k (always odd -> forces a true center slot) and the Hamming
threshold d used for nearest-glyph decoding, and the per-distinct-letter
bonus used in the objective.

The dictionary is engineered in three controlled families so the
rotation-compatibility structure (and the trap) is planted, not accidental:
  - PALINDROME letters: bitmap == reverse(bitmap) exactly (distance 0
    self-rotation) -> usable as the center's near-fixed-point.
  - PAIR letters: 5 pairs (u_i, v_i) built so reverse(u_i) == v_i exactly
    (distance 0 cross-rotation) -> the rotation-compatibility graph edges
    the strong solution walks.
  - EXTRA letter: on "trap" testIds, an adversarial letter T is planted
    with the *largest standalone legibility margin in the whole dictionary*
    but a rotation image that is far (Hamming distance > d) from every
    dictionary glyph, so using it anywhere is globally infeasible. On
    non-trap testIds this slot is filled with one more safe palindrome
    instead, so the ladder stays feasible/sane there.
"""
import sys
import random

BITS = 35  # 7 rows x 5 cols, flattened row-major


def rev(s):
    return s[::-1]


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def rand_bits(rng, n=BITS):
    return "".join(rng.choice("01") for _ in range(n))


def rand_palindrome(rng):
    # 35 = 17 mirrored pairs + 1 free middle bit (index 17)
    half = [rng.choice("01") for _ in range(17)]
    mid = rng.choice("01")
    return "".join(half) + mid + "".join(reversed(half))


def min_dist_to_set(bit, others):
    if not others:
        return BITS + 1
    return min(hamming(bit, o) for o in others)


def gen_pair(rng, existing):
    # v random; u = reverse(v) -> reverse(u) == v exactly (distance 0 both ways).
    for _ in range(200):
        v = rand_bits(rng)
        u = rev(v)
        if u == v:
            continue  # v itself palindromic, degenerate pair; retry
        if u in existing or v in existing:
            continue
        return u, v
    raise RuntimeError("failed to build a fresh pair")


def gen_fresh_palindrome(rng, existing):
    for _ in range(500):
        p = rand_palindrome(rng)
        if p not in existing:
            return p
    raise RuntimeError("failed to build a fresh palindrome")


def gen_trap_letter(rng, existing, d):
    """Letter T: largest min-distance to `existing` (=> largest standalone
    margin), but reverse(T) stays farther than d from every letter in
    `existing` (=> its rotation decode is infeasible everywhere)."""
    best = None
    best_score = -1
    for _ in range(4000):
        cand = rand_bits(rng)
        if cand in existing:
            continue
        rot_min = min_dist_to_set(rev(cand), existing)
        if rot_min <= d + 2:  # keep a healthy safety margin above d
            continue
        own_min = min_dist_to_set(cand, existing)
        if own_min > best_score:
            best_score = own_min
            best = cand
    if best is None:
        raise RuntimeError("failed to build trap letter")
    return best


def gen_weak_anchor(rng, existing, floor=3):
    """A palindrome deliberately placed CLOSE to the rest of the dictionary
    (small standalone margin, but not degenerately tiny) -- this is the
    checker's/trivial's baseline anchor. It stays self-compatible (valid
    at the center) but is a conspicuously weak choice compared to the
    graph's good letters."""
    best = None
    best_score = BITS + 1
    for _ in range(4000):
        cand = rand_palindrome(rng)
        if cand in existing:
            continue
        m = min_dist_to_set(cand, existing)
        if m < floor:
            continue  # keep a floor so the baseline never collapses to ~0
        if m < best_score:
            best_score = m
            best = cand
    if best is None:
        # relax the floor as a fallback (still >= 1, i.e. strictly valid)
        for _ in range(4000):
            cand = rand_palindrome(rng)
            if cand in existing:
                continue
            m = min_dist_to_set(cand, existing)
            if m < 1:
                continue
            if m < best_score:
                best_score = m
                best = cand
    if best is None:
        raise RuntimeError("failed to build weak anchor")
    return best


def build_dictionary(testId, k, d, trap):
    rng = random.Random(1_000_003 * testId + 7)
    rest = []
    seen = set()

    def add(bit):
        rest.append(bit)
        seen.add(bit)

    # 2 "good" palindromes (self-compatible candidates for the center;
    # strong picks among these by margin instead of settling for the
    # weak anchor built below)
    for _ in range(2):
        add(gen_fresh_palindrome(rng, seen))

    # 5 rotation-compatible pairs -> 10 letters (the compatibility graph)
    for _ in range(5):
        u, v = gen_pair(rng, seen)
        add(u)
        add(v)

    # a deliberately WEAK palindrome anchor (small standalone margin,
    # floor-protected) -- this is what the checker's baseline and the
    # trivial solution use everywhere; it is self-compatible (valid) but a
    # conspicuously poor choice next to the graph above. Built into `seen`
    # BEFORE the 14th letter so the trap's rotation-invalidity check below
    # is validated against the anchor too (otherwise the anchor could
    # accidentally land close to reverse(trap) and rescue it).
    anchor = gen_weak_anchor(rng, seen)
    seen.add(anchor)

    # 14th letter: adversarial trap OR one more safe palindrome, checked
    # against the FULL existing dictionary (2 good palindromes + 5 pairs +
    # anchor).
    if trap:
        rest.append(gen_trap_letter(rng, seen, d))
    else:
        rest.append(gen_fresh_palindrome(rng, seen))

    return [anchor] + rest


def schedule(testId):
    # k grows (always odd -> genuine center slot); traps on >=3 of 10 cases
    ks = [3, 3, 5, 5, 7, 7, 9, 9, 11, 11]
    trap_ids = {3, 5, 6, 8, 9, 10}
    k = ks[(testId - 1) % len(ks)]
    trap = testId in trap_ids
    return k, trap


def main():
    testId = int(sys.argv[1])
    k, trap = schedule(testId)
    d = 10
    bonus = 22.0
    letters = build_dictionary(testId, k, d, trap)
    L = len(letters)
    assert len(set(letters)) == L, "dictionary letters must be pairwise distinct"

    out = []
    out.append("%d %d %d %.6f" % (L, k, d, bonus))
    for bit in letters:
        for r in range(7):
            out.append(bit[5 * r:5 * r + 5])
    print("\n".join(out))


if __name__ == "__main__":
    main()
