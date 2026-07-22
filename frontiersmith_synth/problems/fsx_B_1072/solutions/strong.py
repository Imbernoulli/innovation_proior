# TIER: strong
import sys
import itertools


def is_canonical(t):
    """True iff t is already the lexicographically smallest rotation of itself, i.e. t
    IS the necklace's canonical representative (a Lyndon-style minimal-rotation form)."""
    n = len(t)
    doubled = t + t
    for r in range(1, n):
        if doubled[r:r + n] < t:
            return False
    return True


def main():
    d = sys.stdin.read().split()
    a = int(d[0]); k = int(d[1]); L = int(d[2])

    m = L // k  # non-overlapping budget of whole necklace-representative blocks

    # The insight: a plain substring-cover wastes length re-hitting rotations of the same
    # necklace. Instead, enumerate DISTINCT NECKLACE CANONICAL FORMS directly (the
    # lexicographically-least rotation of each length-k word over the alphabet -- exactly
    # the necklace/Lyndon canonical form) and pack them back-to-back with ZERO overlap.
    # Every block-aligned window then reproduces its representative exactly, so each of
    # the m blocks guarantees one FRESH necklace class -- no waste, no double counting.
    reps = []
    if m > 0:
        for tup in itertools.product(range(a), repeat=k):
            if is_canonical(tup):
                reps.append(tup)
                if len(reps) == m:
                    break

    s = []
    for r in reps:
        s.extend(r)
    # Any leftover characters (only if L is not an exact multiple of k) are filled with a
    # fresh block's prefix so the output still has length exactly L.
    leftover = L - len(s)
    if leftover > 0:
        filler = None
        for tup in itertools.product(range(a), repeat=k):
            if is_canonical(tup) and tup not in reps:
                filler = tup
                break
        if filler is None:
            filler = (0,) * k
        s.extend(filler[:leftover])

    print("".join(str(v) for v in s[:L]))


if __name__ == "__main__":
    main()
