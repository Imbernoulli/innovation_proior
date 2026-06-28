import sys, random

# Random small-case generator for "Guess Permutation by <= n log n Comparisons".
# Usage: python3 gen.py <seed>
# Emits: n on the first line, then n DISTINCT integer keys on the second line.
# Keys are distinct (the problem promises distinct keys). We mix two regimes:
#   - a true permutation of 1..n (the canonical "hidden permutation" case), and
#   - n distinct integers drawn from a wider range (to exercise large/negative
#     key values while keeping distinctness),
# and occasionally adversarial orders (already-sorted, reverse-sorted, the
# classic merge-sort worst-case interleave) that stress the comparison count.

def merge_worstcase(arr):
    # Build the permutation of arr that maximizes top-down merge-sort comparisons
    # (the standard "unshuffle" construction): recursively interleave the two
    # halves of the sorted target so every merge runs to the very last element.
    if len(arr) <= 1:
        return arr[:]
    mid = (len(arr) + 1) // 2          # left half gets the ceil, matching split
    left = merge_worstcase(arr[:mid])
    right = merge_worstcase(arr[mid:])
    out = []
    i = j = 0
    take_left = True
    # interleave so the merge alternates and consumes both sides fully
    while i < len(left) or j < len(right):
        if take_left and i < len(left):
            out.append(left[i]); i += 1
        elif j < len(right):
            out.append(right[j]); j += 1
        else:
            out.append(left[i]); i += 1
        take_left = not take_left
    return out

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = rng.randint(0, 5)
    n = rng.randint(0, 12)             # small cases for the brute oracle

    if mode == 0:                      # permutation of 1..n, shuffled
        keys = list(range(1, n + 1)); rng.shuffle(keys)
    elif mode == 1:                    # distinct ints over a wide signed range
        pool = rng.sample(range(-50, 51), n) if n <= 101 else list(range(n))
        keys = pool
    elif mode == 2:                    # already sorted ascending
        keys = list(range(1, n + 1))
    elif mode == 3:                    # reverse sorted
        keys = list(range(n, 0, -1))
    elif mode == 4:                    # merge-sort worst-case interleave
        keys = merge_worstcase(list(range(1, n + 1)))
    else:                              # distinct ints, some with large magnitude
        base = rng.sample(range(-1000000000, 1000000001), n) if n > 0 else []
        keys = base

    print(n)
    print(" ".join(map(str, keys)))

main()
