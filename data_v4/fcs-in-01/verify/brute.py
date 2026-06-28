import sys
sys.setrecursionlimit(1 << 20)

# Independent oracle for "Guess Permutation by <= n log n Comparisons".
#
# Output line 1: the element indices 1..n in ascending order of their key. We
#   obtain this independently of any merge sort by Python's built-in sorted()
#   on (key, index) pairs (stable, ties broken by index -- but keys are
#   distinct here, so order is unique anyway).
# Output line 2: the number of pairwise sign comparisons spent by the CANONICAL
#   top-down merge-sort schedule -- split at mid = (lo+hi)//2, recurse left then
#   right, and in the merge perform exactly one comparison whenever both sides
#   still have an unconsumed head. We recompute this count with a wholly separate
#   recursive routine that returns its sorted list and accumulates the count in a
#   one-element list, deliberately NOT reusing the C++ structure.

def merge_count(a, counter):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_count(a[:mid], counter)
    right = merge_count(a[mid:], counter)
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        counter[0] += 1                     # one sign comparison
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    keys = list(map(int, data[1:1 + n]))

    # line 1: order independently via sorted()
    order = sorted(range(n), key=lambda i: keys[i])
    line1 = " ".join(str(i + 1) for i in order)

    # line 2: canonical merge-sort comparison count, recomputed independently
    counter = [0]
    merge_count(keys[:], counter)

    out = []
    out.append(line1)               # for n == 0 this is "" -> an empty line, matching sol
    out.append(str(counter[0]))
    sys.stdout.write("\n".join(out) + "\n")

main()
