import sys

sys.setrecursionlimit(100000)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    w = [int(data[idx + t]) for t in range(n)]
    idx += n

    if n <= 1:
        print(0)
        return

    # Independent brute force, DIFFERENT METHOD from the DP:
    # We literally simulate the merging process. State = a list of groups,
    # each group is (total_width). Repeatedly pick ANY adjacent pair to merge,
    # paying the combined width; explore ALL orders of merges via recursion,
    # and take the minimum total cost to reduce to a single group.
    #
    # This explores order-of-operations directly rather than tree structure,
    # so it is an independent check of the recurrence.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(groups):
        # groups: tuple of group widths, in left-to-right order
        if len(groups) == 1:
            return 0
        ans = None
        for p in range(len(groups) - 1):
            merged = groups[p] + groups[p + 1]
            new_groups = groups[:p] + (merged,) + groups[p + 2:]
            cur = merged + best(new_groups)
            if ans is None or cur < ans:
                ans = cur
        return ans

    print(best(tuple(w)))

main()
