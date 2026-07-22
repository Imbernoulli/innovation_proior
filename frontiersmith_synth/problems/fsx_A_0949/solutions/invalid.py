# TIER: invalid
import sys


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); K = int(next(it)); S = int(next(it))
    sources = [int(next(it)) for _ in range(S)]

    # Deliberately infeasible: claim to remove K+1 nodes (exceeds the
    # printed budget) and include a source node in the removal list
    # (removing the fire itself is not allowed) -- either violation alone
    # is enough for the checker to reject this with score 0.
    bad_count = K + 1
    bad_ids = [sources[0]] + [((i % N) + 1) for i in range(bad_count - 1)]

    print(bad_count)
    print(" ".join(str(x) for x in bad_ids))


if __name__ == "__main__":
    main()
