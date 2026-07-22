# TIER: greedy
# The standard textbook liveness heuristic: allocate strictly in the given (topological)
# index order, and free every buffer at the EARLIEST legal instant (the moment its last
# remaining DAG-child has been allocated, or immediately if it has none). This minimizes
# concurrently-live bytes at every step -- but it never reconsiders the ALLOCATION order,
# so independent components that happen to be interleaved in the input stay interleaved,
# forcing their peak memory demands to be paid simultaneously.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    for _ in range(N):
        next(it)
    M = int(next(it))
    children = [[] for _ in range(N + 1)]
    parents_of = [[] for _ in range(N + 1)]
    for _ in range(M):
        p = int(next(it))
        c = int(next(it))
        children[p].append(c)
        parents_of[c].append(p)

    remaining = [len(children[i]) for i in range(N + 1)]
    freed = [False] * (N + 1)
    ops = []
    for i in range(1, N + 1):
        ops.append(i)
        eligible = []
        if remaining[i] == 0:
            eligible.append(i)
        for p in parents_of[i]:
            remaining[p] -= 1
            if remaining[p] == 0:
                eligible.append(p)
        for j in sorted(set(eligible)):
            if not freed[j]:
                ops.append(-j)
                freed[j] = True
    for i in range(1, N + 1):
        if not freed[i]:
            ops.append(-i)

    sys.stdout.write('\n'.join(str(x) for x in ops) + '\n')


if __name__ == '__main__':
    main()
