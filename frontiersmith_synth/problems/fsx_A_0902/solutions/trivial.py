# TIER: trivial
# Reproduces the checker's own baseline: allocate every buffer (in the given topological
# index order), then free every buffer only at the very end. No hole is ever reused, so the
# high-water mark equals the sum of all sizes -- exactly the checker's baseline B.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    for _ in range(N):
        next(it)
    M = int(next(it))
    for _ in range(M):
        next(it)
        next(it)

    ops = [i for i in range(1, N + 1)] + [-i for i in range(N, 0, -1)]
    sys.stdout.write('\n'.join(str(x) for x in ops) + '\n')


if __name__ == '__main__':
    main()
