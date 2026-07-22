# TIER: invalid
# Plausible-looking but infeasible: allocates and IMMEDIATELY frees each buffer in index
# order (a naive "LIFO reuse" scheme). For any real precedence edge p->c with c allocated
# well after p's alloc, this frees p (right after its own alloc) long before c is even
# allocated, violating alloc(c) < free(p) -- must score 0.
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

    ops = []
    for i in range(1, N + 1):
        ops.append(i)
        ops.append(-i)
    sys.stdout.write('\n'.join(str(x) for x in ops) + '\n')


if __name__ == '__main__':
    main()
