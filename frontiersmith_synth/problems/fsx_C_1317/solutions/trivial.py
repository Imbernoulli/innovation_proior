# TIER: trivial
"""Never look at the interaction matrix and never balance composition
against the target: fill monomer types, in blocks, in order of how POORLY
their pure Tg matches the target (worst first), until the chain has length
N. This reproduces the checker's own internal baseline construction
exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    tg = [int(next(it)) for _ in range(K)]
    for _ in range(K * K):
        next(it)  # M -- unused
    caps = [int(next(it)) for _ in range(K)]
    target = int(next(it))

    order = sorted(range(K), key=lambda i: -abs(tg[i] - target))
    seq = []
    for t in order:
        if len(seq) >= N:
            break
        take = min(caps[t], N - len(seq))
        seq.extend([t + 1] * take)
    print(" ".join(map(str, seq)))


if __name__ == "__main__":
    main()
