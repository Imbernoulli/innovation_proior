# TIER: trivial
# Forward mode: eliminate intermediate vertices in topological (ascending-id) order.
# This reproduces the checker's baseline construction -> ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V = int(next(it)); E = int(next(it)); M = int(next(it)); N = int(next(it))
    intermediates = list(range(M, V - N))
    sys.stdout.write(" ".join(map(str, intermediates)) + "\n")


if __name__ == "__main__":
    main()
