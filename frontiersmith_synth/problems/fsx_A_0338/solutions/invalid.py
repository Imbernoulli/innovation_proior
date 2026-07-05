# TIER: invalid
# Emits a single garbage stage that does not reconstruct the tensor.  The exact
# equality gate rejects it -> Ratio 0.0.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    u = [1] * I; v = [1] * J; w = [1] * K
    print(1)
    print(" ".join(str(x) for x in u + v + w))


if __name__ == "__main__":
    main()
