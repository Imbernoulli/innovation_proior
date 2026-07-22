# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    R = int(next(it)); C = int(next(it)); Tmax = int(next(it))
    move = int(next(it)); Rfill = int(next(it))
    S = int(next(it))
    for _ in range(S):
        next(it); next(it)
    L = int(next(it))
    for _ in range(L):
        next(it); next(it)
    K = int(next(it))
    for _ in range(K):
        next(it); next(it); next(it)
    # delay grid ignored -- trivial does nothing at all

    out = [str(K)]
    for _ in range(K):
        out.append("0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
