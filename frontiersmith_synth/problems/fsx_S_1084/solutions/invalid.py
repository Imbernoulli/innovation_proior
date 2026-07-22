# TIER: invalid
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

    # Emit an out-of-bounds drop for plane 0 (and empty plans for the rest) --
    # a clean, unambiguous feasibility violation the checker must reject.
    out = [str(K)]
    for i in range(K):
        if i == 0:
            out.append("1")
            out.append(f"{R + 500} {C + 500} D")
        else:
            out.append("0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
