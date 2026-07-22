# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); R = int(next(it)); cap = int(next(it))
    chain = [int(next(it)) for _ in range(D)]
    free = [int(next(it)) for _ in range(R)]

    out = []
    # Garbage: repeat the same onset k_i times for every instrument -- triggers the
    # "duplicate onset within instrument" feasibility check immediately.
    for k in chain + free:
        out.append(" ".join("0" for _ in range(k)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
