# TIER: trivial
# Keep every tier's price unchanged from last cycle (p_i = p0_i). Zero rate change,
# always inside the cap band, no attempt at optimization.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it))
    prices = []
    for _ in range(K):
        p0 = int(next(it)); c = int(next(it)); cap = int(next(it)); B = int(next(it))
        for _ in range(B):
            n = int(next(it)); m = int(next(it))
            for _ in range(m):
                next(it); next(it)
            next(it); next(it)  # tlo thi
        prices.append(p0)
    sys.stdout.write("\n".join(str(p) for p in prices) + "\n")


if __name__ == "__main__":
    main()
