# TIER: trivial
"""
Do the least possible: set every enzyme to the midpoint of its bound,
e_i = e_max_i / 2, completely ignoring the target flux vector. This is
exactly the checker's own internal baseline construction.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    R = int(toks[ptr]); ptr += 1
    ptr += 1  # X0
    e_max = [0.0] * R
    for i in range(R):
        ptr += 1  # parent
        ptr += 1  # yield
        ptr += 1  # kcat
        ptr += 1  # Km
        ptr += 1  # tau
        e_max[i] = float(toks[ptr]); ptr += 1
        ptr += 1  # cost
    out = [str(e_max[i] / 2.0) for i in range(R)]
    print(" ".join(out))


if __name__ == "__main__":
    main()
