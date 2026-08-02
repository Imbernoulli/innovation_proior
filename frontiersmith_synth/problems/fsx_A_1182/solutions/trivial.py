# TIER: trivial
"""Uniform prior over every infected candidate -- 'no idea who patient zero is'.
This exactly reproduces the checker's own baseline construction, so it scores
~0.1 on every case by design."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M = map(int, data[1].split())
    ptr = 2 + M
    T = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    cand = list(map(int, data[ptr].split()))

    out = sys.stdout
    out.write("%d\n" % K)
    out.write(" ".join(["1"] * K))
    out.write("\n")


if __name__ == "__main__":
    main()
