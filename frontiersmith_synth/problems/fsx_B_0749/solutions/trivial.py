# TIER: trivial
"""Do nothing: never commit any capital, leave it all as idle cash.
This reproduces the checker's own baseline construction (B = C0)."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0]); K = int(data[1])
    out = []
    zero_row = " ".join("0" for _ in range(K))
    for _ in range(T):
        out.append(zero_row)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
