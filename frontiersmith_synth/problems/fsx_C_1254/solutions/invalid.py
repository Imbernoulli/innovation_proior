# TIER: invalid
"""
Declares a single domain (Du=1, the only legal id is 1) but then assigns every block to
domain id 2 -- out of range for every instance regardless of N/D -- so the checker rejects it
with "block domain id out of range" -> Ratio: 0.0 on every test case.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    N = int(nx())
    nx(); nx(); nx()  # D, K, T
    # rest of the input (L, W, trace rows) intentionally ignored

    out = ["1", " ".join("2" for _ in range(N)), "0"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
