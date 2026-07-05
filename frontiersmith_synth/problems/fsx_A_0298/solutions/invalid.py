# TIER: invalid
# Emits a single all-zero stage: reconstruction is the zero tensor, which cannot
# equal the dense target -> exact-equality gate fails -> Ratio 0.0.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    lines = ["1", " ".join(["0"] * (a + b + c))]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
