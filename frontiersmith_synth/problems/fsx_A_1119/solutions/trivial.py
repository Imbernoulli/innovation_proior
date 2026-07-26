# TIER: trivial
"""Do nothing: never spend the single per-stage cooling action. Only the
seed cells ever solidify, so this exactly reproduces the checker's own
internal baseline construction."""
import sys


def main():
    data = sys.stdin.read().split()
    K = int(data[1])
    sys.stdout.write("\n".join(["-1"] * K) + "\n")


if __name__ == "__main__":
    main()
