# TIER: invalid
# Emits a structurally well-formed but deliberately incomplete plan: it only
# ever covers day 0, so the checker's "plan must cover exactly T days" gate
# rejects it (T is always >= 5 in every test) -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    sys.stdout.write("1\nSTEP P 0\n")


if __name__ == "__main__":
    main()
