# TIER: greedy
# Obvious recipe: trust the registry's suggested courier set and print it verbatim.
# No check for commuting / triangular / small-order structure -- this is exactly the
# subgroup-avoidance blind spot the problem is designed to punish.
import sys


def main():
    data = [ln for ln in sys.stdin.read().split("\n") if ln.strip() != ""]
    p, k, r = map(int, data[0].split())
    hint_lines = data[1:1 + k]
    print("\n".join(hint_lines))


if __name__ == "__main__":
    main()
