# TIER: invalid
# Emits an out-of-range schedule (actions far above the power cap) -> must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    out = ["1000000 1000000" for _ in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


main()
