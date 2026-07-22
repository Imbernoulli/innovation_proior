# TIER: invalid
# Emits N copies of index 0: correct token count but massively duplicated /
# missing indices -> infeasible, must score 0.
import sys


def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


main()
