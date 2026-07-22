# TIER: trivial
# Do-nothing input x = 0 (reproduces the checker baseline -> ratio ~0.1).
import sys


def main():
    tk = sys.stdin.read().split()
    N = int(tk[0])
    sys.stdout.write(" ".join(["0"] * N) + "\n")


if __name__ == "__main__":
    main()
