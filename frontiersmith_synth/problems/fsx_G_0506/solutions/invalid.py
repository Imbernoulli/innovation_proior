# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1]) if len(toks) > 1 else 1
    sys.stdout.write("0\n")
    sys.stdout.write("OUT " + " ".join(["#0"] * m) + "\n")


if __name__ == "__main__":
    main()
