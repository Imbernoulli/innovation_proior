# TIER: invalid
# Deliberately infeasible: negative heat output (below every envelope's
# L(t) >= 0) and negative dump. Must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    out = ["-5.0 -1.0" for _ in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
