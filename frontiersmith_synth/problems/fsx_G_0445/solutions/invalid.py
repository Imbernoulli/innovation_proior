# TIER: invalid
import sys


def main():
    tok = sys.stdin.read().split()
    n1 = int(tok[0]); n2 = int(tok[1]); n3 = int(tok[2])
    # Emit a single all-zero product -> reconstructs the zero tensor, which does
    # NOT equal the (dense, nonzero) target tensor -> checker must score 0.
    u = [0] * n1
    v = [0] * n2
    w = [0] * n3
    sys.stdout.write("1\n" + " ".join(map(str, u + v + w)) + "\n")


main()
