# TIER: trivial
# Baseline construction: run the experiments in input order (reproduces the
# checker's internal baseline B, so it scores ~0.1 by construction).
import sys


def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    sys.stdout.write("\n".join(str(i) for i in range(n)) + "\n")


main()
