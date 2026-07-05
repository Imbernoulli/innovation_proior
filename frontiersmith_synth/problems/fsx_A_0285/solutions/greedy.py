# TIER: greedy
# Uniform relay density: one team per segment. Flattest "obvious" plan; its
# autocorrelation is a triangle peaking at the center, giving c = 2 (beats the
# half-block baseline, scores ~0.2).
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write(" ".join(["1"] * n) + "\n")


if __name__ == "__main__":
    main()
