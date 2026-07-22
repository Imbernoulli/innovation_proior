# TIER: invalid
# Emits a non-permutation (job 0 repeated N times) -> feasibility check fails -> score 0.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    sys.stdout.write(" ".join(["0"] * N) + "\n")

main()
