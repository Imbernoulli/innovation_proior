# TIER: trivial
# Flat surveillance profile -- reproduces the checker's baseline (c = 2, Ratio = 0.1).
import sys

def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    print(" ".join(["1"] * n))

if __name__ == "__main__":
    main()
