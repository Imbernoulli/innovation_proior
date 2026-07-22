# TIER: invalid
import sys

def main():
    toks = sys.stdin.read().split()
    R = int(toks[0])
    # deliberately infeasible: formulation index 6 is out of the published range [0,5]
    out = ["6"] * (R * R)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
