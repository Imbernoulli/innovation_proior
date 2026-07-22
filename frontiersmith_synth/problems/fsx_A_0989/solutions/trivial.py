# TIER: trivial
import sys

BASELINE_FORMULATION = 0  # must match verify.py's BASELINE_FORMULATION

def main():
    toks = sys.stdin.read().split()
    R = int(toks[0])
    out = [str(BASELINE_FORMULATION)] * (R * R)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
