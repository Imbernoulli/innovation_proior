# TIER: invalid
# Emit prices far above PMAX (infeasible schedule) -> checker rejects -> Ratio 0.
import sys

def main():
    tok = sys.stdin.read().split()
    T = int(tok[0])
    sys.stdout.write(" ".join(["999999999"] * T) + "\n")

if __name__ == "__main__":
    main()
