# TIER: invalid
# Emits an infeasible capacity vector (sum != T and a zero seat) -> must score 0.
import sys

def main():
    toks = sys.stdin.read().split()
    M = int(toks[1]); T = int(toks[2])
    caps = [0] * M            # zero seats + wrong sum: both feasibility violations
    caps[0] = T + 7
    sys.stdout.write("\n".join(map(str, caps)) + "\n")

if __name__ == "__main__":
    main()
