# TIER: invalid
# Emits an infeasible artifact: claims more swaps than the budget allows,
# with self-loop endpoints. Must score Ratio: 0.0.
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); budget = int(tok[3])
    s = budget + 5
    lines = [str(s)]
    for i in range(s):
        lines.append("0 0 1 2")   # self-loop 'edge' (0,0) never exists
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
