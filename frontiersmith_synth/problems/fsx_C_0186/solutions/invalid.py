# TIER: invalid
# Cram model 0 into every slot -> massive row/column collisions (and it clobbers
# givens). Infeasible -> checker must score 0.
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    lines = []
    for i in range(n):
        lines.append(" ".join("0" for _ in range(n)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
