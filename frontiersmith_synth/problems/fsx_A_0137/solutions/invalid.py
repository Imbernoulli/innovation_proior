# TIER: invalid
# Emits an infeasible artifact: depot slots outside [0, M] (and too many of them).
# The checker must reject this -> Ratio 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    n, M = int(data[0]), int(data[1])
    A = [M + 5 + i for i in range(n)]     # all out of range
    out = [str(len(A))] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
