# TIER: invalid
# Deliberately INFEASIBLE: assigns every junction to station 0 (count0=n, count1=0),
# violating the balanced-bisection rule -> must score Ratio 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")

if __name__ == "__main__":
    main()
