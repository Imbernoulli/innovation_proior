# TIER: invalid
import sys


def main():
    sys.stdin.read()
    # f(x) is a sum of C*(B/sqrt(x^2+B^2)) (strictly positive, since B,C>0)
    # plus two nonnegative terms, so f(x) > 0 everywhere. Emitting the
    # constant-zero program (x - x) is infeasible at every single grid point.
    print("1")
    print("SUB x x")


if __name__ == "__main__":
    main()
