# TIER: trivial
# Baseline guess: the squared speed v^2 = vx^2 + vy^2.  It is only PARTIALLY
# conserved (v^2 = 2E + 2 mu/r), so it reproduces the grader's internal baseline
# -> ratio ~0.1.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x3**2 + x4**2\n")


if __name__ == "__main__":
    main()
