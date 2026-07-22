# TIER: invalid
"""
Emits a garbage / infeasible decision program: a single leaf that declares
every patient has label 0, regardless of the input. Since every generated
instance contains patients with multiple distinct true labels, this must be
rejected (Ratio 0.0) by the checker's strict feasibility check.
"""
import sys


def main():
    sys.stdin.read()  # consume input, ignore it entirely
    print("1")
    print("LEAF 0")


if __name__ == "__main__":
    main()
