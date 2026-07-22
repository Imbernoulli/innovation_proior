# TIER: invalid
"""
Deliberately infeasible: attempts to fire group 0's Combo reaction three
times before ever building any X_0 or Y_0 (and without any Z having been
reserved by construction it also never builds precursors). The very first
firing already violates consumption-depletion feasibility, so the checker
must report Ratio: 0.0.
"""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    tokens = ["0C", "0C", "0C"]
    print(len(tokens))
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
