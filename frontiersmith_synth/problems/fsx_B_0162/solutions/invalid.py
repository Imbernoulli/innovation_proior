# TIER: invalid
"""Emits a SWAP on out-of-range qubits and executes nothing -> infeasible -> 0.0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("S 999 1000\n")


if __name__ == "__main__":
    main()
