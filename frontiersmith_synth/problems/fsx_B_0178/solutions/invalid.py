# TIER: invalid
"""Emits an infeasible program (a SWAP on identical hubs, no hand-offs executed).
Must score 0."""
import sys

def main():
    sys.stdin.read()
    sys.stdout.write("SWAP 0 0\nEND\n")

if __name__ == "__main__":
    main()
