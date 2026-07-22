# TIER: invalid
import sys

def main():
    # emit an automaton that accepts nothing useful: one non-accepting state,
    # no accept states -> rejects every positive string -> infeasible -> 0.0
    d = sys.stdin.read().split()
    sys.stdout.write("1\n1 0\n0\n0\n")

main()
