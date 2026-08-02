# TIER: greedy
"""The obvious textbook recipe: always resynchronize on the next `;`,
regardless of which construct is open. Fast, and fine for isolated
statement-level errors -- but inside a paren group or a for-header it
either swallows a nearby opening bracket (whose own closer later reads as
a phantom mismatch) or swallows a second nearby `?` outright (a missed
error). Same fixed sync set {`;`} for all three contexts."""
import sys


def main():
    sys.stdin.read()  # a single context-blind policy, no per-instance logic
    print("0")  # STMT   : ;
    print("0")  # PAREN  : ;   <-- the trap: parens have no `;` of their own
    print("0")  # FORHDR : ;


if __name__ == "__main__":
    main()
