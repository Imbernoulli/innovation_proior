# TIER: invalid
import sys

def main():
    # Ignore the instance entirely and emit a syntactically well-formed circuit
    # that computes x^2 -- wrong for every test (all targets have degree >= 4
    # and a nonzero leading coefficient), so the equivalence check must reject it.
    sys.stdin.read()
    sys.stdout.write("1\nM 0 0\n")

if __name__ == "__main__":
    main()
