# TIER: invalid
# Emits a well-formed but WRONG program (returns x0 + x1), which does not compute
# the hidden continuant -> equivalence gate fails -> score 0.
import sys

def main():
    sys.stdin.read()
    sys.stdout.write("1\nadd 0 1\n")

if __name__ == "__main__":
    main()
