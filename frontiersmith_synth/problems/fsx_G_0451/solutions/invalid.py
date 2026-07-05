# TIER: invalid
"""Infeasible: emits the identity map c_i = a_i (zero gates). This does NOT compute
GF(2^k) multiplication, so the checker's equivalence gate rejects it -> Ratio 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    k = int(toks[0])
    lines = ["0"]
    lines.append(" ".join(str(i) for i in range(k)))   # c_i := a_i (wrong)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
