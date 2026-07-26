# TIER: invalid
"""Infeasible on purpose: repeatedly "cools" cell 0, which is always a seed
(already solid from stage 1 onward -- and solid from the very start), so the
very first action violates "you may only cool a currently liquid cell"."""
import sys


def main():
    data = sys.stdin.read().split()
    K = int(data[1])
    sys.stdout.write("\n".join(["0"] * K) + "\n")


if __name__ == "__main__":
    main()
