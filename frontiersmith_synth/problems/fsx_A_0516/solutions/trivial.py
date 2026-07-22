# TIER: trivial
"""Literal transcript: emit the developed creature verbatim as the axiom, no rules.
This traces the outline instead of discovering the rule -> gene length = |T| (huge).
It reproduces the checker's literal baseline -> ~0.1."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    T = data[1].split() if len(data) > 1 else []
    sys.stdout.write("0\n")
    sys.stdout.write(" ".join(T) + "\n")
    sys.stdout.write("0\n")


if __name__ == "__main__":
    main()
