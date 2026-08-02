# TIER: trivial
"""
Trivial baseline: spend ZERO perturbation queries -- just report "nothing
requested". This reproduces the checker's own internal baseline B exactly
(by construction), so it scores ~0.1.
"""
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this tier
    print(0)
    print()


if __name__ == "__main__":
    main()
