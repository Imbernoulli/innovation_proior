# TIER: invalid
import sys

# Emits a well-formed but functionally wrong program: zero operations. Every instance in
# this family plants at least one non-trivial cycle (a relay cycle or a 2-cycle), so
# perm is never the identity and doing nothing always leaves the final arrangement wrong
# -- the checker's exact-equivalence gate rejects it with Ratio: 0.0.


def main():
    sys.stdin.read()
    print(0)


if __name__ == "__main__":
    main()
