# TIER: trivial
"""Do-nothing stencil: every gear tooth is 0. This is exactly the checker's
own baseline construction, so it reproduces the ~0.1 reference score."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    print(' '.join(['0'] * N))


if __name__ == "__main__":
    main()
