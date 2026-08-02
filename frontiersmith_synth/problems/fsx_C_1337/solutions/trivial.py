# TIER: trivial
# Build nothing: every hot stream dumps to cold utility, every cold stream is
# heated by hot utility. Reproduces the checker's own baseline B exactly.
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this construction
    print(0)


if __name__ == "__main__":
    main()
