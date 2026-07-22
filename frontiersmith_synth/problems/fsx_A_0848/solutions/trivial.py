# TIER: trivial
"""The 'nothing learned' guess: assume the raw codes were never scrambled
(A=pressurize, B=vent, C=polarity, D=null) and that the nudge magnitude is
1 with normal starting polarity, ignoring the logged bursts entirely. This
reproduces the checker's own baseline construction exactly."""
import sys


def main():
    sys.stdin.read()  # consume input, ignored
    print("A B C D 1 1")


if __name__ == "__main__":
    main()
