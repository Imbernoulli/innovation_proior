# TIER: trivial
# The do-nothing baseline: every approach gets the same constant priority
# (a permanent tie, always resolved to approach 0). This is EXACTLY the
# checker's own internal baseline construction, so it scores the calibrated
# ~0.1 by definition.
import sys


def main():
    sys.stdin.read()  # training log is ignored
    print("PRIORITY 0.0 * q")


if __name__ == "__main__":
    main()
