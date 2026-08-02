# TIER: invalid
import sys


def main():
    sys.stdin.read()  # consume input, ignore it
    # Nonsensical artifact: tries to "qualify" supplier 0, which is already
    # qualified from period 1 -- a strict feasibility violation the checker
    # must reject outright.
    out = ["1", "0 1", "0"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
