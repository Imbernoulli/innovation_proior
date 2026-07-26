# TIER: invalid
import sys


def main():
    tokens = sys.stdin.read().split()
    G = int(tokens[0]); T = int(tokens[3])
    # Every move blatantly exceeds the speed budget -> infeasible on step 1 for every glider.
    line = " ".join(["100 100"] * T)
    out = "\n".join([line] * G)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
