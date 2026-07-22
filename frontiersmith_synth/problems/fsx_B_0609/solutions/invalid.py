# TIER: invalid
# Emits a broken layout: positions are NOT strictly increasing (and out of range),
# so the checker's feasibility gate must reject it -> 0.
import sys


def main():
    d = sys.stdin.buffer.read().split()
    N = int(d[0])
    out = ["%d 0" % N, " ".join(["0"] * N)]   # all keys at cell 0 -> not increasing
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
