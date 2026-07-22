# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); base = int(next(it))
    # emit an out-of-range digit (== base, valid range is [0, base-1]) everywhere;
    # guaranteed feasibility failure regardless of hints
    out_lines = []
    for _r in range(R):
        out_lines.append(" ".join([str(base)] * C))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
