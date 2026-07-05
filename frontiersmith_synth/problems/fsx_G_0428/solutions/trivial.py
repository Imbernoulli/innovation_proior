# TIER: trivial
# Baseline: predict a single constant = the LAST observed term.
# Reproduces the grader's internal baseline -> ~0.1.
import sys


def main():
    data = sys.stdin.read().split("\n")
    T = int(data[0].split()[0])
    last = None
    for ln in data[1:1 + T]:
        p = ln.split()
        if len(p) >= 2:
            last = float(p[1])
    sys.stdout.write(repr(last if last is not None else 0.0) + "\n")


if __name__ == "__main__":
    main()
