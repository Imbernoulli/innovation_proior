# TIER: trivial
"""
Reproduces the checker's own baseline: predict the constant mean of the
training throughputs for every held-out batch size, ignoring x entirely.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    idx += 1  # test id
    idx += 4  # C W F D
    ys = []
    for _ in range(n):
        idx += 1  # x
        y = float(data[idx]); idx += 1
        ys.append(y)
    mean_y = sum(ys) / len(ys)
    print(repr(mean_y))


if __name__ == "__main__":
    main()
