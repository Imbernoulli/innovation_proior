# TIER: trivial
# Do-nothing baseline: predict the constant mean of the training flow (this is
# EXACTLY the checker's own internal baseline) -- ignores precip/temp/proxy
# entirely -> reproduces ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.3"); return
    n = int(data[0])
    vals = data[2:]
    ys = []
    for i in range(n):
        ys.append(float(vals[4 * i + 2]))
    mean_y = sum(ys) / len(ys) if ys else 0.3
    print("OUT %.6f" % mean_y)


if __name__ == "__main__":
    main()
