# TIER: trivial
# Do-nothing baseline: ignore the time axis entirely and predict the constant
# arithmetic mean of the training tide heights. This reproduces the checker's
# own constant baseline -> Ratio ~ 0.1.
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    ys = [float(vals[2 * i + 1]) for i in range(n)]
    ybar = sum(ys) / len(ys)
    print("%.10g" % ybar)

if __name__ == "__main__":
    main()
