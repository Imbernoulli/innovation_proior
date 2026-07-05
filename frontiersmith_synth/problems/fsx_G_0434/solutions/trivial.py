# TIER: trivial
# Predict a single constant concentration = the geometric mean of the early
# draws (i.e. the mean of log C).  This reproduces the checker's constant
# baseline in log space -> Ratio ~ 0.1.  Ignores all decay structure.
import sys, math


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    logs = []
    for i in range(n):
        c = float(vals[2 * i + 1])
        logs.append(math.log(max(c, 1e-9)))
    m = sum(logs) / len(logs)
    # constant concentration exp(m); print as a closed-form expression in t
    print("%r + 0.0*t" % math.exp(m))


if __name__ == "__main__":
    main()
