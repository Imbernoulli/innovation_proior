# TIER: trivial
"""
Reads the training rows and emits a single CONSTANT rate: the geometric mean
of the observed training R.  Ignores every input variable, so it reproduces
the checker's own internal baseline almost exactly -> Ratio ~ 0.1.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    logs = []
    for i in range(n):
        row = vals[i * 6:(i + 1) * 6]
        R = float(row[5])
        logs.append(math.log(R))
    const = math.exp(sum(logs) / len(logs)) if logs else 1e-3
    # spaced tokens so a nan/inf-flood adversarial test can corrupt the literal
    print("%.10e + 0.0 * Cl" % const)


if __name__ == "__main__":
    main()
