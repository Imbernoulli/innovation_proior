# TIER: trivial
# Naive definition of a polynomial: recompute every power x^2..x^d from scratch
# (no reuse), scale each by its coefficient, and add. Nonscalar multiplications =
# sum_{i=2}^d (i-1) = d(d-1)/2 = the checker's B_hi baseline -> score ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    a = [int(x) for x in data[1 : 1 + d + 1]]
    lines = []
    nreg = 1  # r0 = x already exists

    def emit(s):
        nonlocal nreg
        lines.append(s)
        r = nreg
        nreg += 1
        return r

    acc = emit("CON %d" % a[0])
    t = emit("SMUL 0 %d" % a[1])
    acc = emit("ADD %d %d" % (acc, t))
    for i in range(2, d + 1):
        p = emit("MUL 0 0")           # x^2
        for _ in range(3, i + 1):      # extend to x^i, from scratch (no reuse)
            p = emit("MUL %d 0" % p)
        term = emit("SMUL %d %d" % (p, a[i]))
        acc = emit("ADD %d %d" % (acc, term))
    lines.append("RET %d" % acc)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
