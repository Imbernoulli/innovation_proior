# TIER: invalid
# Emits a program that is NOT a formal identity for p(x): it computes p(x) + 1 (a
# constant off-by-one via Horner). The checker's exact coefficient-wise identity test
# must reject it -> score 0. Demonstrates that "almost correct" still fails feasibility.
import sys


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    a = [int(x) for x in data[1 : 1 + d + 1]]
    lines = []
    nreg = 1

    def emit(s):
        nonlocal nreg
        lines.append(s)
        r = nreg
        nreg += 1
        return r

    acc = emit("CON %d" % a[d])
    for i in range(d - 1, -1, -1):
        m = emit("MUL %d 0" % acc)
        acc = emit("SADD %d %d" % (m, a[i]))
    acc = emit("SADD %d 1" % acc)   # <-- corrupts the result by +1
    lines.append("RET %d" % acc)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
