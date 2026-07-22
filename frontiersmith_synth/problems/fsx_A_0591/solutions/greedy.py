# TIER: greedy
# Horner's rule: the textbook "optimal" scheme, one online multiply per degree -> d
# nonscalar multiplications. It is optimal only when the coefficients are UNKNOWN;
# here they are known offline, so Horner is the trap -- beaten by preconditioning.
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
        m = emit("MUL %d 0" % acc)      # acc * x   (nonscalar)
        acc = emit("SADD %d %d" % (m, a[i]))
    lines.append("RET %d" % acc)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
