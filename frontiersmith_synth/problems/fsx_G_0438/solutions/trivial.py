# TIER: trivial
# Horner's rule: evaluate H(x)=sum c[i] x^i with exactly d multiplications
# (one per degree step).  This reproduces the checker's baseline -> ratio ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    c = [int(t) for t in data[1:2 + d]]   # c[0..d]

    instrs = []

    def emit(op, a, b):
        instrs.append("%s %s %s" % (op, a, b))
        return "r%d" % (len(instrs) - 1)

    # acc = c[d]; for i=d-1..0: acc = acc*x + c[i]
    prev = str(c[d])              # literal leading coefficient
    for i in range(d - 1, -1, -1):
        t = emit("mul", prev, "x")
        prev = emit("add", t, str(c[i]))

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")


if __name__ == "__main__":
    main()
