# TIER: invalid
# Emits a syntactically-valid straight-line program that computes the WRONG
# polynomial (constant tap corrupted) -> the exact-equivalence gate must reject
# it with Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    c = [int(t) for t in data[1:2 + d]]

    instrs = []

    def emit(op, a, b):
        instrs.append("%s %s %s" % (op, a, b))
        return "r%d" % (len(instrs) - 1)

    c[0] += 1   # corrupt the constant term -> no longer equals H
    prev = str(c[d])
    for i in range(d - 1, -1, -1):
        t = emit("mul", prev, "x")
        prev = emit("add", t, str(c[i]))

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")


if __name__ == "__main__":
    main()
