# TIER: invalid
# Otherwise-correct Horner evaluation, but the first and last output registers
# are swapped -- the program is syntactically well-formed and internally
# consistent, yet computes the WRONG value at (at least) two query positions,
# so the exact-equivalence gate must reject it with Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    int(next(it))  # test_id (unused by this tier)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n + 1)]
    Q = int(next(it))
    q = [int(next(it)) for _ in range(Q)]
    m1 = int(next(it)); idx1 = [int(next(it)) for _ in range(m1)]
    m2 = int(next(it)); idx2 = [int(next(it)) for _ in range(m2)]
    m3 = int(next(it)); idx3 = [int(next(it)) for _ in range(m3)]

    instrs = []

    def emit(op, x, y):
        instrs.append("%s %s %s" % (op, x, y))
        return "r%d" % (len(instrs) - 1)

    def horner(qi):
        xreg = "q%d" % qi
        acc = "a%d" % n
        for k in range(n - 1, -1, -1):
            t = emit("mul", acc, xreg)
            acc = emit("add", t, "a%d" % k)
        return acc

    outputs = [horner(qi) for qi in idx1 + idx2 + idx3]
    outputs[0], outputs[-1] = outputs[-1], outputs[0]  # corrupt

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")
    sys.stdout.write("%d\n" % len(outputs))
    sys.stdout.write("\n".join(r[1:] for r in outputs) + "\n")


if __name__ == "__main__":
    main()
