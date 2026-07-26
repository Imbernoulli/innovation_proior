# TIER: trivial
# Naive per-point evaluation: for EVERY query, independently build the power
# ladder x^2..x^n from scratch (n-1 mults) then dot with coefficients (n more
# mults) -- 2n-1 multiplications per point, no sharing across queries at all.
# This reproduces the checker's own baseline construction -> ratio ~0.1.
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

    def eval_naive(qi):
        xreg = "q%d" % qi
        powers = [xreg]
        for _ in range(2, n + 1):
            powers.append(emit("mul", powers[-1], xreg))
        terms = ["a0"]
        for k in range(1, n + 1):
            terms.append(emit("mul", "a%d" % k, powers[k - 1]))
        acc = terms[0]
        for t in terms[1:]:
            acc = emit("add", acc, t)
        return acc

    outputs = [eval_naive(qi) for qi in idx1 + idx2 + idx3]

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")
    sys.stdout.write("%d\n" % len(outputs))
    sys.stdout.write("\n".join(r[1:] for r in outputs) + "\n")


if __name__ == "__main__":
    main()
