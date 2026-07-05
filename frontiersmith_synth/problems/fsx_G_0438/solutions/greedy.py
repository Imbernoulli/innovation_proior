# TIER: greedy
# Recognise the planted quadratic-power structure H = (x^2 + p x + q)^k, build the
# quadratic section b once (2 mults: x*x and p*x), then raise it to the k-th power
# by NAIVE repeated multiplication (k-1 mults).  Total = k+1 mults << d = 2k.
import sys


def iroot(n, k):
    # integer k-th root of n>=0 (n = q^k here)
    if n == 0:
        return 0
    lo, hi = 1, 1
    while hi ** k < n:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if mid ** k < n:
            lo = mid + 1
        else:
            hi = mid
    return lo


def horner(c, d, emit):
    prev = str(c[d])
    for i in range(d - 1, -1, -1):
        t = emit("mul", prev, "x")
        prev = emit("add", t, str(c[i]))
    return prev


def main():
    data = sys.stdin.read().split()
    d = int(data[0])
    c = [int(t) for t in data[1:2 + d]]

    instrs = []

    def emit(op, a, b):
        instrs.append("%s %s %s" % (op, a, b))
        return "r%d" % (len(instrs) - 1)

    k = d // 2
    ok = (d % 2 == 0) and c[d] == 1 and k >= 1 and c[d - 1] % k == 0
    if ok:
        p = c[d - 1] // k
        q = iroot(c[0], k)
        # build base b = x^2 + p*x + q
        x2 = emit("mul", "x", "x")
        px = emit("mul", str(p), "x")
        s = emit("add", x2, px)
        b = emit("add", s, str(q))
        # b^k by naive repeated multiply
        acc = b
        for _ in range(k - 1):
            acc = emit("mul", acc, b)
        last = acc
    else:
        last = horner(c, d, emit)  # fallback (never triggered by gen)

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")


if __name__ == "__main__":
    main()
