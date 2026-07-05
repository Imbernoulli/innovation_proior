# TIER: strong
# Same structure recovery H = (x^2 + p x + q)^k, build b (2 mults), then compute
# b^k with a repeated-squaring addition chain: floor(log2 k)+popcount(k)-1 mults.
# Total ~ 2 + O(log k) << k+1 (greedy) << d (Horner).  Still NOT optimal --
# shortest addition chains can be shorter -- so headroom remains.
import sys


def iroot(n, k):
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
        x2 = emit("mul", "x", "x")
        px = emit("mul", str(p), "x")
        s = emit("add", x2, px)
        b = emit("add", s, str(q))
        # b^k via right-to-left binary exponentiation
        result = None
        cur = b
        e = k
        while e > 0:
            if e & 1:
                result = cur if result is None else emit("mul", result, cur)
            e >>= 1
            if e > 0:
                cur = emit("mul", cur, cur)
        last = result
    else:
        last = horner(c, d, emit)

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")


if __name__ == "__main__":
    main()
