# TIER: trivial
# Naive "power chain, then scale" construction: build x^2..x^n by repeated
# multiplication by x (n-1 mults), then multiply EVERY monomial a_i*x^i
# (i=1..n) by its coefficient unconditionally, even if a_i happens to be zero
# (n more mults), and sum. Total F = 2n-1 -- exactly the checker's own
# baseline B, so this always scores Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = [int(v) for v in data[1:1 + n + 1]]

    lines = []
    nxt = 1

    def emit(s):
        nonlocal nxt
        lines.append(s)
        r = nxt
        nxt += 1
        return r

    pow_reg = [None] * (n + 1)
    pow_reg[0] = None  # unused (constant term handled separately)
    pow_reg[1] = 0     # x itself is register 0
    for i in range(2, n + 1):
        pow_reg[i] = emit("M %d 0" % pow_reg[i - 1])  # x^i = x^(i-1) * x

    sum_reg = emit("C %d 1" % a[0])
    for i in range(1, n + 1):
        c_reg = emit("C %d 1" % a[i])
        term_reg = emit("M %d %d" % (c_reg, pow_reg[i]))
        sum_reg = emit("A %d %d" % (sum_reg, term_reg))

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
