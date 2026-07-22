# TIER: greedy
# The textbook recipe: Horner's rule applied blindly to the given coefficients.
#   result = a_n; for i = n-1 downto 0: result = result*x + a_i
# This is correct and beats the naive baseline (n multiplications instead of
# 2n-1), but it treats every coefficient as an opaque number -- it never looks
# for (and never finds) the value-dependent shortcut the instances are built
# around. F = n on every test case, regardless of what the coefficients are.
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

    result = emit("C %d 1" % a[n])
    for i in range(n - 1, -1, -1):
        mul_reg = emit("M %d 0" % result)  # result * x
        if a[i] != 0:
            c_reg = emit("C %d 1" % a[i])
            result = emit("A %d %d" % (mul_reg, c_reg))
        else:
            result = mul_reg

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
