# TIER: invalid
"""
Emits a syntactically well-formed program that reconstructs the WRONG
shape (the target shifted by one unit along x): every op is legal on its
own, but the final assembly fails the checker's exact-equality test, so
this must score 0.0 on every case.
"""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    v = int(data[0])
    if v == 0:
        print("U")
        return
    pts = [(data[1 + 3 * k], data[2 + 3 * k], data[3 + 3 * k]) for k in range(v)]

    out = []

    def emit(line):
        out.append(line)
        return len(out) - 1

    unit = emit("U")
    leaves = []
    for (x, y, z) in pts:
        xi = int(x) + 1  # deliberate off-by-one -> wrong shape
        t = emit("T %d %d %s %s" % (unit, xi, y.decode(), z.decode()))
        leaves.append(t)

    level = leaves
    while len(level) > 1:
        nxt = []
        i, n = 0, len(level)
        while i + 1 < n:
            nxt.append(emit("M %d %d" % (level[i], level[i + 1])))
            i += 2
        if i < n:
            nxt.append(level[i])
        level = nxt

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
