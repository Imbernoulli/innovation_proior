# TIER: trivial
"""Reproduce the checker's baseline exactly: compute each target independently
with the classic binary method (double, and add r_0 when the bit is 1), with NO
sharing of registers across targets. Op count = sum(floor(log2 t) + popcount(t) - 1)
so the score is exactly 0.1."""
import sys


def bin_chain_vals(e):
    bits = bin(e)[3:]
    vals = [1]
    v = 1
    for b in bits:
        vals.append(v + v)
        v += v
        if b == "1":
            vals.append(v + 1)
            v += 1
    return vals


def main():
    data = sys.stdin.read().split()
    K = int(data[0])
    targets = [int(x) for x in data[1:1 + K]]
    have = {1: 0}
    ops = []
    for t in targets:
        vals = bin_chain_vals(t)
        for i in range(1, len(vals)):
            v = vals[i]
            prev = vals[i - 1]
            if v == prev * 2:
                ops.append((have[prev], have[prev]))
            else:
                ops.append((have[prev], have[1]))
            have[v] = len(ops)
    out = [str(len(ops))]
    out += ["%d %d" % (a, b) for (a, b) in ops]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
