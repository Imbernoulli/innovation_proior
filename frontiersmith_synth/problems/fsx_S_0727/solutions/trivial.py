# TIER: trivial
# Build each target completely independently via the standard left-to-right
# binary (double & add) method, starting from register 0 (value 1) every
# time. Never shares anything across targets. This reproduces exactly the
# checker's own internal baseline B.
import sys


def build_plain(v, ops):
    if v == 1:
        return 0
    bits = bin(v)[2:]
    cur = 0
    for b in bits[1:]:
        ops.append((cur, cur))
        cur = len(ops)
        if b == '1':
            ops.append((cur, 0))
            cur = len(ops)
    return cur


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); B0 = int(next(it))
    targets = [int(next(it)) for _ in range(N)]

    ops = []
    for v in targets:
        build_plain(v, ops)

    out = [str(len(ops))]
    out.extend("%d %d" % (i, j) for (i, j) in ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
