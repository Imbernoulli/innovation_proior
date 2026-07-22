# TIER: greedy
# Textbook baby-step-giant-step: build a table sized ~sqrt(p-1) around the given primitive
# root g and look up EVERY target uniformly via discrete log, with no attempt to notice that
# most targets live in a small hidden coset. This is the "obvious" precomputation-table
# recipe -- it always finds a short (<=2 factor) recipe per target, but the table itself is
# sized for the whole group, so it pays sqrt(p-1) parts regardless of how few distinct
# "shapes" the targets actually have.
import sys, math


def main():
    data = sys.stdin.read().split()
    pos = 0
    p = int(data[pos]); pos += 1
    g = int(data[pos]); pos += 1
    LAMBDA = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    targets = [int(x) for x in data[pos:pos + T]]

    N = p - 1
    B = math.isqrt(N) + 1

    baby = {}
    cur = 1
    for i in range(B):
        if cur not in baby:
            baby[cur] = i
        cur = (cur * g) % p

    gB = pow(g, B, p)          # g^{+B}: what we STOCK and use in the recipe
    giant_step = pow(gB, p - 2, p)  # g^{-B}: only used to drive the meet-in-the-middle search

    table = [pow(g, i, p) for i in range(B)] + [gB]
    m = len(table)
    giant_idx = m  # 1-based

    lines_targets = []
    steps = N // B + 2
    for t in targets:
        curv = t
        found = None
        for j in range(steps):
            if curv in baby:
                found = (baby[curv], j)
                break
            curv = (curv * giant_step) % p
        if found is None:
            # should not happen since g is a primitive root of the full group
            raise RuntimeError("discrete log not found")
        i, j = found
        # identity: t == g^i * (g^B)^j  (since t * (g^{-B})^j == g^i by the search above)
        if j == 0:
            lines_targets.append("1 %d 1" % (i + 1))
        else:
            lines_targets.append("2 %d 1 %d %d" % (i + 1, giant_idx, j))

    out = []
    out.append(str(m))
    out.append(" ".join(str(x) for x in table))
    out.extend(lines_targets)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
