# TIER: strong
# Best-of {binary, CSD} recoding PLUS global common-subexpression elimination:
# every intermediate value is memoized by its exact integer vector, so power-of-two
# multiples of each input (and any coinciding partial sums) are shared across all outputs.
import sys

def csd(a):
    naf = []
    while a > 0:
        if a & 1:
            z = 2 - (a & 3)
            naf.append(z)
            a -= z
        else:
            naf.append(0)
        a >>= 1
    return naf[::-1] if naf else [0]

def cost_bin(a):
    bits = bin(a)[2:]
    return (len(bits) - 1) + (bits.count('1') - 1)

def cost_csd(a):
    d = csd(a)
    nz = sum(1 for x in d if x != 0)
    return (len(d) - 1) + (nz - 1)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    M = [[int(next(it)) for _ in range(n)] for _ in range(m)]

    regs = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    instrs = []
    memo = {regs[i]: i for i in range(n)}

    def emit(vec, text):
        r = memo.get(vec)
        if r is not None:
            return r
        regs.append(vec)
        instrs.append(text)
        rid = len(regs) - 1
        memo[vec] = rid
        return rid

    def dbl(a):
        return emit(tuple(2 * v for v in regs[a]), "DBL %d" % a)

    def add(a, b):
        return emit(tuple(x + y for x, y in zip(regs[a], regs[b])), "ADD %d %d" % (a, b))

    def sub(a, b):
        return emit(tuple(x - y for x, y in zip(regs[a], regs[b])), "SUB %d %d" % (a, b))

    def binmul(a, j):
        cur = j
        for bit in bin(a)[2:][1:]:
            cur = dbl(cur)
            if bit == '1':
                cur = add(cur, j)
        return cur

    def csdmul(a, j):
        d = csd(a)
        cur = j
        for dig in d[1:]:
            cur = dbl(cur)
            if dig == 1:
                cur = add(cur, j)
            elif dig == -1:
                cur = sub(cur, j)
        return cur

    def mul(a, j):
        if cost_csd(a) < cost_bin(a):
            return csdmul(a, j)
        return binmul(a, j)

    outs = []
    for i in range(m):
        items = [(j, M[i][j]) for j in range(n) if M[i][j] != 0]
        pos = [(j, c) for (j, c) in items if c > 0]
        neg = [(j, c) for (j, c) in items if c < 0]
        order = pos + neg
        running = None
        for j, c in order:
            term = mul(abs(c), j)
            if running is None:
                # running starts as the (already-emitted, possibly shared) term itself
                running = term
            else:
                running = add(running, term) if c > 0 else sub(running, term)
        outs.append(running)

    lines = [str(len(instrs))]
    lines.extend(instrs)
    lines.append(" ".join(str(o) for o in outs))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
