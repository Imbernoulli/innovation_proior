# TIER: greedy
# Per-coefficient best-of {binary, canonical-signed-digit} recoding, no cross-output sharing.
import sys

def csd(a):
    # non-adjacent form (canonical signed digit), MSB-first list of {-1,0,1}
    naf = []
    while a > 0:
        if a & 1:
            z = 2 - (a & 3)   # 1 or -1
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

    def dbl(a):
        regs.append(tuple(2 * v for v in regs[a]))
        instrs.append("DBL %d" % a)
        return len(regs) - 1

    def add(a, b):
        regs.append(tuple(x + y for x, y in zip(regs[a], regs[b])))
        instrs.append("ADD %d %d" % (a, b))
        return len(regs) - 1

    def sub(a, b):
        regs.append(tuple(x - y for x, y in zip(regs[a], regs[b])))
        instrs.append("SUB %d %d" % (a, b))
        return len(regs) - 1

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
