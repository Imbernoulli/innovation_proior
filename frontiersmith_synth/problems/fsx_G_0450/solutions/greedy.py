# TIER: greedy
# Bidirectional MMD (fix each row from whichever of the two sides is cheaper),
# plus a peephole pass that cancels adjacent inverse gates; take the best of
# {basic, bidirectional}. Beats the trivial baseline on most instances.
import sys


def inverse(perm, N):
    inv = [0] * N
    for x in range(N):
        inv[perm[x]] = x
    return inv


def synth_basic(perm, n):
    N = 1 << n
    t = inverse(perm, N)
    gates = []
    for i in range(N):
        ns = i & ~t[i]
        b = 0
        while ns:
            if ns & 1:
                cmask = t[i]; mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                gates.append((b, cmask))
            ns >>= 1; b += 1
        nc = t[i] & ~i
        b = 0
        while nc:
            if nc & 1:
                cmask = i; mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                gates.append((b, cmask))
            nc >>= 1; b += 1
    return gates


def synth_bidir(perm, n):
    N = 1 << n
    t = inverse(perm, N)
    out_gates = []
    in_gates = []
    for i in range(N):
        v = t[i]
        u = t.index(i)
        cost_out = bin(i ^ v).count("1")
        cost_in = bin(i ^ u).count("1")
        if cost_out <= cost_in:
            ns = i & ~v; b = 0
            while ns:
                if ns & 1:
                    cmask = t[i]; mask = 1 << b
                    for x in range(N):
                        if (t[x] & cmask) == cmask:
                            t[x] ^= mask
                    out_gates.append((b, cmask))
                ns >>= 1; b += 1
            nc = t[i] & ~i; b = 0
            while nc:
                if nc & 1:
                    cmask = i; mask = 1 << b
                    for x in range(N):
                        if (t[x] & cmask) == cmask:
                            t[x] ^= mask
                    out_gates.append((b, cmask))
                nc >>= 1; b += 1
        else:
            ns = i & ~u; b = 0
            while ns:
                if ns & 1:
                    uu = t.index(i); cmask = uu; mask = 1 << b
                    new = t[:]
                    for x in range(N):
                        if (x & cmask) == cmask:
                            new[x] = t[x ^ mask]
                    t = new
                    in_gates.append((b, cmask))
                ns >>= 1; b += 1
            uu = t.index(i); nc = uu & ~i; b = 0
            while nc:
                if nc & 1:
                    cmask = i; mask = 1 << b
                    new = t[:]
                    for x in range(N):
                        if (x & cmask) == cmask:
                            new[x] = t[x ^ mask]
                    t = new
                    in_gates.append((b, cmask))
                nc >>= 1; b += 1
    return out_gates + in_gates[::-1]


def peephole(gates):
    g = gates[:]
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(g):
            if i + 1 < len(g) and g[i] == g[i + 1]:
                i += 2; changed = True
            else:
                out.append(g[i]); i += 1
        g = out
    return g


def verify(gates, perm, n):
    N = 1 << n
    for x in range(N):
        y = x
        for (tgt, cmask) in gates:
            if (y & cmask) == cmask:
                y ^= (1 << tgt)
        if y != perm[x]:
            return False
    return True


def emit(gates, n):
    out = [str(len(gates))]
    for (tgt, cmask) in gates:
        controls = [c for c in range(n) if (cmask >> c) & 1]
        out.append(" ".join([str(len(controls)), str(tgt)] + [str(c) for c in controls]))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    N = 1 << n
    perm = [int(v) for v in data[1:1 + N]]
    cands = [synth_basic(perm, n), synth_bidir(perm, n)]
    best = None; bl = 1 << 60
    for g in cands:
        gp = peephole(g)
        if verify(gp, perm, n) and len(gp) < bl:
            bl = len(gp); best = gp
    emit(best, n)


if __name__ == "__main__":
    main()
