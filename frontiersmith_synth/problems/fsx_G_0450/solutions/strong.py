# TIER: strong
# Searches over synthesis directions AND qubit relabelings: for the permutation and
# its inverse, and for several bit-order relabelings (conjugations), run basic + a
# bidirectional MMD, remap the gates back, peephole-reduce, and keep the smallest
# circuit that still verifies. Qubit ordering is a hard sub-problem, so this
# reliably beats the single-strategy greedy without reaching the (unknown) optimum.
import sys, random


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
        ns = i & ~t[i]; b = 0
        while ns:
            if ns & 1:
                cmask = t[i]; mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                gates.append((b, cmask))
            ns >>= 1; b += 1
        nc = t[i] & ~i; b = 0
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
    out_gates = []; in_gates = []
    for i in range(N):
        v = t[i]; u = t.index(i)
        if bin(i ^ v).count("1") <= bin(i ^ u).count("1"):
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


def relabel_value(x, sigma, n):
    y = 0
    for b in range(n):
        if (x >> b) & 1:
            y |= (1 << sigma[b])
    return y


def conj_perm(perm, sigma, n):
    N = 1 << n
    P = [relabel_value(x, sigma, n) for x in range(N)]
    Pinv = [0] * N
    for x in range(N):
        Pinv[P[x]] = x
    permp = [0] * N
    for xp in range(N):
        permp[xp] = P[perm[Pinv[xp]]]
    return permp


def remap_gates(gates, sigma_inv, n):
    out = []
    for (tgt, cmask) in gates:
        ntgt = sigma_inv[tgt]
        ncm = 0
        for b in range(n):
            if (cmask >> b) & 1:
                ncm |= (1 << sigma_inv[b])
        out.append((ntgt, ncm))
    return out


def peephole(gates):
    g = gates[:]
    changed = True
    while changed:
        changed = False
        out = []; i = 0
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
    inv = inverse(perm, N)

    cands = []
    cands.append(synth_basic(perm, n))
    cands.append(synth_bidir(perm, n))
    cands.append(synth_basic(inv, n)[::-1])
    cands.append(synth_bidir(inv, n)[::-1])

    # deterministic qubit relabelings
    rng = random.Random(12345 + n)
    seen = {tuple(range(n))}
    sigmas = []
    nperm = 16
    while len(sigmas) < nperm:
        s = list(range(n)); rng.shuffle(s)
        ts = tuple(s)
        if ts not in seen:
            seen.add(ts); sigmas.append(s)
    for sigma in sigmas:
        sinv = [0] * n
        for b in range(n):
            sinv[sigma[b]] = b
        permp = conj_perm(perm, sigma, n)
        invp = conj_perm(inv, sigma, n)
        cands.append(remap_gates(synth_basic(permp, n), sinv, n))
        cands.append(remap_gates(synth_bidir(permp, n), sinv, n))
        cands.append(remap_gates(synth_bidir(invp, n)[::-1], sinv, n))

    best = None; bl = 1 << 60
    for g in cands:
        gp = peephole(g)
        if len(gp) < bl and verify(gp, perm, n):
            bl = len(gp); best = gp
    emit(best, n)


if __name__ == "__main__":
    main()
