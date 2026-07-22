# TIER: strong
# INSIGHT: recover the hidden modulus m by brute force over small m, then write
# every exponent as  e_i = q_i*m + r_i.  Build ONE shared chain for the g^m
# powers (a single squaring table of h = g^m, reused across the whole batch) and
# ONE tiny table of residue powers g^r for the few distinct residues; each
# target is then just g^{q_i*m} * g^{r_i}.  Because the planted q_i have low
# popcount, assembling each g^{q_i*m} costs only a couple of multiplies -- the
# batch collapses onto one shared furnace schedule.
import sys

MMAX = 512


def build_slp(m, targets):
    exps = [1]
    pairs = []
    index = {1: 0}

    def emit(a, b):
        v = exps[a] + exps[b]
        i = len(exps)
        exps.append(v)
        pairs.append((a, b))
        if v not in index:
            index[v] = i
        return i

    # squaring table for base g (exponent 1)
    pg = [0]

    def gp(j):
        while len(pg) <= j:
            p = pg[-1]
            pg.append(emit(p, p))
        return pg[j]

    def build_g(e):
        if e in index:
            return index[e]
        bits = [j for j in range(e.bit_length()) if (e >> j) & 1]
        acc = gp(bits[0])
        for j in bits[1:]:
            acc = emit(acc, gp(j))
        return acc

    hidx = build_g(m)            # h = g^m
    ph = [hidx]

    def hp(j):
        while len(ph) <= j:
            p = ph[-1]
            ph.append(emit(p, p))
        return ph[j]

    def build_h(q):
        target = q * m
        if target in index:
            return index[target]
        bits = [j for j in range(q.bit_length()) if (q >> j) & 1]
        acc = hp(bits[0])
        for j in bits[1:]:
            acc = emit(acc, hp(j))
        return acc

    for e in targets:
        if e in index:
            continue
        q = e // m
        r = e % m
        if q == 0:
            build_g(e)
            continue
        hq = build_h(q)          # g^{q*m}
        if r == 0:
            continue             # e already produced
        gr = build_g(r)          # g^r
        emit(hq, gr)             # g^{q*m + r} = e

    return len(pairs), pairs


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    targets = [int(x) for x in data[1:1 + k]]

    best_L = None
    best_pairs = None
    for m in range(2, MMAX + 1):
        L, pairs = build_slp(m, targets)
        if best_L is None or L < best_L:
            best_L = L
            best_pairs = pairs

    outp = [str(best_L)]
    outp += ["%d %d" % (a, b) for (a, b) in best_pairs]
    sys.stdout.write("\n".join(outp) + "\n")


if __name__ == "__main__":
    main()
