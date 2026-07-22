# TIER: strong
# Insight: the hidden subgroup H is the UNIQUE subgroup of (Z/pZ)^* whose order is the
# "small" cofactor of p-1 -- concretely, if Q is the LARGEST prime factor of p-1, then
# o = (p-1)/Q is H's exact order and h = g^Q mod p is an exact generator of H (this holds
# by construction: the instance planted H with order o coprime to, and much smaller than,
# the remaining large prime factor Q). Stock the table with just the coset representative
# (target 1, guaranteed clean) and this single generator h. Every OTHER clean target's ratio
# to the representative is an exact power of h (found by a tiny brute-force discrete log
# inside the small subgroup) -- a 1-2 factor recipe. Every target whose ratio fails the
# exact membership test r^o == 1 (mod p) is a decoy: give it its own dedicated part instead
# of wasting the compact recipe machinery on it. The table stays tiny (encodes the subgroup,
# not the whole group) while almost every recipe stays short.
import sys


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def main():
    data = sys.stdin.read().split()
    pos = 0
    p = int(data[pos]); pos += 1
    g = int(data[pos]); pos += 1
    LAMBDA = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    targets = [int(x) for x in data[pos:pos + T]]

    N = p - 1
    fac = factorize(N)
    Qstar = max(fac.keys())
    o = N // Qstar
    h = pow(g, Qstar, p)
    # sanity: h has order dividing o (guaranteed by construction of the instance)

    c = targets[0]
    cinv = pow(c, p - 2, p)

    # precompute powers of h for brute-force discrete log inside the small subgroup H
    hpow = [1] * o
    v = 1
    for i in range(o):
        hpow[i] = v
        v = (v * h) % p
    hindex = {val: i for i, val in enumerate(hpow)}

    table = [c, h]  # 1-based idx 1 = c, idx 2 = h
    lines = [None] * T
    for ti in range(T):
        t = targets[ti]
        if ti == 0:
            lines[ti] = "1 1 1"
            continue
        r = (t * cinv) % p
        if pow(r, o, p) == 1 and r in hindex:
            a = hindex[r]
            if a == 0:
                lines[ti] = "1 1 1"
            else:
                lines[ti] = "2 1 1 2 %d" % a
        else:
            table.append(t)
            idx = len(table)
            lines[ti] = "1 %d 1" % idx

    m = len(table)
    out = [str(m), " ".join(str(x) for x in table)]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
