# TIER: greedy
import sys

def conv(a, b, n, p):
    res = [0] * n
    for idx in range(n):
        ai = a[idx]
        if ai == 0:
            continue
        for j in range(n):
            k = idx + j
            if k >= n:
                k -= n
            res[k] = (res[k] + ai * b[j]) % p
    return res

def poly_pow(c0, t, n, p):
    result = [0] * n
    result[0] = 1 % p
    base = c0[:]
    while t > 0:
        if t & 1:
            result = conv(result, base, n, p)
        t >>= 1
        if t > 0:
            base = conv(base, base, n, p)
    return result

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); p = int(next(it)); r = int(next(it))
    s = int(next(it)); k = int(next(it))
    coeffs = [int(next(it)) for _ in range(2 * r + 1)]
    targets = []
    for _ in range(k):
        t = int(next(it)); pos = int(next(it)); val = int(next(it)); w = int(next(it))
        targets.append((t, pos, val, w))

    c0 = [0] * n
    for d in range(-r, r + 1):
        c0[d % n] = coeffs[d + r] % p

    distinct_t = sorted(set(t for (t, _, _, _) in targets))
    Ct_cache = {t: poly_pow(c0, t, n, p) for t in distinct_t}

    # "Obvious" recipe: handle the most important portraits first. For each target, use
    # exact algebra to solve for a NEW seed cell at exactly that target's own position,
    # accounting for every seed already planted so far (this looks properly principled --
    # it is a real, correct solve of that one equation given the current state). What it
    # never does is go back and re-balance portraits it already locked in once later seeds
    # start contributing to their (generally dense, mixed-by-time) linear functional too.
    order = sorted(range(k), key=lambda i: (-targets[i][3], i))

    support = {}  # position -> value
    for idx in order:
        if len(support) >= s:
            break
        t, pos, val, w = targets[idx]
        if pos in support:
            continue
        Ct = Ct_cache[t]
        a_self = Ct[0] % p
        if a_self == 0:
            continue  # this equation offers no usable pivot at its own cell; skip it
        partial = 0
        for m, v in support.items():
            partial = (partial + Ct[(pos - m) % n] * v) % p
        needed = (val - partial) % p
        value = (needed * pow(a_self, p - 2, p)) % p
        support[pos] = value

    x0 = [0] * n
    for m, v in support.items():
        x0[m] = v
    print(" ".join(map(str, x0)))

main()
