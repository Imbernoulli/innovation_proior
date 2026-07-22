# TIER: trivial
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

    # Plant a single seed: try every (target i, planting cell j) pair, solve target i
    # exactly at cell j, and see how many total targets that one seed happens to hit.
    # Do not chase the rest -- one honest, well-aimed seed and stop.
    best_matched = -1.0
    best_x0 = [0] * n
    for i, (ti, posi, vali, wi) in enumerate(targets):
        Ci = Ct_cache[ti]
        for j in range(n):
            a_j = Ci[(posi - j) % n] % p
            if a_j == 0:
                continue
            value = (vali * pow(a_j, p - 2, p)) % p
            matched = 0.0
            for (tj_, posj, valj, wj) in targets:
                Cj = Ct_cache[tj_]
                predicted = (Cj[(posj - j) % n] * value) % p
                if predicted == valj:
                    matched += wj
            if matched > best_matched:
                best_matched = matched
                x0 = [0] * n
                x0[j] = value
                best_x0 = x0

    print(" ".join(map(str, best_x0)))

main()
