# TIER: greedy
# Obvious first idea: use the full curve budget, spread band-centers evenly
# across the annulus, and for each curve pick the LARGEST feasible pen
# offset AND a PRIME rolling radius near that cap (the classic textbook
# "use a prime gear-tooth count so the pattern never repeats" trick: a
# prime r guarantees gcd(R,r)=1 for almost any R, giving this curve its
# maximal per-curve petal count w=R). This never looks at how R interacts
# with the fixed sampling budget S, and never touches phase.
import sys, math


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def largest_prime_leq(n):
    for cand in range(n, 1, -1):
        if is_prime(cand):
            return cand
    return None


def band_centers_and_caps(r_in, r_out, Q):
    span = r_out - r_in
    seg = span // Q
    centers, caps = [], []
    for i in range(Q):
        lo = r_in + i * seg
        hi = r_in + (i + 1) * seg if i < Q - 1 else r_out
        c = (lo + hi) // 2
        cap = max(1, min(c - lo, hi - c, (hi - lo) // 2))
        centers.append(c)
        caps.append(cap)
    return centers, caps


def delta_seq(bound):
    yield 0
    for k in range(1, bound + 1):
        yield k
        yield -k


def main():
    r_in, r_out, K, Q, S, M = (int(x) for x in sys.stdin.read().split()[:6])
    centers, caps = band_centers_and_caps(r_in, r_out, Q)
    curves = []
    for C, cap in zip(centers, caps):
        r = largest_prime_leq(min(cap, M - C - 2))
        if r is None or r < 2:
            r = 2
        chosen_R = None
        for delta in delta_seq(2):
            R = C + r + delta
            if R <= r or R > M:
                continue
            Cp = R - r
            if not (r_in <= Cp - r and Cp + r <= r_out):
                continue
            if math.gcd(R, r) == 1:
                chosen_R = R
                break
        if chosen_R is None:
            chosen_R = C + r
        R = chosen_R
        Cp = R - r
        d = max(1, min(r, Cp - r_in, r_out - Cp))
        p = 0
        curves.append((R, r, d, p))

    print(len(curves))
    for (R, r, d, p) in curves:
        print(f"{R} {r} {d} {p}")


if __name__ == "__main__":
    main()
