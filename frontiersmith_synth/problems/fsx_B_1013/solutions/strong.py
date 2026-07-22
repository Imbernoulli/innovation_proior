# TIER: strong
# Insight: the greedy trick "use a prime r near the cap" only maximizes
# THIS curve's own petal count w = R/gcd(R,r); it says nothing about how w
# interacts with the FIXED sampling budget S. What actually controls
# whether a curve's S samples spread across its whole (maximal-width) band
# or alias onto a handful of repeated radii is gcd(w, S). So: also use the
# largest feasible pen offset, but search over a few candidate prime
# rolling radii near the cap (and a small shift of the fixed-gear radius)
# for a curve that is BOTH coprime-with-r (maximal per-curve symmetry) AND
# coprime with S (so gcd(w,S)=1, guaranteeing all S samples land on
# distinct ANGLE INDICES, finely sweeping the whole band -- radii themselves
# still pair up under cos()'s left-right symmetry, which is harmless here).
# When the search window is too tight to find such a curve, fall back to
# minimizing gcd(w,S) and then pick the PHASE that makes that curve's few
# surviving (aliased) angles land in bins NOT yet covered by curves already
# placed -- optimizing the portfolio jointly instead of tuning each curve
# in isolation.
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


def primes_leq(n, count):
    out = []
    cand = n
    while cand > 1 and len(out) < count:
        if is_prime(cand):
            out.append(cand)
        cand -= 1
    return out


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


def bin_of(rho, r_in, K, binwidth):
    pos = (rho - r_in) / binwidth
    b = int(pos)
    if b < 0:
        b = 0
    if b >= K:
        b = K - 1
    return b


def curve_bins(R, r, d, p, S, r_in, K, binwidth):
    g = math.gcd(R, r)
    w = R // g
    C = R - r
    bins = set()
    for k in range(S):
        idx = (w * k + p) % S
        rho = C + d * math.cos(2.0 * math.pi * idx / S)
        bins.add(bin_of(rho, r_in, K, binwidth))
    return bins


def choose_gear(C, cap, r_in, r_out, M, S):
    """Search a handful of prime rolling radii near the cap, and a small
    shift of the fixed-gear radius, for (R,r) with gcd(R,r)=1 AND
    gcd(R,S)=1 (w=R exactly when coprime). Falls back to minimizing
    gcd(w,S) over the same search space."""
    best = None
    best_key = None
    for r in primes_leq(min(cap, M - C - 2), 4):
        if r < 2:
            continue
        for delta in delta_seq(10):
            R = C + r + delta
            if R <= r or R > M:
                continue
            Cp = R - r
            if not (r_in <= Cp - r and Cp + r <= r_out):
                continue
            g = math.gcd(R, r)
            if g != 1:
                continue
            w = R  # coprime, so w = R
            gs = math.gcd(w, S)
            if gs == 1:
                return (R, r)
            key = (gs, abs(delta), -r)
            if best_key is None or key < best_key:
                best_key = key
                best = (R, r)
    return best


def main():
    r_in, r_out, K, Q, S, M = (int(x) for x in sys.stdin.read().split()[:6])
    binwidth = (r_out - r_in) / K

    centers, caps = band_centers_and_caps(r_in, r_out, Q)
    curves = []
    covered = set()
    for C, cap in zip(centers, caps):
        gear = choose_gear(C, cap, r_in, r_out, M, S)
        if gear is None:
            r = 2
            R = min(M, C + r)
            if R <= r:
                continue
        else:
            R, r = gear
        Cp = R - r
        d = max(1, min(r, Cp - r_in, r_out - Cp))

        g = math.gcd(R, r)
        w = R // g
        gs = math.gcd(w, S)

        if gs == 1:
            p = 0
        else:
            best_p, best_new = 0, -1
            for p_cand in range(S):
                b = curve_bins(R, r, d, p_cand, S, r_in, K, binwidth)
                new_cnt = len(b - covered)
                if new_cnt > best_new:
                    best_new = new_cnt
                    best_p = p_cand
            p = best_p

        curves.append((R, r, d, p))
        covered |= curve_bins(R, r, d, p, S, r_in, K, binwidth)

    if not curves:
        r = 2
        R = min(M, r_in + 1 + r)
        d = 1
        curves = [(R, r, d, 0)]

    print(len(curves))
    for (R, r, d, p) in curves:
        print(f"{R} {r} {d} {p}")


if __name__ == "__main__":
    main()
