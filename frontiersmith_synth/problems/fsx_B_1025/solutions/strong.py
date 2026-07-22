# TIER: strong
# Exploits the two orthogonal descriptors the target actually decomposes into:
#  1. echo DENSITY is governed by which sums of delay lengths are reachable (every
#     echo arrival time is n = sum k_i*L_i for nonnegative integers k_i, since each
#     pass through line i adds L_i samples of delay) -- so we greedily build a
#     PAIRWISE-COPRIME set of delay lengths across the allowed range: with no shared
#     factor, the reachable arrival times densify quickly (numerical-semigroup /
#     coin-problem argument); a shared common factor d caps every reachable time to a
#     multiple of d, forever.
#  2. decay SHAPE is governed by gains, and a single global gain can only realize one
#     exponential -- to match a two-stage (fast+slow) decay curve we need (at least)
#     two gain groups. We split the coprime delay set into a short-delay ("fast")
#     group and a long-delay ("slow") group, and solve each group's gain in closed
#     form by matching the target curve's LOCAL dB slope at early vs late checkpoints
#     to the per-round-trip attenuation model  slope(dB/sample) = 20*log10(g)/Lavg.
import sys, math
GAIN_MIN, GAIN_MAX = 1e-3, 0.99


def coprime_greedy(Lmin, Lmax, N):
    chosen = []
    c = Lmin
    while c <= Lmax and len(chosen) < N:
        if all(math.gcd(c, x) == 1 for x in chosen):
            chosen.append(c)
        c += 1
    # fallback: fill remainder with any unused values in range if not enough coprime
    if len(chosen) < N:
        used = set(chosen)
        c = Lmin
        while c <= Lmax and len(chosen) < N:
            if c not in used:
                chosen.append(c)
                used.add(c)
            c += 1
    return sorted(chosen[:N])


def main():
    tok = sys.stdin.read().split()
    p = 0
    N = int(tok[p]); p += 1
    T = int(tok[p]); p += 1
    Lmin = int(tok[p]); p += 1
    Lmax = int(tok[p]); p += 1
    K = int(tok[p]); p += 1
    ts = [int(tok[p + j]) for j in range(K)]; p += K
    target_db = [float(tok[p + j]) for j in range(K)]; p += K
    p += K  # target_density values themselves aren't needed: density is controlled
            # structurally via coprimality, not by fitting the curve numerically
    w_decay = float(tok[p]); p += 1
    w_density = float(tok[p]); p += 1

    L = coprime_greedy(Lmin, Lmax, N)
    if len(L) < N:
        # extreme fallback (should not trigger given the generous ranges)
        while len(L) < N:
            L.append(Lmax)

    n_fast = max(1, N // 2)
    n_slow = N - n_fast
    if n_slow == 0:
        n_fast -= 1
        n_slow = 1
    fast_idx = list(range(n_fast))
    slow_idx = list(range(n_fast, N))

    Lavg_fast = sum(L[i] for i in fast_idx) / len(fast_idx)
    Lavg_slow = sum(L[i] for i in slow_idx) / len(slow_idx)

    slope_early = (target_db[1] - target_db[0]) / max(1, (ts[1] - ts[0]))
    slope_late = (target_db[-1] - target_db[-2]) / max(1, (ts[-1] - ts[-2]))

    g_fast = 10.0 ** (slope_early * Lavg_fast / 20.0)
    g_slow = 10.0 ** (slope_late * Lavg_slow / 20.0)
    g_fast = min(GAIN_MAX, max(GAIN_MIN, g_fast))
    g_slow = min(GAIN_MAX, max(GAIN_MIN, g_slow))

    g = [0.0] * N
    for i in fast_idx:
        g[i] = g_fast
    for i in slow_idx:
        g[i] = g_slow

    out = []
    out.append(" ".join(str(x) for x in L))
    out.append(" ".join("%.6f" % x for x in g))
    sys.stdout.write("\n".join(out) + "\n")


main()
