# TIER: strong
# Insight: reformulate scheduling as MATCHING PURSUIT against the checker's own
# soil-plant simulator instead of a fixed split-application recipe. Passes are a
# shared, fungible resource -- any nutrient may ride for free on an already-open
# calendar day -- except that K and Mg are never allowed to open, or join, the
# same day (the antagonism-avoidance exchange rule). At each step we try every
# (day, nutrient) lump the remaining budget could support and keep whichever
# single addition raises total value the most; this naturally discovers HOW MANY
# taps each nutrient's demand curve needs (a fast-leaching or peaky curve needs
# more, smaller taps) instead of committing to a fixed pass quota up front. A
# deterministic local search then polishes lump sizes, day placement, and can
# even TRANSFER a whole pass-slot from one nutrient to another.
import sys, random

toks = sys.stdin.read().split()
p = iter(toks)
T = int(next(p)); P = int(next(p))
vN = float(next(p)); vK = float(next(p)); vMg = float(next(p))
rN = float(next(p)); rK = float(next(p)); rMg = float(next(p))
kappa = float(next(p))
BN = float(next(p)); BK = float(next(p)); BMg = float(next(p))
DN = []; DK = []; DMg = []
for _ in range(T):
    DN.append(float(next(p))); DK.append(float(next(p))); DMg.append(float(next(p)))

Dcurve = {'N': DN, 'K': DK, 'Mg': DMg}
Budget = {'N': BN, 'K': BK, 'Mg': BMg}
antagonist = {'K': 'Mg', 'Mg': 'K'}


def simulate(AN, AK, AMg):
    pN = pK = pMg = 0.0
    F = 0.0
    for t in range(T):
        pN = pN * rN + AN[t]
        pK = pK * rK + AK[t]
        pMg = pMg * rMg + AMg[t]
        upN = min(pN, DN[t])
        upK = min(pK, DK[t])
        af = kappa * pMg / max(pK, 1e-9)
        if af > 1.0:
            af = 1.0
        upMg = min(pMg, DMg[t] * af)
        pN -= upN; pK -= upK; pMg -= upMg
        F += vN * upN + vK * upK + vMg * upMg
    return F


def in_use(d, AN, AK, AMg, eps=1e-9):
    return AN[d] > eps or AK[d] > eps or AMg[d] > eps


def count_passes(AN, AK, AMg):
    return sum(1 for d in range(T) if in_use(d, AN, AK, AMg))


AN = [0.0] * T; AK = [0.0] * T; AMg = [0.0] * T
Aarr = {'N': AN, 'K': AK, 'Mg': AMg}
Sset = {'N': set(), 'K': set(), 'Mg': set()}
day_users = set()
window = max(1, int(round(T / max(P, 1))))


def raw_estimate(X, d):
    Dc = Dcurve[X]
    lo = max(0, d - window // 2)
    hi = min(T - 1, d + window // 2)
    covered = sum(Dc[lo:hi + 1])
    return max(covered, Dc[d], 1e-9)


def remaining(X):
    return max(0.0, Budget[X] - sum(Aarr[X]))


# ---------- Round A: open up to P new days by pure marginal-value forward selection ----------
cur_val = 0.0
for _step in range(P):
    best_gain, best_choice = -1.0, None
    for d in range(T):
        if d in day_users:
            continue
        for X in ('N', 'K', 'Mg'):
            rem = remaining(X)
            if rem <= 1e-9:
                continue
            if X in antagonist and d in Sset[antagonist[X]]:
                continue
            amt = min(rem, raw_estimate(X, d))
            if amt <= 1e-9:
                continue
            old = Aarr[X][d]
            Aarr[X][d] = old + amt
            f = simulate(AN, AK, AMg)
            Aarr[X][d] = old
            gain = f - cur_val
            if gain > best_gain + 1e-12:
                best_gain, best_choice = gain, (d, X, amt)
    if best_choice is None or best_gain <= 1e-9:
        break
    d, X, amt = best_choice
    Aarr[X][d] += amt
    Sset[X].add(d)
    day_users.add(d)
    cur_val = simulate(AN, AK, AMg)

# ---------- Round B: let other nutrients ride for free on already-open days ----------
for d in sorted(day_users):
    for X in ('N', 'K', 'Mg'):
        if Aarr[X][d] > 1e-9:
            continue
        rem = remaining(X)
        if rem <= 1e-9:
            continue
        if X in antagonist and d in Sset[antagonist[X]]:
            continue
        amt = min(rem, raw_estimate(X, d))
        if amt <= 1e-9:
            continue
        old = Aarr[X][d]
        Aarr[X][d] = old + amt
        f = simulate(AN, AK, AMg)
        if f >= cur_val - 1e-12:
            cur_val = f
            Sset[X].add(d)
        else:
            Aarr[X][d] = old

# ---------- Round C: spend any leftover budget as a proportional top-up ----------
for X in ('N', 'K', 'Mg'):
    rem = remaining(X)
    if rem <= 1e-9 or not Sset[X]:
        continue
    Dc = Dcurve[X]
    tot = sum(Dc[d] for d in Sset[X])
    A = Aarr[X]
    if tot <= 1e-12:
        share = rem / len(Sset[X])
        for d in Sset[X]:
            A[d] += share
    else:
        for d in Sset[X]:
            A[d] += rem * Dc[d] / tot


def enforce_pass_budget():
    guard = 0
    while count_passes(AN, AK, AMg) > P and guard < 4 * T + 10:
        guard += 1
        used = [d for d in range(T) if in_use(d, AN, AK, AMg)]
        d = min(used, key=lambda dd: AN[dd] + AK[dd] + AMg[dd])
        for X, A, Dc in (('N', AN, DN), ('K', AK, DK), ('Mg', AMg, DMg)):
            if A[d] > 1e-12:
                others = [dd for dd in range(T) if dd != d and A[dd] > 1e-12]
                if others:
                    dest = max(others, key=lambda dd: A[dd])
                else:
                    dest = max((dd for dd in range(T) if dd != d), key=lambda dd: Dc[dd])
                A[dest] += A[d]
                A[d] = 0.0
                Sset[X].discard(d)


enforce_pass_budget()

# ---------- Phase D: deterministic local search polish against the true simulator ----------
rng = random.Random(1234)
AMT_MULT = (0.5, 1.0, 1.5, 2.5, 4.0)


def best_amt_gain(X, d, rem, base_val):
    r = raw_estimate(X, d)
    A = Aarr[X]
    old = A[d]
    best_amt, best_gain = 0.0, -1.0
    for mult in AMT_MULT:
        amt = min(rem, r * mult)
        if amt <= 1e-9:
            continue
        A[d] = old + amt
        f = simulate(AN, AK, AMg)
        A[d] = old
        gain = f - base_val
        if gain > best_gain:
            best_gain, best_amt = gain, amt
    return best_amt, best_gain


def pass_transfer_move(cur_v):
    """Donate an exclusively-owned, low-value day from one nutrient to another
    nutrient's brand-new (or just-freed) day -- lets the search fix a bad
    initial pass split at runtime, directly against the true objective."""
    donors = [X for X in ('N', 'K', 'Mg') if Sset[X]]
    if not donors:
        return cur_v, False
    Xd = rng.choice(donors)
    Sd = Sset[Xd]
    excl = [d for d in Sd if not any(Aarr[Y][d] > 1e-9 for Y in ('N', 'K', 'Mg') if Y != Xd)]
    if not excl:
        return cur_v, False
    d_donor = min(excl, key=lambda d: Aarr[Xd][d])
    amt_donor = Aarr[Xd][d_donor]
    if amt_donor <= 1e-12:
        return cur_v, False
    others = [d for d in Sd if d != d_donor]
    dest = max(others, key=lambda d: Aarr[Xd][d]) if others else None

    Xr = rng.choice([X for X in ('N', 'K', 'Mg') if X != Xd])
    rem_Xr = remaining(Xr)
    if rem_Xr <= 1e-9:
        return cur_v, False

    if dest is not None:
        Aarr[Xd][dest] += amt_donor
    Aarr[Xd][d_donor] = 0.0

    cand_days = [d for d in range(T) if Aarr[Xr][d] <= 1e-9]
    if Xr in antagonist:
        other = antagonist[Xr]
        cand_days = [d for d in cand_days if Aarr[other][d] <= 1e-9]
    cand_days = sorted(cand_days, key=lambda d: -Dcurve[Xr][d])[:6]

    base_after_removed = simulate(AN, AK, AMg)
    best_gain, best_amt, best_d = -1.0, 0.0, None
    for d in cand_days:
        amt, gain = best_amt_gain(Xr, d, rem_Xr, base_after_removed)
        if gain > best_gain:
            best_gain, best_amt, best_d = gain, amt, d

    if best_d is not None and (base_after_removed + best_gain) > cur_v + 1e-9:
        Aarr[Xr][best_d] += best_amt
        if count_passes(AN, AK, AMg) <= P:
            Sset[Xr].add(best_d)
            Sset[Xd].discard(d_donor)
            return simulate(AN, AK, AMg), True
        Aarr[Xr][best_d] -= best_amt

    if dest is not None:
        Aarr[Xd][dest] -= amt_donor
    Aarr[Xd][d_donor] = amt_donor
    return cur_v, False


best = simulate(AN, AK, AMg)
iters = min(60 * max(T, 1), 3500)
for _it in range(iters):
    roll = rng.random()
    if roll < 0.20:
        best, _ = pass_transfer_move(best)
        continue
    X = rng.choice(['N', 'K', 'Mg'])
    A = Aarr[X]; S = Sset[X]
    if roll < 0.45:
        if not S:
            continue
        d_old = rng.choice(list(S))
        if A[d_old] <= 1e-12:
            continue
        cand = [d for d in range(T) if A[d] <= 1e-12 and d != d_old]
        if not cand:
            continue
        d_new = rng.choice(cand)
        if X in antagonist:
            other = Aarr[antagonist[X]]
            if other[d_new] > 1e-9:
                continue
        amt = A[d_old]
        A[d_old] = 0.0
        A[d_new] += amt
        if count_passes(AN, AK, AMg) <= P:
            cur = simulate(AN, AK, AMg)
            if cur >= best - 1e-12:
                best = cur
                S.discard(d_old); S.add(d_new)
                continue
        A[d_new] = 0.0
        A[d_old] = amt
    else:
        if len(S) < 2:
            continue
        d1, d2 = rng.sample(list(S), 2)
        old1, old2 = A[d1], A[d2]
        best_delta, best_local = 0.0, best
        for frac in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0):
            delta = old1 * frac
            if delta <= 1e-9:
                continue
            A[d1] = old1 - delta; A[d2] = old2 + delta
            cur = simulate(AN, AK, AMg)
            if cur > best_local + 1e-12:
                best_local, best_delta = cur, delta
        A[d1] = old1 - best_delta
        A[d2] = old2 + best_delta
        best = best_local

out = []
for t in range(T):
    out.append("%.6f %.6f %.6f" % (AN[t], AK[t], AMg[t]))
sys.stdout.write("\n".join(out) + "\n")
