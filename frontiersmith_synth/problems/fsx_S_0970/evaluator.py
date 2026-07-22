#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0970 -- "Prony Query Prospector: Mapping a Buried
Sparse Seam from a Few Boreholes".

Family: prony-query-prospector (Ben-Or/Tiwari sparse polynomial interpolation +
Prony's method), skinned as a mineral-survey black box. A geological survey firm
models a buried ore seam as a SPARSE polynomial over a finite field:

    f(x) = sum_{i=0}^{t-1} c_i * x^{e_i}   (mod p)

with exactly t nonzero terms out of an ambient degree range [0, p-2] -- "the seam
is thin, but it could be buried anywhere across a huge depth range". The survey
crew already drilled Q boreholes at STANDARD, GEOMETRICALLY-SPACED sampling
depths x_k = g^k mod p for k = 0..Q-1 (g is the field's public survey base, a
primitive root of F_p), and logged the readings s_k = f(g^k) mod p. Q is small
-- far below the ambient degree p-1, and (on the trap instances) even below the
2t samples that a structure-UNAWARE sparse-recovery routine (plain Ben-Or-Tiwari /
Prony: fit an order-t linear recurrence from 2t consecutive samples, its roots
g^{e_i}) would need.

THE NOVELTY the candidate must exploit: the survey company's geologists already
know the ore veins run in an EVENLY-SPACED SEAM PATTERN -- the t exponents form an
arithmetic progression e_i = a + i*d for SECRET a (start depth) and d (spacing),
1 <= d <= d_max (d_max public). Writing z_i = g^{e_i} = g^a * (g^d)^i, the t
unknown roots collapse onto a 2-PARAMETER family (a, d) instead of t independent
unknowns. So instead of solving the *generic* order-t recurrence (which provably
needs 2t samples), the candidate can (1) hypothesize (a, d) -- a small 2-D search,
since d ranges over d_max values and a ranges over the (still modest) field size --
(2) for each hypothesis, solve a t x t linear system for the t coefficients c_i
using only t of the Q samples, and (3) use the FEW leftover samples (Q - t of
them) purely to VERIFY/reject the hypothesis. This halves the query requirement
from ~2t down to ~t + O(1): the substitution x -> g^x is what turns "t unknown
exponents" into "2 unknown scalars", and it is the a-priori AP structure -- not
more queries -- that buys back the missing samples.

TRAP: on tight instances Q < 2t, plain structure-unaware Prony (2t-sample order-t
recurrence) is information-theoretically UNDERDETERMINED; a "recognize this is
Ben-Or-Tiwari, just do the textbook thing" solver that uses the KNOWN sparsity t
as recurrence order but only has Q < 2t samples will silently corrupt its own
linear system (e.g. zero-padding the missing future samples) and recover the
WRONG term list -- while the AP-aware solver, using the SAME Q samples, recovers
the seam exactly.

Answer format: {"terms": [[e_0, c_0], [e_1, c_1], ...]}  (<= t pairs, distinct
integer exponents in [0, p-2], integer coefficients mod p).

Scoring (deterministic; no wall-time): reconstruct fhat(x) = sum listed c*x^e over
F_p and count how many of the Q GIVEN sample points it reproduces EXACTLY
(fit_count in [0, Q]). This needs no secret ground truth at scoring time -- it is
purely a function of the candidate's answer and the public samples -- but every
instance is verified at GENERATION time to have a UNIQUE <=t-sparse explanation
(the planted one), so matching all Q samples with <= t terms means exact recovery.
The evaluator's own baseline is the "guess a single constant term equal to the
first reading" construction, which by construction matches exactly 1 of the Q
samples (verified per-instance at generation time -- no accidental extra
collisions). A submission with more than t terms, duplicate exponents,
out-of-range exponents, or non-finite/non-integral values is REJECTED (score 0).

  r = min(1, 0.1 * fit_count / baseline_fit_count)

so the trivial single-constant-term guess scores exactly 0.1, and full exact
recovery of a Q-query instance scores 0.1*Q (headroom-calibrated so this never
saturates 1.0 -- see make_instances()).

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ============================= modular arithmetic ===========================
def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_factors(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d); n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def _primitive_root(p):
    """Smallest primitive root of F_p^* (p prime)."""
    if p == 2:
        return 1
    phi = p - 1
    fs = _prime_factors(phi)
    for cand in range(2, p):
        if all(pow(cand, phi // f, p) != 1 for f in fs):
            return cand
    raise ValueError("no primitive root found")


def _gauss_solve(A, b, p, n):
    """Solve A*x = b mod prime p (n x n). Returns list x, or None if singular."""
    A = [row[:] for row in A]
    b = b[:]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if A[r][col] % p != 0:
                piv = r; break
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = pow(A[col][col], p - 2, p)
        A[col] = [(x * inv) % p for x in A[col]]
        b[col] = (b[col] * inv) % p
        for r in range(n):
            if r != col and A[r][col] % p != 0:
                factor = A[r][col] % p
                A[r] = [(A[r][j] - factor * A[col][j]) % p for j in range(n)]
                b[r] = (b[r] - factor * b[col]) % p
    return [x % p for x in b]


# ---------------------- reference AP-aware search (generation self-check) ----
def _ap_search(p, g, t, d_max, Q, s):
    """Return list of (a, d) hypotheses that EXACTLY reproduce all Q samples
    using <= t terms (the reference "strong" algorithm). Used only at
    generation time to verify each instance has a UNIQUE planted explanation --
    never exposed to candidates."""
    hits = []
    for d in range(1, d_max + 1):
        maxa = p - 2 - (t - 1) * d
        if maxa < 0:
            continue
        gd = pow(g, d, p)
        gd_pows = [pow(gd, i, p) for i in range(t)]
        for a in range(0, maxa + 1):
            ga = pow(g, a, p)
            zs = [(ga * gd_pows[i]) % p for i in range(t)]
            M = [[pow(zs[i], k, p) for i in range(t)] for k in range(t)]
            rhs = [s[k] % p for k in range(t)]
            sol = _gauss_solve(M, rhs, p, t)
            if sol is None:
                continue
            ok = True
            for k in range(t, Q):
                pred = sum(sol[i] * pow(zs[i], k, p) for i in range(t)) % p
                if pred != s[k] % p:
                    ok = False; break
            if ok:
                hits.append((a, d, tuple(sol)))
    return hits


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
# (seed, t, Q, d_max, p) -- p chosen prime; Q < 2t on TIGHT/trap instances
# (structure-unaware order-t Prony is info-theoretically underdetermined there),
# Q >= 2t on LOOSE instances (generic Prony also succeeds there -- no free lunch
# claimed when the budget is generous). idx 9-10 are larger-t held-out cases.
_SPECS = [
    (9701, 3, 8, 10, 401),   # loose  (2t=6 <= 8)
    (9702, 3, 9, 10, 419),   # loose  (2t=6 <= 9)
    (9703, 4, 9, 10, 433),   # loose  (2t=8 <= 9)
    (9704, 3, 5, 10, 271),   # tight  (2t=6 > 5)
    (9705, 4, 6, 10, 283),   # tight  (2t=8 > 6)
    (9706, 4, 7, 10, 307),   # tight  (2t=8 > 7)
    (9707, 5, 7, 10, 311),   # tight  (2t=10 > 7)
    (9708, 5, 8, 10, 499),   # tight  (2t=10 > 8)
    (9709, 5, 9, 10, 521),   # tight, held-out (2t=10 > 9)
    (9710, 6, 9, 10, 601),   # tight, held-out, larger t (2t=12 > 9)
]


def make_instances():
    out = []
    for seed, t, Q, d_max, p in _SPECS:
        assert _is_prime(p), p
        g = _primitive_root(p)
        r = _rng(seed)
        d = r(1, d_max)
        maxa = p - 2 - (t - 1) * d
        assert maxa >= 0
        a = r(0, maxa)
        coeffs = [r(1, p - 1) for _ in range(t)]
        exps = [a + i * d for i in range(t)]
        s = []
        for k in range(Q):
            xk = pow(g, k, p)
            val = 0
            for i in range(t):
                val = (val + coeffs[i] * pow(xk, exps[i], p)) % p
            s.append(val)

        # generation-time self-check: the planted (a,d,coeffs) must be the
        # UNIQUE <=t-sparse AP explanation of these Q samples.
        hits = _ap_search(p, g, t, d_max, Q, s)
        assert len(hits) == 1, (seed, "ambiguous or unsolved instance", len(hits))
        ha, hd, hc = hits[0]
        assert ha == a and hd == d and list(hc) == coeffs, (seed, "self-check mismatch")

        # baseline self-check: "guess constant = s[0]" must match EXACTLY 1 sample
        fit0 = sum(1 for k in range(Q) if s[0] == s[k])
        assert fit0 == 1, (seed, "baseline collision", fit0)

        public = {"p": p, "g": g, "t": t, "d_max": d_max, "Q": Q, "s": s}
        hidden = {"a": a, "d": d, "coeffs": coeffs, "exps": exps}
        out.append({"public": public, "hidden": hidden})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    """fit_count of the 'guess a single constant term = s[0]' construction."""
    pub = inst["public"]
    s = pub["s"]; Q = pub["Q"]
    return float(sum(1 for k in range(Q) if s[0] == s[k]))


def score(inst, answer):
    """Strictly validate the answer; return (ok, fit_count)."""
    pub = inst["public"]
    p = pub["p"]; g = pub["g"]; t = pub["t"]; Q = pub["Q"]; s = pub["s"]
    if not isinstance(answer, dict):
        return False, None
    terms = answer.get("terms", None)
    if not isinstance(terms, list) or len(terms) > t:
        return False, None
    parsed = []
    seen_exp = set()
    for pair in terms:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            return False, None
        e_raw, c_raw = pair
        for v in (e_raw, c_raw):
            if isinstance(v, bool):
                return False, None
            if isinstance(v, float) and not math.isfinite(v):
                return False, None
        try:
            e = int(e_raw)
            if isinstance(e_raw, float) and abs(e_raw - e) > 1e-9:
                return False, None
            c = int(c_raw)
            if isinstance(c_raw, float) and abs(c_raw - c) > 1e-9:
                return False, None
        except (TypeError, ValueError):
            return False, None
        if e < 0 or e > p - 2:
            return False, None
        if e in seen_exp:
            return False, None
        seen_exp.add(e)
        parsed.append((e, c % p))

    # reconstruct fhat at the Q query depths and count exact matches
    fit = 0
    for k in range(Q):
        xk = pow(g, k, p)
        val = 0
        for e, c in parsed:
            val = (val + c * pow(xk, e, p)) % p
        if val == s[k] % p:
            fit += 1
    return True, float(fit)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=5)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
