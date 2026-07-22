#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0527 -- "Tidewater Berth: Admission Under Value Drift"
(family: online-admission-value-drift; format B, quality-metric).

THEME.  A single deep-water berth (a fixed pool of B mooring credits) serves a
stream of T incoming vessels over one tide window.  Vessels arrive one at a time.
Each vessel belongs to a CLASS (coastal / regional / deep-sea) and carries a
per-vessel VALUE (harbor fee) plus a SIZE (mooring credits it locks up for the
rest of the window; credits are NOT released).  The harbor master must decide
ADMIT or DECLINE the instant a vessel arrives -- irrevocably -- and may never
exceed B credits.  The catch: the fee distribution of each class DRIFTS at HIDDEN
mid-window breakpoints (a tariff regime change).  On several tides the lucrative
deep-sea class only shows up AFTER the drift, so a bar fitted to the calm early
water spends every credit on cheap coastal traffic and turns the whales away.

WHAT THE CANDIDATE CONTROLS (the innovation surface).  The candidate does NOT see
future fees (they are hidden, drawn from a drifting distribution the candidate is
never told the late half of).  Instead the candidate SHIPS A CAUSAL ADMISSION
POLICY: a 3-D bar table `bars[class][rem_bucket][sig_bucket]`.  The evaluator then
runs the tide CAUSALLY -- revealing each vessel's fee only at its own arrival --
and admits vessel t iff

        fee_t >= bars[class_t][rem_bucket][sig_bucket]   AND   size_t <= remaining

where, at arrival t (documented in statement.md, echoed in the PUBLIC instance):
  * rem_bucket = min(R-1, floor(R * remaining / B))            # how full the berth is
  * sig_bucket : where the running mean of the last L observed fees sits relative
                 to the PUBLISHED early-regime mean `prior_g` -- a causal probe of
                 whether the tariff regime has drifted UP.  Thresholds `sig_edges`
                 (multiples of prior_g) are given in the PUBLIC instance.
Because the bar may depend on rem_bucket (shadow price of a scarce credit) and on
sig_bucket (a detected drift), the policy can RESERVE credits through the cheap
early water and RELEASE them once a value drift is probed -- reserving before the
scarce high-fee class arrives.  A flat per-class bar fitted to the early stream
(the obvious move) cannot: it spends the berth on early traffic and misses the
whales.

SCORING (deterministic; no wall-time).  Per tide we compute two references on the
HIDDEN fees:
    v_open = value of the OPEN-DOOR harbor master (admit-if-it-fits, arrival order)
    v_opt  = offline FRACTIONAL optimum (fractional knapsack on hidden fee/size,
             full hindsight -- an optimistic, generally-unreachable upper bound)
and normalize the candidate's admitted value v_cand with an affine anchor
(open-door -> 0.1, offline ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (v_cand - v_open) / max(1e-9, v_opt - v_open), 0, 1 )
The open-door policy scores ~0.1; matching the (unreachable) fractional optimum
scores 1.0; doing worse than open-door scores < 0.1.  Because v_opt is a loose
hindsight bound and the late tariff is HIDDEN, even a strong reserving policy
stays strictly below 1.0 -> headroom for the RL policy above `strong`.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (classes, sizes,
early-regime prior, bucket rules -- never the fees, breakpoints, or late means).
The references are computed by THIS parent, so frame-walking learns nothing.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all tides, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

R_BUCKETS = 5           # remaining-credit buckets
S_BUCKETS = 4           # drift-signal buckets
WINDOW = 10             # running-mean window for the drift probe
SIG_EDGES = [0.90, 1.30, 2.00]   # sig_bucket edges as multiples of prior_g
K = 3                   # coastal(0), regional(1), deep-sea(2)


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_instance(spec):
    """Deterministic tide. spec = (seed,B,T,trap,shift,late_frac,dsea_share)."""
    seed, B, T, trap, shift, late_frac, dsea_share = spec
    ni = _rng(seed)
    bp = int(T * late_frac)                       # HIDDEN drift breakpoint
    # early-regime class means (PUBLISHED as prior); late-regime means (HIDDEN)
    mu0 = [10, 22, 34]
    mu1 = [11, 30, int(34 * shift)]               # deep-sea fee jumps after drift

    classes, sizes, fees = [], [], []
    for t in range(T):
        seg = 0 if t < bp else 1
        # ---- class arrival schedule ----
        if trap:
            # deep-sea (class 2) is scarce AND concentrated AFTER the breakpoint
            if seg == 0:
                c = 0 if ni(0, 99) < 68 else 1          # early: coastal + regional
            else:
                if ni(0, 99) < dsea_share:
                    c = 2
                else:
                    c = 0 if ni(0, 99) < 55 else 1
        else:
            # non-trap: deep-sea sprinkled throughout, milder drift
            roll = ni(0, 99)
            if roll < 18:
                c = 2
            elif roll < 58:
                c = 0
            else:
                c = 1
        w = ni(1, 3)                                    # mooring credits
        mu = (mu0 if seg == 0 else mu1)[c]
        noise = ni(60, 140)                             # +/-40% deterministic noise
        v = (mu * noise) // 100
        if v < 1:
            v = 1
        classes.append(c)
        sizes.append(w)
        fees.append(v)

    prior_g = (mu0[0] * 2 + mu0[1] * 2 + mu0[2]) // 5   # published early global mean
    if prior_g < 1:
        prior_g = 1
    return {"seed": seed, "B": B, "T": T, "classes": classes, "sizes": sizes,
            "fees": fees, "prior_mu": list(mu0), "prior_g": prior_g}


def _specs():
    # (seed, B, T, trap, shift, late_frac, dsea_share)
    return [
        (4527101, 40, 150, True,  3.0, 0.50, 42),   # trap
        (4527102, 46, 165, True,  3.2, 0.55, 38),   # trap
        (4527103, 38, 140, True,  2.8, 0.48, 45),   # trap
        (4527104, 52, 180, True,  3.4, 0.52, 40),   # trap
        (4527105, 44, 160, False, 1.6, 0.50, 22),   # non-trap
        (4527106, 50, 175, False, 1.8, 0.50, 22),   # non-trap
        (4527107, 42, 155, False, 1.5, 0.50, 22),   # non-trap
        (4527108, 48, 170, True,  3.1, 0.53, 40),   # trap (held-out-ish)
        (4527109, 60, 200, False, 2.0, 0.50, 22),   # non-trap larger
        (4527110, 56, 190, True,  3.3, 0.56, 36),   # trap larger
    ]


def _instances():
    return [_build_instance(s) for s in _specs()]


# ----------------------------- bucket rules --------------------------------
def _rem_bucket(remaining, B):
    b = (R_BUCKETS * remaining) // B
    if b < 0:
        b = 0
    if b > R_BUCKETS - 1:
        b = R_BUCKETS - 1
    return b


def _sig_bucket(run_mean, prior_g):
    x = run_mean / prior_g if prior_g else 0.0
    b = 0
    for e in SIG_EDGES:
        if x >= e:
            b += 1
    return b if b < S_BUCKETS else S_BUCKETS - 1


# ----------------------------- references ----------------------------------
def _open_door(inst):
    B = inst["B"]; rem = B; tot = 0
    for w, v in zip(inst["sizes"], inst["fees"]):
        if w <= rem:
            rem -= w; tot += v
    return tot


def _offline_frac(inst):
    B = inst["B"]
    items = sorted(zip(inst["fees"], inst["sizes"]),
                   key=lambda fv: fv[0] / fv[1], reverse=True)
    rem = B; tot = 0.0
    for v, w in items:
        if w <= rem:
            rem -= w; tot += v
        else:
            tot += v * (rem / w); rem = 0; break
    return tot


# ----------------------------- causal simulation ---------------------------
def _simulate(inst, bars):
    """Run the tide causally under the candidate bar table. Returns value or None
    if the table is malformed."""
    if not isinstance(bars, list) or len(bars) != K:
        return None
    for cl in bars:
        if not isinstance(cl, list) or len(cl) != R_BUCKETS:
            return None
        for row in cl:
            if not isinstance(row, list) or len(row) != S_BUCKETS:
                return None
            for x in row:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    return None
                if x != x or x in (float("inf"), float("-inf")):
                    return None
    B = inst["B"]; prior_g = inst["prior_g"]
    classes = inst["classes"]; sizes = inst["sizes"]; fees = inst["fees"]
    rem = B; tot = 0
    win = []
    run_sum = 0
    for t in range(inst["T"]):
        v = fees[t]                       # fee revealed on arrival
        win.append(v); run_sum += v
        if len(win) > WINDOW:
            run_sum -= win.pop(0)
        run_mean = run_sum / len(win)
        c = classes[t]; w = sizes[t]
        rb = _rem_bucket(rem, B)
        sb = _sig_bucket(run_mean, prior_g)
        bar = bars[c][rb][sb]
        if v >= bar and w <= rem:
            rem -= w; tot += v
    return tot


def _public(inst):
    return {"B": inst["B"], "T": inst["T"], "K": K,
            "classes": list(inst["classes"]), "sizes": list(inst["sizes"]),
            "prior_mu": list(inst["prior_mu"]), "prior_g": inst["prior_g"],
            "R_buckets": R_BUCKETS, "S_buckets": S_BUCKETS,
            "window": WINDOW, "sig_edges": list(SIG_EDGES)}


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = _instances()
    vec = []
    for inst in insts:
        v_open = _open_door(inst)
        v_opt = _offline_frac(inst)
        denom = v_opt - v_open
        if denom < 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            vec.append(0.0); continue
        bars = ans.get("bars")
        try:
            v_cand = _simulate(inst, bars)
        except Exception:
            v_cand = None
        if v_cand is None:
            vec.append(0.0); continue
        r = 0.1 + 0.9 * (v_cand - v_open) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0); continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
