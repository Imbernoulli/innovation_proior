#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0808 -- "Field-Camera Trap Network: Adaptive Reservoir
Budgeting Across Unknown, Skewed Species Strata".

Family: stratified-reservoir-variance. A wildlife-monitoring network streams photo
detections from camera traps; each detection is tagged with a coarse species-class
("stratum") and a scene brightness/exposure reading ("value") that downstream
analysis must summarize as a per-class-weighted population mean. Only a fixed total
number of full-resolution frames -- the RESERVOIR budget K -- can be archived (the
rest are discarded), so the archiving policy must decide, per species stratum, how
many of the K reservoir slots to devote to it. The true relative frequency w_s of
each species and the true within-class variance sigma_s^2 of its readings are BOTH
unknown in advance and SKEWED across classes: most classes are common and mild, but
occasionally one class is RARE and highly erratic (e.g. a skittish nocturnal animal
that is seen only a handful of times yet produces wildly different exposures). The
candidate observes a BURN-IN prefix of the stream (in arrival order, tagged by
stratum) and must commit, from that alone, to a per-stratum reservoir allocation for
the remainder of the deployment.

THREE COMPOSED MECHANISMS (no single one is "the" intended solution):
  1. strata-size-estimation:   estimate each class's population share w_hat_s from
     the (possibly zero-count, noisy) burn-in observations.
  2. adaptive-per-stratum-reservoir: the reservoir is NOT split into equal shares;
     the candidate reports one integer n_s per stratum, sum n_s <= K.
  3. variance-balancing-reallocation: the estimator's asymptotic variance is
        Var(alloc) = sum_s  w_s^2 * sigma_s^2 / max(n_s, 0.5)
     (the classical stratified-mean variance, using TRUE population w_s/sigma_s --
     hidden from the candidate, known only to this evaluator) and MUST be minimized.
     By Cauchy-Schwarz this is minimized (Neyman allocation) at n_s ~ w_s*sigma_s,
     NOT n_s ~ w_s alone and NOT n_s ~ K/S -- but the candidate only ever sees NOISY,
     often near-zero-count burn-in evidence about sigma_s, so plugging a raw sample
     variance in naively is itself a trap: a rare class seen 0-2 times in burn-in
     looks falsely "safe" to a naive estimator, while its TRUE sigma_s (used for
     scoring) can be enormous. A genuine insight must hedge allocation UP for
     under-observed strata rather than trust a small/absent sample estimate,
     compose that hedge with a frequency estimate that also survives zero counts,
     and only then apply variance-balancing across strata.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance (burn-in + K + S) on stdin, and returns
ONLY its allocation on stdout -- it can never see the hidden w_s/sigma_s or read
this evaluator's frames.

Scoring (deterministic; no wall-time):
  baseline b = Var(alloc) of the EQUAL-SHARE allocation n_s = K/S for every s.
  For a FEASIBLE answer (alloc of length S, all entries finite, >= 0, sum <= K+eps):
      obj = sum_s w_s^2 * sigma_s^2 / max(n_s, 0.5)          (TRUE hidden w_s,sigma_s)
      r   = min(1, 0.1 * b / obj)
  -> equal-share allocation maps to exactly 0.1; an allocation that drives the
     stratified variance k times below equal-share maps to min(1, 0.1*k). Malformed
     / infeasible answers score 0 on that instance.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def uf():  # uniform float in [0,1)
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    nxt.uf = uf
    return nxt


def _gauss(r):
    """Standard normal via Box-Muller, driven by the deterministic LCG above."""
    u1 = max(r.uf(), 1e-12)
    u2 = r.uf()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ----------------------------- instance family -----------------------------
def _make_strata(r, S, trap_idx, trap_w, trap_sigma_mul, base_sigma_lo, base_sigma_hi):
    """Build true (w, mu, sigma) per stratum. If trap_idx is not None, that stratum
    gets a forced (moderate, not vanishing) weight and a heavily inflated sigma; the
    rest split the remaining weight with a strongly skewed (power-law) random split
    and mild-to-moderate sigma variation."""
    raw = [(0.1 + r.uf() * 0.9) ** 3 for _ in range(S)]  # power-law skewed weight shares
    if trap_idx is not None:
        raw[trap_idx] = 0.0                              # will be forced below
    tot = sum(raw)
    remaining = 1.0 - (trap_w if trap_idx is not None else 0.0)
    w = [remaining * (x / tot) for x in raw]
    if trap_idx is not None:
        w[trap_idx] = trap_w
    mu = [round(r.uf() * 20.0 - 10.0, 4) for _ in range(S)]
    sigma = [round(base_sigma_lo + r.uf() * (base_sigma_hi - base_sigma_lo), 4) for _ in range(S)]
    if trap_idx is not None:
        sigma[trap_idx] = round(sigma[trap_idx] * trap_sigma_mul, 4)
    return w, mu, sigma


def _draw_burnin(r, w, mu, sigma, B):
    cum = []
    acc = 0.0
    for x in w:
        acc += x
        cum.append(acc)
    out = []
    for _ in range(B):
        u = r.uf()
        s = 0
        while s < len(cum) - 1 and u > cum[s]:
            s += 1
        v = mu[s] + sigma[s] * _gauss(r)
        out.append({"s": s, "v": round(v, 5)})
    return out


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{...}}, ...].

    10 instances: 7 "normal" (skewed strata sizes, MILD sigma variation across
    strata -- freq-proportional allocation is a decent but not optimal heuristic)
    and 3 TRAP instances (one stratum is rare -- 1..4% of the population -- and its
    true sigma is 8..16x the others; burn-in typically shows it 0-3 times, so any
    allocation policy that trusts burn-in frequency/variance at face value starves
    it, while true variance-balancing must still give it a large reservoir share).
    """
    specs = [
        # seed, S, K, B, trap_idx, trap_w, trap_sigma_mul, sigma_lo, sigma_hi
        (311, 5, 80, 220, None, None, None, 1.0, 2.0),
        (312, 6, 90, 260, None, None, None, 1.0, 2.0),
        (313, 4, 70, 180, None, None, None, 1.0, 2.0),
        (314, 6, 100, 300, None, None, None, 1.0, 2.0),
        (315, 7, 110, 320, None, None, None, 1.0, 2.0),
        (316, 5, 85, 240, None, None, None, 1.0, 2.0),
        (317, 6, 95, 280, None, None, None, 1.0, 2.0),
        # trap instances: one moderately-weighted (not vanishing) but wildly
        # erratic class; a frequency-only allocation still tracks its share
        # reasonably but ignores that it needs FAR more slots to tame its variance
        (321, 6, 90, 260, 4, 0.10, 20.0, 1.0, 2.0),
        (322, 7, 120, 340, 2, 0.09, 22.0, 1.0, 2.0),
        # larger / held-out trap instance for generalization
        (323, 8, 140, 400, 6, 0.10, 18.0, 1.0, 2.0),
    ]
    out = []
    for seed, S, K, B, trap_idx, trap_w, trap_mul, slo, shi in specs:
        r = _rng(seed)
        w, mu, sigma = _make_strata(r, S, trap_idx, trap_w, trap_mul, slo, shi)
        burnin = _draw_burnin(r, w, mu, sigma, B)
        public = {"S": S, "K": K, "burnin": burnin}
        hidden = {"w": w, "sigma": sigma, "mu": mu}
        out.append({"public": public, "hidden": hidden})
    return out


# ----------------------------- scoring -------------------------------------
def _variance(alloc, w, sigma):
    return sum(w[s] * w[s] * sigma[s] * sigma[s] / max(alloc[s], 0.5) for s in range(len(w)))


def baseline(inst):
    """Equal-share allocation n_s = K/S for every stratum. Always feasible."""
    pub = inst["public"]; hid = inst["hidden"]
    S = pub["S"]; K = pub["K"]
    alloc = [K / S] * S
    return _variance(alloc, hid["w"], hid["sigma"])


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]; hid = inst["hidden"]
    S = pub["S"]; K = pub["K"]
    if not isinstance(answer, dict):
        return False, None
    alloc = answer.get("alloc", None)
    if not isinstance(alloc, list) or len(alloc) != S:
        return False, None
    try:
        alloc = [float(x) for x in alloc]
    except (TypeError, ValueError):
        return False, None
    for x in alloc:
        if not math.isfinite(x):
            return False, None
        if x < -1e-9:
            return False, None
    alloc = [max(0.0, x) for x in alloc]
    if sum(alloc) > K + 1e-6:
        return False, None
    obj = _variance(alloc, hid["w"], hid["sigma"])
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
