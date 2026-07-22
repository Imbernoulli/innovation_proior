#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0572 -- "The Aliased Assay: Which Sample to Test"
(family: sequential-probe-disambiguation; format B, quality-metric).

THEME.  A materials lab must identify an unknown sample.  A finite set of K
candidate identities (hypotheses) is consistent with what is already known; each
identity h carries a known prior belief w[h] and implies a scalar property
theta[h] (the number the lab ultimately has to report -- e.g. a yield strength).
The lab owns a catalogue of M assays ("probes").  Running assay j costs cost[j]
lab-hours and, because every instrument has a finite resolution (its NOISE
level), returns only a COARSE reading read[j][h]: two candidate identities are
told apart by assay j iff they read differently, read[j][h] != read[j][h'].
A cheap coarse assay lumps many near-identities into the same reading; a precise
assay costs more but splits a subtle alias.  The lab has a fixed budget of
lab-hours and pays a running penalty gamma per hour spent.

WHAT YOU CHOOSE (non-adaptive design).  Pick a SET S of assays to run
(sum of costs <= budget).  After running S you observe, for the true identity,
the reading vector (read[j][*])_{j in S}; this narrows the candidates to the
CONFUSION CLASS -- every identity that produced the identical reading vector.
Your Bayes estimate of theta is the prior-weighted mean of theta over that class.

OBJECTIVE (minimise).  Averaged over the prior on the true identity, the
posterior estimation error is the within-class weighted variance of theta:
    residual(S) = sum over confusion classes C of
                  sum_{h in C} w[h] * (theta[h] - mean_C)^2 ,
                  mean_C = (sum_{h in C} w[h]*theta[h]) / (sum_{h in C} w[h]).
The lab minimises
    J(S) = residual(S) + gamma * (sum_{j in S} cost[j])
subject to the hard budget sum_{j in S} cost[j] <= budget.  Running nothing is
feasible and gives J = residual(empty) = the full prior variance of theta.

THE TRAP (why the obvious recipe is not the answer).  The textbook adaptive-
experiment recipe buys assays by marginal INFORMATION-GAIN per cost (entropy of
the posterior over identities).  Information gain depends only on the prior
weights of what an assay separates -- NOT on how far apart their theta values
are.  Some cheap assays split high-prior but theta-clustered identities (lots of
information, almost no error reduction); a few costly assays split a low-prior
but theta-EXTREME aliased pair (little information, huge error reduction).  Under
the budget, information-per-cost greedily buys the cheap high-information assays
and the wrong premium assays, leaving the alias that actually dominates the
estimation error unbroken.  The insight the strong solver exploits: value an
assay by the theta-variance it removes from the SURVIVING confusion classes,
not by its marginal information under the current posterior.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "K": int, "prior": [w..], "theta": [t..],
             "M": int, "cost": [c..], "read": [[..K ints..] .. M rows],
             "budget": int, "gamma": float}
  stdout: ONE JSON object:
            {"probes": [j0, j1, ...]}   # indices of assays to run (a SET)

  Valid iff "probes" is a list of distinct integers in [0,M) whose total cost is
  <= budget.  Anything else (bad type, out-of-range, duplicate, over budget,
  crash, timeout, non-JSON) scores 0.0 on that instance.  The empty list is valid.

SCORING (deterministic; no wall-time).  Per instance with baseline b = J(empty):
    r = min(1.0, 0.1 * b / max(J(S), 1e-12))
Running nothing reproduces the baseline -> r = 0.1.  Fully resolving every alias
would cost so many lab-hours that gamma*cost keeps J well above 0.1*b, so even
excellent designs stay below 1.0 -> headroom is guaranteed.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS
via isorun.run_candidate; it only ever sees the PUBLIC instance.  The baseline
and the confusion/variance computation run in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_instance(seed, n_bulk, pairs, L, c_prem_list, k_afford, base_b, bulk_spread):
    """Construct one deterministic instance.

    Hypotheses = n_bulk "bulk" identities (theta clustered near 0, cheap assays
    shatter them) + one aliased PAIR per entry in `pairs`.  Each pair entry is
    (omega, R): both members carry prior omega and theta = +R and -R (theta-
    EXTREME, high variance) but read IDENTICALLY on every cheap assay -- only a
    dedicated PREMIUM assay separates them.

    `pairs` is engineered so prior-weight order (what information-greedy follows)
    disagrees with theta-variance order (what actually cuts the error): the
    high-variance pairs are given LOWER prior weight.  c_prem_list gives each
    premium assay's cost; the budget only affords ~k_afford of them, forcing a
    choice of WHICH aliases to break.
    """
    nxt = _rng(seed)
    n_pairs = len(pairs)
    K = n_bulk + 2 * n_pairs

    theta = [0.0] * K
    prior = [0.0] * K

    # ---- bulk identities: clustered theta, spread deterministically ----
    for t in range(n_bulk):
        theta[t] = float(((t * 7 + nxt(0, 3)) % (2 * bulk_spread + 1)) - bulk_spread)
    for t in range(n_bulk):
        prior[t] = float(3 + nxt(0, 2))

    # ---- aliased pairs: extreme theta, weight anti-correlated with variance ----
    pair_idx = []                      # (A, B) hypothesis indices per pair
    for i, (omega, R) in enumerate(pairs):
        A = n_bulk + 2 * i
        B = n_bulk + 2 * i + 1
        theta[A] = float(R)
        theta[B] = float(-R)
        prior[A] = float(omega)
        prior[B] = float(omega)
        pair_idx.append((A, B))

    # normalise prior to sum 1
    tot = sum(prior)
    prior = [p / tot for p in prior]

    # ---- assays ----
    cost = []
    read = []
    # cheap coarse assays: assign every hypothesis a base-`base_b` code of length L.
    # bulk identities get distinct codes (shattered by the full cheap set); the two
    # members of a pair SHARE their pair's code (a cheap assay can never split them);
    # distinct pairs get distinct codes (separated from bulk and from each other).
    codes = []
    for t in range(n_bulk):
        codes.append(t)                     # unique code per bulk id
    for i in range(n_pairs):
        c = n_bulk + i                      # unique shared code per pair
        codes.append(c)                     # A
        codes.append(c)                     # B  (same code -> aliased on cheap)
    # emit L cheap assays; assay j reveals digit j of the base-b code
    for j in range(L):
        row = [(codes[h] // (base_b ** j)) % base_b for h in range(K)]
        read.append(row)
        cost.append(1)
    # premium assays: one per pair, separates ONLY that pair (member B reads 1)
    for i in range(n_pairs):
        A, B = pair_idx[i]
        row = [0] * K
        row[B] = 1
        read.append(row)
        cost.append(int(c_prem_list[i]))

    M = len(cost)
    # budget: all cheap assays + about k_afford of the cheapest premium assays
    prem_sorted = sorted(c_prem_list)
    budget = L + sum(prem_sorted[:k_afford])
    gamma = 0.30

    return {"name": f"assay{seed}", "K": K, "prior": prior, "theta": theta,
            "M": M, "cost": cost, "read": read, "budget": int(budget),
            "gamma": gamma}


def _build_instances():
    """Deterministic instance family.  Each spec:
       (seed, n_bulk, pairs=[(omega,R)..], L, c_prem_list, k_afford, base_b, bulk_spread)
    Pairs place the HIGH-variance (large R) alias at LOW prior weight, so
    information-per-cost greedy (weight-ordered) breaks the wrong aliases.
    Instances 0..7 are trap-structured; 8..10 are larger held-out variants.
    """
    specs = [
        # seed  nb  pairs                                          L  cprem        kaf b spread
        (5101,  8, [(0.05, 30), (0.12, 8), (0.10, 10)],            6, [5, 5, 5],    1, 4, 4),
        (5102,  9, [(0.04, 34), (0.13, 7), (0.09, 12)],            6, [5, 5, 5],    1, 4, 5),
        (5103, 10, [(0.05, 28), (0.11, 9), (0.08, 14), (0.10, 11)],7, [5,5,5,5],    2, 4, 4),
        (5104,  8, [(0.045, 32), (0.14, 6), (0.10, 9)],            6, [6, 4, 4],    1, 4, 5),
        (5105, 11, [(0.05, 26), (0.12, 8), (0.09, 13), (0.11, 7)], 7, [5,5,5,5],    2, 4, 4),
        (5106,  9, [(0.04, 36), (0.13, 6), (0.08, 15)],            6, [6, 5, 5],    1, 4, 6),
        (5107, 10, [(0.05, 30), (0.12, 9), (0.10, 12), (0.09, 8)], 7, [5,5,5,5],    2, 4, 5),
        (5108,  8, [(0.05, 24), (0.11, 10), (0.09, 11)],           6, [5, 5, 5],    1, 4, 4),
        # held-out: larger, more pairs / more assays
        (5211, 12, [(0.035, 38), (0.11, 7), (0.08, 14), (0.10, 10), (0.09, 12)],
                                                                   8, [6,6,6,6,6],  2, 4, 5),
        (5212, 13, [(0.04, 34), (0.12, 6), (0.075, 16), (0.09, 11), (0.10, 9)],
                                                                   8, [6,5,6,5,6],  2, 4, 6),
        (5213, 11, [(0.045, 40), (0.10, 8), (0.085, 13), (0.095, 10)],
                                                                   8, [7,6,6,6],    2, 4, 5),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- objective -----------------------------------
def _residual(inst, probe_set):
    """Within-confusion-class prior-weighted variance of theta under assays S."""
    K = inst["K"]
    prior = inst["prior"]
    theta = inst["theta"]
    read = inst["read"]
    groups = {}
    for h in range(K):
        key = tuple(read[j][h] for j in probe_set)
        groups.setdefault(key, []).append(h)
    total = 0.0
    for members in groups.values():
        W = sum(prior[h] for h in members)
        if W <= 0:
            continue
        mean = sum(prior[h] * theta[h] for h in members) / W
        for h in members:
            total += prior[h] * (theta[h] - mean) ** 2
    return total


def _objective(inst, probe_set):
    res = _residual(inst, probe_set)
    c = sum(inst["cost"][j] for j in probe_set)
    return res + inst["gamma"] * c


def _baseline(inst):
    return _objective(inst, [])     # empty design: full prior variance, no cost


# ----------------------------- validation / scoring ------------------------
def _validate_and_J(inst, answer):
    """Return J(S) if the answer is a feasible design, else None."""
    if not isinstance(answer, dict):
        return None
    probes = answer.get("probes")
    if not isinstance(probes, list):
        return None
    M = inst["M"]
    seen = set()
    for j in probes:
        if isinstance(j, bool) or not isinstance(j, int):
            return None
        if j < 0 or j >= M:
            return None
        if j in seen:
            return None
        seen.add(j)
    cost = sum(inst["cost"][j] for j in seen)
    if cost > inst["budget"]:
        return None
    return _objective(inst, sorted(seen))


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = _baseline(inst)
        public = {"name": inst["name"], "K": inst["K"],
                  "prior": list(inst["prior"]), "theta": list(inst["theta"]),
                  "M": inst["M"], "cost": list(inst["cost"]),
                  "read": [list(row) for row in inst["read"]],
                  "budget": inst["budget"], "gamma": inst["gamma"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            J = _validate_and_J(inst, ans)
        except Exception:
            J = None
        if J is None:
            vec.append(0.0)
            continue
        r = 0.1 * b / max(J, 1e-12)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
