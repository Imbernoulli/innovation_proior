#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1312 -- "Asking the Question That Tells You the Most:
Exposure-Aware Adaptive Item Selection" (family: exam-item-select-policy;
eval_form: quality-metric).

A computerized-adaptive-testing (CAT) program shares ONE item bank (each item has
an IRT discrimination `a` and difficulty `b`) across a whole EXAMINATION SEASON --
a sequence of N examinees, processed in a fixed order, each taking a test of L
items drawn adaptively from the M-item bank. At every step the running ability
estimate theta_hat is updated by this (frozen) evaluator via a standard
Bayesian EAP update over a fixed quadrature grid -- this is the
ability-estimate-convergence mechanism and is NOT something the candidate
controls. The candidate's job is only to choose, at each step, WHICH item to
administer next; it does this by submitting a compact PRIORITIZATION POLICY
(four numbers), which this evaluator then applies, step by step, using its own
live-tracked state (current ability estimate + current per-item exposure count).

The trap: an item that is administered to more than `cap_frac_i` of the N
examinees becomes "compromised" (its answer circulates / gets memorized) for
every administration AFTER the cap is crossed -- from that point on, the
observed response for that item is generated from a constant high pass-rate
(LEAK_PASS_RATE) that is INDEPENDENT of the examinee's true ability, instead of
from the item's real 2PL model. The EAP update, unaware of the leak, still
scores a "correct" answer on a hard item as strong evidence of high ability --
so once a pool is compromised, ability ESTIMATES for every later examinee who
draws that item become systematically biased, not just less precise. A policy
that always administers the maximum-information item at the current estimate
(the standard, obvious CAT recipe) ignores this entirely and burns through the
handful of locally-best items long before the season ends whenever true ability
is concentrated (not uniformly spread like the policy-agnostic forecast
assumes) -- collapsing estimate accuracy for the back half of the season. The
insight this problem rewards is trading a little per-decision information for
long-run POOL SURVIVAL: throttling popular items pre-emptively (using the
supplied population-agnostic demand forecast to anticipate scarcity before it
is observed) and, crucially, trusting the LIVE observed exposure counts over
that forecast once real data is available, since the forecast can systematically
under-predict demand when the true ability distribution is not the generic one
it assumes.

The candidate is run as an ISOLATED subprocess (isorun): it reads ONE JSON
"public instance" from stdin (item bank, exposure caps, a population-agnostic
demand forecast) and writes ONE JSON "policy" to stdout. It never sees the
hidden true abilities, hidden hint-population internals, or this evaluator's
state. Given the policy, THIS evaluator (not the candidate) runs the full,
deterministic, item-by-item season simulation.

Public instance JSON (what the candidate reads on stdin):
    {
      "M": int, "L": int, "N": int,             # bank size, test length, season size
      "items": [{"a": float, "b": float, "cap_frac": float}, ...] * M,
      "typical_demand_hint": [float] * M,        # per item, admin-rate under a GENERIC
                                                  # N(0,1)-ability reference population running
                                                  # pure max-information selection (no exposure
                                                  # control) -- may UNDER-predict real demand
      "leak_pass_rate": float,                   # P(correct) once an item is compromised
      "prior_mean": float, "prior_sd": float,    # EAP prior for ability estimation
      "selection_anchor_jitter_sd": float,       # each examinee's item-CHOICE anchor is the
                                                  # running estimate plus a fixed idiosyncratic
                                                  # N(0, this_sd) offset (does NOT bias the
                                                  # estimate itself, only which item looks best)
      "seed": int
    }

Answer JSON (what the candidate writes on stdout):
    {"info_weight": float, "exposure_weight": float,
     "exposure_shape": float, "hint_trust": float}
    Clipped to info_weight,exposure_weight in [0,50], exposure_shape in [0,6],
    hint_trust in [0,1]. Non-finite / wrong type / missing key -> instance score 0.

Per-step selection (evaluator-applied, using ITS OWN live state):
    for each not-yet-used-by-this-examinee item i:
      p          = 1/(1+exp(-a_i*(theta_hat-b_i)))     # at the CURRENT ability estimate
      info_i     = a_i^2 * p*(1-p)                      # Fisher information
      usage_i    = admins_so_far_i / cap_count_i          # live, this policy's own run
      hint_i     = typical_demand_hint[i] * N / cap_count_i   # forecast, same units
      pred_i     = hint_trust*hint_i + (1-hint_trust)*usage_i
      score_i    = info_weight*info_i - exposure_weight*max(pred_i,0)**exposure_shape
    administer argmax_i score_i (ties -> smallest index)

Per-instance objective: obj = -mean_n (theta_hat_final_n - theta_true_n)^2 over
the N examinees, i.e. negative mean-squared ability-estimation error. Score is
an affine anchor between two references this evaluator computes ITSELF: a weak
non-adaptive baseline (info_weight=0, exposure_weight=0 -> fixed round-robin
item order; obj_base -> r=0.1) and a loose, unreachable upper bound (pure
max-information selection with an INFINITE exposure cap, i.e. pool scarcity
assumed away; obj_upper -> r -> up to ~1.0):

    r = clamp(0.1 + 0.9*(obj_cand - obj_base) / max(obj_upper - obj_base, 1e-6), 0, 1)

floored at 0.01 for a valid answer. `Ratio` is the arithmetic mean of r over 10
instances: several calm seasons (ability spread matches the generic forecast)
and several trap seasons (concentrated / bimodal / mid-season-shifting /
far-from-generic / very-tight-cap true ability) engineered so a policy that
always chases maximum information collapses on the back half of the season.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <arithmetic mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ..., r_10]
"""
import sys, json, math, random
import isorun

CAND_TIMEOUT = 20
VALID_FLOOR = 0.01
LEAK_PASS_RATE = 0.85
PRIOR_MEAN, PRIOR_SD = 0.0, 1.5
ANCHOR_JITTER_SD = 0.45  # per-examinee idiosyncratic offset applied to the SELECTION anchor only
GRID = [(-4.0 + 8.0 * k / 60.0) for k in range(61)]  # 61-pt quadrature grid on [-4,4]
INF_CAP = 10 ** 9


# ============================ math helpers ===================================
def _p_correct(a, b, theta):
    x = -a * (theta - b)
    if x > 35:
        return 0.0
    if x < -35:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def _normpdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z)


def _prior(grid, mu, sd):
    w = [_normpdf(x, mu, sd) for x in grid]
    s = sum(w)
    return [wi / s for wi in w]


def _post_mean(grid, post):
    return sum(x * w for x, w in zip(grid, post))


def _u01(seed, label, ex_idx, item_idx):
    return random.Random(f"{seed}|{label}|{ex_idx}|{item_idx}").random()


# ============================ instance family ================================
def _gen_items(seed, M):
    rng = random.Random(f"{seed}|bank")
    items = []
    for i in range(M):
        a = round(0.7 + rng.uniform(0.0, 1.4), 3)
        b = round(-3.0 + 6.0 * i / (M - 1) + rng.uniform(-0.18, 0.18), 3)
        items.append({"a": a, "b": b})
    return items


def _gen_thetas(seed, N, kind):
    rng = random.Random(f"{seed}|thetas")

    def gauss(mu, sd):
        return mu + sd * rng.gauss(0.0, 1.0)

    if kind.startswith("calm"):
        return [gauss(0.0, 1.0) for _ in range(N)]
    if kind == "trap_narrow":
        return [gauss(1.2, 0.25) for _ in range(N)]
    if kind == "trap_extreme_shift":
        return [gauss(2.3, 0.3) for _ in range(N)]
    if kind == "trap_bimodal":
        return [gauss(-1.8, 0.3) if i % 2 == 0 else gauss(1.8, 0.3) for i in range(N)]
    if kind == "trap_regime_shift":
        half = N // 2
        return [gauss(-1.5, 0.3) for _ in range(half)] + [gauss(1.5, 0.3) for _ in range(N - half)]
    if kind == "trap_tightcap":
        return [gauss(0.3, 0.9) for _ in range(N)]
    if kind == "trap_narrow_heldout":
        return [gauss(-0.9, 0.2) for _ in range(N)]
    raise ValueError(kind)


CAP_FRAC = {
    "calm_wide": 0.55, "calm_wide2": 0.55, "calm_wide3": 0.55, "calm_moderate": 0.45,
    "trap_narrow": 0.28, "trap_extreme_shift": 0.26, "trap_bimodal": 0.30,
    "trap_regime_shift": 0.30, "trap_tightcap": 0.20, "trap_narrow_heldout": 0.24,
}


def _cap_counts(cap_frac, N):
    return max(1, int(math.ceil(cap_frac * N)))


# ============================ core simulation =================================
def _run_population(items, thetas, L, cap_counts, policy, seed, label, hint_ratio, track_bias=True):
    """Simulate the whole examinee season for ONE policy. Returns (theta_hats, counts)."""
    M = len(items)
    N = len(thetas)
    counts = [0] * M
    iw = policy["info_weight"]; ew = policy["exposure_weight"]
    esh = policy["exposure_shape"]; ht = policy["hint_trust"]
    theta_hats = []
    for ex_idx in range(N):
        theta_true = thetas[ex_idx]
        anchor_jitter = random.Random(f"{seed}|anchor|{ex_idx}").gauss(0.0, 1.0) * ANCHOR_JITTER_SD
        used = [False] * M
        post = _prior(GRID, PRIOR_MEAN, PRIOR_SD)
        theta_hat = _post_mean(GRID, post)
        L_eff = min(L, M)
        for _step in range(L_eff):
            theta_sel = theta_hat + anchor_jitter  # idiosyncratic selection anchor; scoring uses theta_hat only
            best_i = -1
            best_score = float("-inf")
            for i in range(M):
                if used[i]:
                    continue
                a, b = items[i]["a"], items[i]["b"]
                p = _p_correct(a, b, theta_sel)
                info = a * a * p * (1.0 - p)
                cap = cap_counts[i]
                usage = counts[i] / cap if cap > 0 else 0.0
                pred = ht * hint_ratio[i] + (1.0 - ht) * usage
                penalty = ew * (max(pred, 0.0) ** esh)
                score = iw * info - penalty
                if score > best_score + 1e-12:
                    best_score = score
                    best_i = i
            a, b = items[best_i]["a"], items[best_i]["b"]
            counts[best_i] += 1
            used[best_i] = True
            compromised = counts[best_i] > cap_counts[best_i]
            p_true = LEAK_PASS_RATE if compromised else _p_correct(a, b, theta_true)
            u = _u01(seed, label, ex_idx, best_i)
            resp = 1 if u < p_true else 0
            # posterior update always uses the NOMINAL item model (unaware of any leak)
            new_post = []
            for x, w in zip(GRID, post):
                pg = _p_correct(a, b, x)
                lik = pg if resp == 1 else (1.0 - pg)
                new_post.append(w * max(lik, 1e-12))
            s = sum(new_post)
            post = [w / s for w in new_post]
            theta_hat = _post_mean(GRID, post)
        theta_hats.append(theta_hat)
    return theta_hats, counts


def _mse(theta_hats, thetas):
    n = len(thetas)
    return sum((h - t) ** 2 for h, t in zip(theta_hats, thetas)) / n


_GREEDY_POLICY = {"info_weight": 1.0, "exposure_weight": 0.0, "exposure_shape": 1.0, "hint_trust": 0.0}
_TRIVIAL_POLICY = {"info_weight": 0.0, "exposure_weight": 0.0, "exposure_shape": 1.0, "hint_trust": 0.0}


def _compute_hint(items, L, cap_counts_inf, seed):
    R = 200
    ref_thetas = [random.Random(f"{seed}|hintpop|{i}").gauss(0.0, 1.0) for i in range(R)]
    zero_hint = [0.0] * len(items)
    _, counts = _run_population(items, ref_thetas, L, cap_counts_inf, _GREEDY_POLICY,
                                 seed, "hint", zero_hint)
    return [c / R for c in counts]


# ============================ instance construction ===========================
def _build_instances():
    kinds = ["calm_wide", "calm_wide2", "trap_narrow", "trap_extreme_shift",
             "trap_bimodal", "calm_moderate", "trap_regime_shift", "trap_tightcap",
             "calm_wide3", "trap_narrow_heldout"]
    out = []
    for idx, kind in enumerate(kinds):
        seed = 20260726 + idx
        M, L, N = 40, 9, 80
        items = _gen_items(seed, M)
        thetas = _gen_thetas(seed, N, kind)
        cap_frac = CAP_FRAC[kind]
        cap_counts = [_cap_counts(cap_frac, N)] * M
        cap_counts_inf = [INF_CAP] * M
        hint = _compute_hint(items, L, cap_counts_inf, seed)
        out.append({
            "name": f"{kind}_{idx}", "seed": seed, "M": M, "L": L, "N": N,
            "items": items, "thetas": thetas, "cap_frac": cap_frac,
            "cap_counts": cap_counts, "cap_counts_inf": cap_counts_inf, "hint": hint,
        })
    return out


def _public_view(inst):
    hr = [round(h * inst["N"] / c, 6) for h, c in zip(inst["hint"], inst["cap_counts"])]
    return {
        "M": inst["M"], "L": inst["L"], "N": inst["N"],
        "items": [{"a": it["a"], "b": it["b"], "cap_frac": inst["cap_frac"]} for it in inst["items"]],
        "typical_demand_hint": [round(h, 6) for h in inst["hint"]],
        "leak_pass_rate": LEAK_PASS_RATE,
        "prior_mean": PRIOR_MEAN, "prior_sd": PRIOR_SD,
        "selection_anchor_jitter_sd": ANCHOR_JITTER_SD,
        "seed": inst["seed"],
    }


def _hint_ratio(inst):
    return [h * inst["N"] / c for h, c in zip(inst["hint"], inst["cap_counts"])]


def _valid_policy(ans):
    if not isinstance(ans, dict):
        return None
    try:
        iw = float(ans.get("info_weight"))
        ew = float(ans.get("exposure_weight"))
        esh = float(ans.get("exposure_shape"))
        ht = float(ans.get("hint_trust"))
    except (TypeError, ValueError):
        return None
    for v in (iw, ew, esh, ht):
        if not math.isfinite(v):
            return None
    iw = min(max(iw, 0.0), 50.0)
    ew = min(max(ew, 0.0), 50.0)
    esh = min(max(esh, 0.0), 6.0)
    ht = min(max(ht, 0.0), 1.0)
    return {"info_weight": iw, "exposure_weight": ew, "exposure_shape": esh, "hint_trust": ht}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = _public_view(inst)
        hr = _hint_ratio(inst)

        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        policy = _valid_policy(ans)
        if policy is None:
            vec.append(0.0)
            continue

        try:
            th_cand, _ = _run_population(inst["items"], inst["thetas"], inst["L"],
                                          inst["cap_counts"], policy, inst["seed"], "cand", hr)
            obj_cand = -_mse(th_cand, inst["thetas"])
        except Exception:
            vec.append(0.0)
            continue
        if not math.isfinite(obj_cand):
            vec.append(0.0)
            continue

        th_base, _ = _run_population(inst["items"], inst["thetas"], inst["L"],
                                      inst["cap_counts"], _TRIVIAL_POLICY, inst["seed"], "base", hr)
        obj_base = -_mse(th_base, inst["thetas"])

        th_upper, _ = _run_population(inst["items"], inst["thetas"], inst["L"],
                                       inst["cap_counts_inf"], _GREEDY_POLICY, inst["seed"], "upper",
                                       [0.0] * inst["M"])
        obj_upper = -_mse(th_upper, inst["thetas"])

        denom = max(obj_upper - obj_base, 1e-6)
        r = 0.1 + 0.9 * (obj_cand - obj_base) / denom
        r = max(0.0, min(1.0, r))
        r = max(r, VALID_FLOOR)
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
