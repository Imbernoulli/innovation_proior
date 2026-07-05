#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0354 -- "Coral Reef Survey: Pre-Positioning Sample Kits
across a Dive Supply Tree" (family: constrained-OR; format B, quality-metric).

THEME.  A marine institute runs periodic surveys of a coral reef.  Before each
expedition it must PRE-POSITION single-use sample kits (vials + specimen tags) at
N dive SITES, and it may also hold a shared reserve of kits aboard the research
VESSEL (a central depot that can be ferried to whichever site runs short).  The
number of specimens actually encountered at each site on survey day is UNCERTAIN;
its distribution (mean / spread / shape) is known from prior seasons, but the exact
count is not.  This is a two-echelon safety-stock / newsvendor problem skinned as a
reef survey: kits pre-positioned locally = local safety stock, kits on the vessel =
pooled central safety stock, over-provisioning = holding cost, missed specimens =
shortage cost, ferrying from the vessel = a transfer cost, and a per-site service
level (probability the site's own kits suffice) must be respected.  MINIMIZE the
deterministic composite objective (holding + shortage + transfer + service penalty).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {
      "name": str,
      "n_sites": N,
      "sites": [ {"mean": float, "std": float, "dist": "normal"|"lognormal"|"bimodal",
                  "h": float,   # local holding cost per unused kit at the site
                  "p": float,   # shortage penalty per specimen missed
                  "c": float},  # acquisition cost per kit placed at the site
                 ... N entries ],
      "central": {"h0": float,  # holding cost per unused kit left on the vessel
                  "t":  float,  # transfer cost per kit ferried from vessel to a site
                  "c0": float}, # acquisition cost per kit stocked on the vessel
      "budget": float,          # cap on total acquisition spend
      "tau":    float,          # per-site target service level (no-stockout prob.)
      "lam":    float,          # weight of the service-level penalty term
      "n_scenarios": S          # number of hidden survey-day scenarios used to score
    }
  stdout: ONE JSON object:
      {"q": [q_0, ..., q_{N-1}], "q0": q0}
    where q_i >= 0 is the number of kits pre-positioned at site i and q0 >= 0 is the
    number of kits held on the vessel.  Values may be fractional; all must be finite.

  FEASIBILITY (candidate-verifiable from PUBLIC data): the acquisition spend
      sum_i c_i * q_i + c0 * q0
  must not exceed `budget` (a 1e-6 relative tolerance is allowed).  A missing/extra
  entry, a non-finite / negative value, a budget violation, a crash, a timeout, or
  non-JSON -> that instance scores 0.0.

SCORING (deterministic; NO wall-time).  The evaluator draws S HIDDEN survey-day
scenarios from each site's declared distribution (seed kept in this parent process;
the candidate never sees it).  For each scenario k the cost is computed as follows.
  * local leftover_i  = max(0, q_i - d_ik)       -> local holding  h_i * leftover_i
  * local shortfall_i = max(0, d_ik - q_i)
  * The vessel reserve q0 is ferried to cover shortfalls, DETERMINISTICALLY, in
    decreasing order of (p_i - t): only sites with p_i > t are helped, each covered
    kit costs t (transfer) and removes one unit of shortage, until the reserve runs
    out.  Uncovered shortfall_i costs p_i each; the reserve left unused costs h0 each.
  scenario_cost = sum_i [ h_i*leftover_i + p_i*unmet_i + t*covered_i ] + h0*unused_res
The scenario-average of that is the operating cost.  A SERVICE penalty is added:
for each site let phat_i = fraction of scenarios in which its LOCAL kits alone
sufficed (d_ik <= q_i); the penalty is
  service_penalty = lam * sum_i max(0, tau - phat_i) * p_i * mean_i .
The instance objective is  obj = operating_cost + service_penalty  (MINIMIZE).

The evaluator computes an internal BASELINE objective b by pre-positioning each site
to exactly its mean (q_i = mean_i, no vessel reserve) -- a weak, service-poor policy.
The per-instance normalized score is
  r = min(1.0, 0.1 * b / max(obj, 1e-12))
so a candidate matching the baseline scores ~0.1, and a candidate must be ~10x
cheaper than the baseline to approach 1.0 (the pooled critical-fractile optimum is
far from that, so strong policies keep headroom below 1.0).

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The hidden scenario
seed, the sampled demands, and the baseline are computed by THIS parent, so a
frame-walking / filesystem-snooping candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ------------------------------- deterministic RNG --------------------------
def _rng(seed):
    state = (int(seed) * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u


def _normals(u, k):
    out = []
    while len(out) < k:
        u1 = u()
        u2 = u()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2))
        out.append(r * math.sin(2 * math.pi * u2))
    return out[:k]


def _inv_norm(p):
    if p <= 0:
        return -1e9
    if p >= 1:
        return 1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ------------------------------- instance family ----------------------------
def _build_instance(seed, N, tighten):
    u = _rng(seed)
    sites = []
    for _ in range(N):
        mean = 20 + u() * 60
        cv = 0.3 + u() * 0.4
        std = mean * cv
        h = 1 + u() * 2
        p = 6 + u() * 9
        c = 1 + u() * 1.0
        dt = ["normal", "lognormal", "bimodal"][int(u() * 3) % 3]
        sites.append({"mean": mean, "std": std, "dist": dt,
                      "h": h, "p": p, "c": c})
    central = {"h0": 1.0, "t": 2.0 + u() * 2.0, "c0": 1.0}
    tau = 0.90
    lam = 0.1
    S = 200
    tot = 0.0
    for s in sites:
        z = _inv_norm(s["p"] / (s["p"] + s["h"]))
        tot += s["c"] * (s["mean"] + z * s["std"])
    B = tighten * tot
    return {"seed": seed, "N": N, "sites": sites, "central": central,
            "tau": tau, "lam": lam, "S": S, "B": B}


def make_instances():
    specs = [
        (101, 6, 1.30), (102, 7, 1.30), (103, 8, 1.25), (104, 6, 1.35),
        (105, 7, 1.20), (106, 8, 1.30), (207, 9, 1.25), (208, 10, 1.30),
        # harder / larger held-out instances (more sites, tighter/looser budgets)
        (311, 11, 1.20), (312, 12, 1.30), (313, 10, 1.35), (314, 12, 1.25),
    ]
    out = []
    for seed, N, tight in specs:
        inst = _build_instance(seed, N, tight)
        public = {
            "name": f"reef{seed}",
            "n_sites": inst["N"],
            "sites": [dict(s) for s in inst["sites"]],
            "central": dict(inst["central"]),
            "budget": inst["B"],
            "tau": inst["tau"],
            "lam": inst["lam"],
            "n_scenarios": inst["S"],
        }
        out.append({"public": public, "hidden": {"seed": inst["seed"]}})
    return out


# ------------------------------- hidden scenarios ---------------------------
def _scenarios(public, seed):
    u = _rng(seed * 7919 + 13)
    S = public["n_scenarios"]
    D = []
    for s in public["sites"]:
        z = _normals(u, S)
        col = []
        for k in range(S):
            if s["dist"] == "normal":
                d = s["mean"] + s["std"] * z[k]
            elif s["dist"] == "lognormal":
                cv = s["std"] / s["mean"]
                sig = math.sqrt(math.log(1 + cv * cv))
                mu = math.log(s["mean"]) - sig * sig / 2
                d = math.exp(mu + sig * z[k])
            else:  # bimodal
                if u() < 0.5:
                    d = s["mean"] * 0.4 + s["std"] * z[k] * 0.3
                else:
                    d = s["mean"] * 1.6 + s["std"] * z[k] * 0.3
            col.append(max(0.0, d))
        D.append(col)
    return D


# ------------------------------- objective ----------------------------------
def _objective(public, D, q, q0):
    sites = public["sites"]
    cen = public["central"]
    N = public["n_sites"]
    S = public["n_scenarios"]
    order = sorted(range(N), key=lambda i: (sites[i]["p"] - cen["t"]), reverse=True)
    total = 0.0
    stockout = [0] * N
    for k in range(S):
        rem = q0
        sc = 0.0
        short = [0.0] * N
        left = [0.0] * N
        for i in range(N):
            d = D[i][k]
            if q[i] >= d:
                left[i] = q[i] - d
            else:
                short[i] = d - q[i]
                stockout[i] += 1
        for i in order:
            if short[i] > 0 and sites[i]["p"] > cen["t"] and rem > 0:
                cov = short[i] if short[i] < rem else rem
                rem -= cov
                sc += cen["t"] * cov
                short[i] -= cov
        for i in range(N):
            sc += sites[i]["h"] * left[i] + sites[i]["p"] * short[i]
        sc += cen["h0"] * rem
        total += sc
    avg = total / S
    serv = 0.0
    for i in range(N):
        phat = 1.0 - stockout[i] / S
        serv += public["lam"] * max(0.0, public["tau"] - phat) * sites[i]["p"] * sites[i]["mean"]
    return avg + serv


def baseline(inst):
    public = inst["public"]
    D = _scenarios(public, inst["hidden"]["seed"])
    q = [s["mean"] for s in public["sites"]]
    return _objective(public, D, q, 0.0)


# ------------------------------- validation + score -------------------------
def _finite_nonneg(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and x == x and x not in (float("inf"), float("-inf")) and x >= -1e-9


def score(inst, answer):
    public = inst["public"]
    N = public["n_sites"]
    if not isinstance(answer, dict):
        return False, 0.0
    q = answer.get("q")
    q0 = answer.get("q0", 0.0)
    if not isinstance(q, list) or len(q) != N:
        return False, 0.0
    if not _finite_nonneg(q0):
        return False, 0.0
    for x in q:
        if not _finite_nonneg(x):
            return False, 0.0
    q = [max(0.0, float(x)) for x in q]
    q0 = max(0.0, float(q0))
    sites = public["sites"]
    spend = sum(sites[i]["c"] * q[i] for i in range(N)) + public["central"]["c0"] * q0
    if spend > public["budget"] * (1 + 1e-6):
        return False, 0.0
    D = _scenarios(public, inst["hidden"]["seed"])
    obj = _objective(public, D, q, q0)
    if not (obj == obj) or obj in (float("inf"), float("-inf")):
        return False, 0.0
    return True, obj


# ------------------------------- driver -------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        if not (r == r) or r in (float("inf"), float("-inf")) or r < 0.0:
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
