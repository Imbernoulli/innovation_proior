#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0962 -- "Price a product to spark network adoption"
(family: externality-adoption-pricing; format B, quality-metric).

THEME.  A product launches across M regional markets that sit on a fixed influence
network.  Region i's population has a base willingness-to-pay `base_i`, but that
willingness RISES with how much OTHER regions have already adopted (a positive
NETWORK EXTERNALITY): the perceived value in region i is

    v_i(x) = base_i + gamma_i * sum_j W[i][j] * x_j

where `x_j` in [0,1] is the adoption FRACTION in region j and `W[i][j] >= 0` is how
much region j's adoption feeds region i's value (the influence network; it need not
be symmetric -- a "hub" region can shape everyone downstream while depending on no
one itself).  Consumers in region i have idiosyncratic taste around that perceived
value with spread `spread_i`; given the region's price `p_i`, the adoption fraction
that a BEST-RESPONDING population settles on is the logistic

    x_i = 1 / (1 + exp((p_i - v_i(x)) / spread_i)).

Because `x_i` depends on `x_j` and vice versa, the market-wide adoption vector is a
BEST-RESPONSE EQUILIBRIUM: starting from zero adoption (no one owns the product yet),
everyone repeatedly best-responds to everyone else's current adoption until nothing
moves.  This process is monotone (adoption can only rise as it iterates), so it
converges to a well-defined fixed point -- but that fixed point depends heavily on
where each region's price sits relative to its network-boosted value.

YOUR JOB.  Choose ONE price per region (`region-price-setting`).  You are graded on
the total revenue `sum_i p_i * x_i * pop_i` at the induced equilibrium, where `pop_i`
is region i's market size.  Higher prices earn more per adopter but suppress local
(and, through the network, downstream) adoption; lower prices spread adoption but
sacrifice margin.

THE TRAP (myopic per-region pricing).  A region-blind monopolist prices each region
in ISOLATION -- maximizing `p_i * x_i` assuming `v_i = base_i` (i.e. pretending no one
else ever adopts, so the network term is ignored entirely).  This ignores that a
region with modest local value but *large downstream influence* (`gamma_j * W[j][i]`
summed over many dependent regions j) is worth far more to the network than to
itself: pricing it at its own myopic optimum can leave the whole cascade below the
threshold needed to lift the regions that depend on it.

THE INSIGHT (sacrifice margin to ignite the cascade).  Identify regions whose
adoption has large downstream value (`sum_j gamma_j * W[j][i] * pop_j`), price them
BELOW their own myopic optimum -- even toward zero -- to push their adoption toward
1, then RE-PRICE every other region upward against the value the cascade now
delivers to it.  The local margin given up in the "hub" region is repaid many times
over in the regions it feeds.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "m": M, "base": [..], "gamma": [..], "spread": [..],
             "pop": [..], "W": [[..]]}   # W[i][j] = j's influence on i, MxM
  stdout: ONE JSON object:
            {"prices": [p_0, ..., p_{M-1}]}   # p_i finite, 0 <= p_i <= 1e6
  A price vector is VALID iff it is a list of exactly M finite numbers in [0, 1e6].
  Anything else (wrong length, negative, NaN/inf, crash, timeout, non-JSON) makes
  that instance score 0.0.

SCORING (deterministic; no wall-time).  For each instance we precomputed, ONCE while
authoring this problem (never at grading time), two reference revenues from the same
public data: `R_flat` -- the revenue of the best SINGLE flat price applied to every
region (a naive, network-blind, region-blind baseline) and `R_ceil_raw` -- the best
revenue found by an extensive multi-start coordinate-ascent search that explicitly
tries sacrificing every plausible subset of high-influence regions. We inflate the
ceiling by `CEIL_SCALE = 1.3` (the general contract's prescribed remedy for "the
reference solution nearly saturates the score") so headroom remains above what that
search found. Then, at grading time, we compute your `R_cand` at the equilibrium
YOUR prices induce and normalise:

    r = clamp( 0.1 + 0.9 * (R_cand - R_flat) / (CEIL_SCALE * (R_ceil_raw - R_flat)), 0, 1 )

The flat baseline scores ~0.1; the coordinate-ascent search would score ~1/1.3 = 0.77;
matching or beating it scores higher, up to the 1.0 cap. Pricing worse than the flat
baseline scores below 0.1.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a fresh subprocess
via `isorun.run_candidate`; it only ever sees the PUBLIC instance. The reference
revenues were computed OFFLINE while authoring this problem and are baked in as
constants -- an introspecting/frame-walking candidate learns nothing useful from
the live evaluator process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

CEIL_SCALE = 1.3

# ----------------------------- instance family (frozen, precomputed offline) -----
_INSTANCES = [
    {
        "name": "region11001", "m": 6,
        "base": [26.290539, 20.978261, 28.396189, 22.630705, 6.9076, 28.574295],
        "gamma": [26.0, 26.0, 26.0, 26.0, 0.3, 26.0],
        "spread": [7.361351, 5.873913, 7.950933, 6.336597, 1.7269, 8.000803],
        "pop": [57.128215, 50.552229, 52.333975, 44.740747, 86.053153, 42.446583],
        "W": [[0.0, 0.0, 0.0, 0.0, 1.079342, 0.0], [0.0, 0.0, 0.0, 0.0, 1.142884, 0.0], [0.0, 0.0, 0.0, 0.0, 1.17005, 0.0], [0.0, 0.0, 0.0, 0.0, 0.926914, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.149259, 0.0]],
        "R_flat": 3301.941313, "R_ceil_raw": 8647.304511,
    },
    {
        "name": "region22002", "m": 7,
        "base": [22.525682, 27.23571, 19.237385, 26.854102, 5.419817, 26.550681, 21.039787],
        "gamma": [30.0, 30.0, 30.0, 30.0, 0.3, 30.0, 30.0],
        "spread": [5.631421, 6.808928, 4.809346, 6.713525, 1.19236, 6.63767, 5.259947],
        "pop": [51.385757, 31.378049, 44.325031, 46.222874, 88.065391, 35.514347, 35.002112],
        "W": [[0.0, 0.061476, 0.0, 0.0, 1.154173, 0.0, 0.0], [0.0, 0.0, 0.058012, 0.0, 1.150447, 0.0, 0.0], [0.0, 0.0, 0.0, 0.043379, 1.183271, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.230821, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.036671, 0.0], [0.0, 0.0, 0.0, 0.0, 1.195279, 0.0, 0.042462], [0.06031, 0.0, 0.0, 0.0, 1.077178, 0.0, 0.0]],
        "R_flat": 3270.297233, "R_ceil_raw": 10251.026433,
    },
    {
        "name": "region33003", "m": 6,
        "base": [10.373647, 7.967346, 24.52251, 28.919316, 30.54625, 27.4584],
        "gamma": [0.3, 0.3, 18.0, 18.0, 18.0, 18.0],
        "spread": [3.112094, 2.390204, 7.356753, 8.675795, 9.163875, 8.23752],
        "pop": [80.034507, 86.482708, 45.090333, 38.383863, 31.757384, 38.857506],
        "W": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.640624, 0.660179, 0.0, 0.0, 0.0, 0.0], [0.578004, 0.58886, 0.0, 0.0, 0.0, 0.0], [0.690981, 0.562223, 0.0, 0.0, 0.0, 0.0], [0.523351, 0.661362, 0.0, 0.0, 0.0, 0.0]],
        "R_flat": 2308.938273, "R_ceil_raw": 4940.693933,
    },
    {
        "name": "region44004", "m": 8,
        "base": [28.429236, 23.585671, 28.063622, 20.684875, 27.900453, 20.547247, 9.249805, 27.820551],
        "gamma": [22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 0.35, 22.0],
        "spread": [7.675894, 6.368131, 7.577178, 5.584916, 7.533122, 5.547757, 2.219953, 7.511549],
        "pop": [42.016932, 56.468511, 50.205421, 30.366144, 52.062968, 35.925478, 80.869211, 51.810514],
        "W": [[0.0, 0.036745, 0.0, 0.0, 0.0, 0.0, 0.820284, 0.0], [0.0, 0.0, 0.060124, 0.0, 0.0, 0.0, 0.857873, 0.0], [0.0, 0.0, 0.0, 0.05025, 0.0, 0.0, 0.751215, 0.0], [0.0, 0.0, 0.0, 0.0, 0.06167, 0.0, 0.792335, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.037766, 0.835863, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.056963, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045362], [0.056107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.951601, 0.0]],
        "R_flat": 4506.814944, "R_ceil_raw": 9137.830074,
    },
    {
        "name": "region55005", "m": 8,
        "base": [27.579561, 13.989299, 23.207927, 26.8235, 24.342505, 28.401812, 30.915731, 30.850244],
        "gamma": [10.0, 0.6, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "spread": [9.652846, 4.476576, 8.122775, 9.388225, 8.519877, 9.940634, 10.820506, 10.797585],
        "pop": [49.171832, 90.462254, 32.026979, 58.974903, 35.26843, 49.328536, 47.696363, 33.343094],
        "W": [[0.0, 1.02561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.077909, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.824628, 0.0, 0.09044, 0.0, 0.0, 0.0, 0.0], [0.0, 0.888376, 0.0, 0.0, 0.111566, 0.0, 0.0, 0.0], [0.0, 0.919557, 0.0, 0.0, 0.0, 0.111015, 0.0, 0.0], [0.0, 1.009474, 0.0, 0.0, 0.0, 0.0, 0.116248, 0.0], [0.0, 0.826386, 0.0, 0.0, 0.0, 0.0, 0.0, 0.072726], [0.115158, 1.053215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "R_flat": 4888.219993, "R_ceil_raw": 6485.437968,
    },
    {
        "name": "region66006", "m": 6,
        "base": [23.030175, 21.383908, 26.969594, 22.681786, 13.981063, 24.627646],
        "gamma": [9.0, 9.0, 9.0, 9.0, 0.7, 9.0],
        "spread": [8.060561, 7.484368, 9.439358, 7.938625, 4.47394, 8.619676],
        "pop": [49.372377, 56.326733, 49.224869, 43.848537, 116.778218, 48.473891],
        "W": [[0.0, 0.099751, 0.0, 0.0, 0.680204, 0.0], [0.0, 0.0, 0.122593, 0.0, 0.669137, 0.0], [0.0, 0.0, 0.0, 0.073487, 0.682619, 0.0], [0.0, 0.0, 0.0, 0.0, 0.804127, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.116695], [0.078986, 0.0, 0.0, 0.0, 0.924158, 0.0]],
        "R_flat": 3854.188438, "R_ceil_raw": 4666.22255,
    },
    {
        "name": "region77007", "m": 7,
        "base": [22.746363, 9.580272, 21.180914, 28.701029, 28.857727, 21.503806, 11.689109],
        "gamma": [12.0, 0.5, 12.0, 12.0, 12.0, 12.0, 0.5],
        "spread": [7.278836, 2.874082, 6.777893, 9.184329, 9.234473, 6.881218, 3.506733],
        "pop": [53.640632, 83.100613, 46.489616, 33.302251, 57.679831, 48.332831, 94.991884],
        "W": [[0.0, 0.822297, 0.0, 0.0, 0.0, 0.0, 0.80947], [0.0, 0.0, 0.090587, 0.0, 0.0, 0.0, 0.0], [0.0, 0.806477, 0.0, 0.056257, 0.0, 0.0, 0.590386], [0.0, 0.791165, 0.0, 0.0, 0.070416, 0.0, 0.835582], [0.0, 0.560611, 0.0, 0.0, 0.0, 0.097584, 0.698725], [0.0, 0.832439, 0.0, 0.0, 0.0, 0.0, 0.675844], [0.077383, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "R_flat": 3599.060124, "R_ceil_raw": 6410.825889,
    },
    {
        "name": "region88008", "m": 6,
        "base": [25.829273, 14.418321, 24.270651, 28.112675, 25.3915, 24.13493],
        "gamma": [7.0, 0.9, 7.0, 7.0, 7.0, 7.0],
        "spread": [10.331709, 5.478962, 9.70826, 11.24507, 10.1566, 9.653972],
        "pop": [52.270042, 94.182046, 41.981226, 37.491654, 54.801183, 44.884815],
        "W": [[0.0, 0.732023, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.108516, 0.0, 0.0, 0.0], [0.0, 0.895377, 0.0, 0.126701, 0.0, 0.0], [0.0, 0.784923, 0.0, 0.0, 0.101458, 0.0], [0.0, 0.688297, 0.0, 0.0, 0.0, 0.10028], [0.144561, 0.86818, 0.0, 0.0, 0.0, 0.0]],
        "R_flat": 3619.380572, "R_ceil_raw": 4201.58658,
    },
    {
        "name": "region99009", "m": 9,
        "base": [25.930478, 10.893234, 26.437737, 27.803314, 29.363425, 24.313589, 28.624727, 26.491138, 23.605476],
        "gamma": [20.0, 0.4, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "spread": [7.519839, 2.832241, 7.666944, 8.062961, 8.515393, 7.050941, 8.301171, 7.68243, 6.845588],
        "pop": [44.704424, 119.1406, 52.134002, 48.332146, 47.460224, 55.740633, 56.907049, 47.173988, 45.022008],
        "W": [[0.0, 0.974782, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.064779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.830984, 0.0, 0.072439, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.007181, 0.0, 0.0, 0.076161, 0.0, 0.0, 0.0, 0.0], [0.0, 0.924417, 0.0, 0.0, 0.0, 0.051398, 0.0, 0.0, 0.0], [0.0, 0.78443, 0.0, 0.0, 0.0, 0.0, 0.048036, 0.0, 0.0], [0.0, 0.877532, 0.0, 0.0, 0.0, 0.0, 0.0, 0.068987, 0.0], [0.0, 1.014609, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.057813], [0.068563, 1.104999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "R_flat": 5963.382754, "R_ceil_raw": 11283.882627,
    },
    {
        "name": "region10010", "m": 7,
        "base": [30.49337, 27.894148, 29.151177, 29.259273, 17.417903, 25.515991, 24.05565],
        "gamma": [9.0, 9.0, 9.0, 9.0, 1.1, 9.0, 9.0],
        "spread": [11.587481, 10.599776, 11.077447, 11.118524, 6.270445, 9.696077, 9.141147],
        "pop": [41.329935, 41.661272, 55.046462, 49.73572, 109.886283, 49.554175, 35.922725],
        "W": [[0.0, 0.105364, 0.0, 0.0, 1.046308, 0.0, 0.0], [0.0, 0.0, 0.139313, 0.0, 0.869208, 0.0, 0.0], [0.0, 0.0, 0.0, 0.142775, 0.790487, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.031375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.132014, 0.0], [0.0, 0.0, 0.0, 0.0, 0.790834, 0.0, 0.115489], [0.114119, 0.0, 0.0, 0.0, 1.042337, 0.0, 0.0]],
        "R_flat": 5017.266946, "R_ceil_raw": 5869.119011,
    },
]


# ----------------------------- equilibrium + revenue (pure python) ---------------
def _sigmoid(z):
    if z > 40.0:
        return 0.0
    if z < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _fixed_point(base, gamma, spread, W, p, T=300):
    m = len(base)
    x = [0.0] * m
    for _ in range(T):
        xn = [0.0] * m
        for i in range(m):
            v = base[i] + gamma[i] * sum(W[i][j] * x[j] for j in range(m))
            xn[i] = _sigmoid((p[i] - v) / spread[i])
        x = xn
    return x


def _revenue(x, p, pop):
    return sum(p[i] * x[i] * pop[i] for i in range(len(p)))


# ----------------------------- validation -----------------------------------------
def _valid_prices(inst, answer):
    if not isinstance(answer, dict):
        return None
    pr = answer.get("prices")
    if not isinstance(pr, list) or len(pr) != inst["m"]:
        return None
    out = []
    for v in pr:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if v < 0.0 or v > 1e6:
            return None
        out.append(v)
    return out


def _score_instance(inst, prices):
    x = _fixed_point(inst["base"], inst["gamma"], inst["spread"], inst["W"], prices)
    R_cand = _revenue(x, prices, inst["pop"])
    denom = CEIL_SCALE * (inst["R_ceil_raw"] - inst["R_flat"])
    if denom < 1e-9:
        denom = 1e-9
    r = 0.1 + 0.9 * (R_cand - inst["R_flat"]) / denom
    if r != r or r in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, r))


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    vec = []
    for inst in _INSTANCES:
        public = {"name": inst["name"], "m": inst["m"], "base": list(inst["base"]),
                  "gamma": list(inst["gamma"]), "spread": list(inst["spread"]),
                  "pop": list(inst["pop"]), "W": [list(r) for r in inst["W"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            prices = _valid_prices(inst, ans)
        except Exception:
            prices = None
        if prices is None:
            vec.append(0.0)
            continue
        try:
            r = _score_instance(inst, prices)
        except Exception:
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
