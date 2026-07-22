#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0549 -- "Correlated Fleet Scheduling"
(family: correlated-multiway-partition; format B, quality-metric).

THEME.  A depot partitions N jobs across K identical machines to run in parallel.
A job's processing time is NOT fixed: every run it is perturbed by a few HIDDEN
low-rank factors shared across jobs (ambient load, a shared bus, a common feed).
Under this covariance some jobs move TOGETHER and some move in OPPOSITE directions.
The depot minimizes the EXPECTED MAKESPAN -- the average, over the fluctuation
distribution, of the slowest machine's total load.

WHY COVARIANCE MATTERS (the innovation hook).  Balancing only the MEAN load per
machine (textbook Longest-Processing-Time) ignores that co-locating two
ANTI-correlated jobs cancels their peaks: when one spikes the other dips, so their
machine's load stays flat and rarely becomes the bottleneck.  Split an
anti-correlated pair across two machines and BOTH turn volatile; the
max-over-machines picks up whichever spiked -> a higher expected makespan.  The
signal lives in the SECOND-ORDER structure, recoverable only by probing the
sampled scenarios the candidate is handed.

MECHANISMS COMPOSED.
  * covariance-sampling-probe : the candidate must infer the joint structure from a
    finite TRAIN sample of scenarios (means alone are insufficient).
  * longest-processing-time-seed : a natural first move (balance mean load) -- the
    `greedy` reference -- which this evaluator's instances make a TRAP.
  * pairwise-swap-refine : the covariance-aware improvement (regroup jobs so each
    machine's net factor loading -> 0) that the `strong` reference exploits.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N, "k": K, "s": S,
             "probe": [[t_00,...,t_0{N-1}], ...]}   # S train scenarios, each a
                                                    # length-N row of >=0 floats
  stdout: ONE JSON object:
            {"assign": [m_0, ..., m_{N-1}]}          # m_i in [0, K-1]

  VALID iff `assign` is a list of exactly N integers each in [0, K-1].  Anything
  else (wrong length, out of range, non-int, crash, timeout, non-JSON) -> 0.0.

SCORING (deterministic; no wall-time).  The probe is a TRAIN sample; the score is
computed on a DISJOINT HELD-OUT sample H of the same distribution, so overfitting
the probe does not pay.  For an assignment:
    makespan(scenario) = max_m  sum_{i: m_i=m} t_i(scenario)
    q_cand = mean over H of makespan(scenario)                  # expected makespan
  Two references THIS parent computes on H:
    q_base = expected makespan of the index round-robin assignment (i -> i % K)
             -- a weak reference ignoring sizes AND covariance
    q_lb   = mean over H of (sum of that scenario's times) / K  -- the perfectly
             balanced per-scenario load, an UNREACHABLE lower bound on makespan
  and normalize (round-robin -> 0.1; lower q_cand -> higher score):
    r = clamp( 0.1 + GAIN * (q_base - q_cand) / (q_base - q_lb), 0, 1 )
  Round-robin scores ~0.1; balancing means (LPT) does better; additionally
  co-locating anti-correlated jobs does better still.  q_lb is unreachable (a
  single assignment cannot balance every scenario at once), so even the strong
  reference stays well below the cap -> headroom for an RL policy above it.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap SUBPROCESS via
`isorun.run_candidate`; it sees ONLY the public probe.  The held-out sample, the
factor loadings and the references live in THIS parent, so a frame-walking /
filesystem-reading candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

GAIN = 0.50          # anchor slope (keeps the strong reference well below ~0.92)


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():                                  # uniform in [0, 1)
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return u


def _normals(u, count):
    """Deterministic standard normals via Box-Muller."""
    out = []
    while len(out) < count:
        u1 = u()
        u2 = u()
        if u1 < 1e-12:
            u1 = 1e-12
        rad = math.sqrt(-2.0 * math.log(u1))
        out.append(rad * math.cos(2.0 * math.pi * u2))
        out.append(rad * math.sin(2.0 * math.pi * u2))
    return out[:count]


# ----------------------------- instance family -----------------------------
# Each instance plants a rank-`R` covariance.  Factor 0 is the ANTI-correlation
# factor whose loading is A * pattern[j % k] -- ALIGNED with the machine cycle, so
# the index round-robin (and mean-balancing when means tie) piles same-sign jobs
# onto the same machine and maximizes its load swing.  Factors 1.. are a small,
# all-positive common factor (shared shock, uncancellable) plus idiosyncratic noise.
def _build_model(sp):
    n = sp["n"]; R = sp["R"]; pat = sp["pat"]
    up = _rng(sp["seed"] * 131 + 7)
    if sp["mhi"] <= sp["mlo"]:
        mu = [float(sp["mlo"]) for _ in range(n)]              # equal means (trap)
    else:
        mu = [sp["mlo"] + (sp["mhi"] - sp["mlo"]) * up() for _ in range(n)]
    L = [[0.0] * R for _ in range(n)]
    for j in range(n):
        L[j][0] = sp["A"] * pat[j % len(pat)] * (0.9 + 0.2 * up())
        for r in range(1, R):
            L[j][r] = sp["Acom"] * (0.5 + up())
    psi = [sp["psi"] * (0.7 + 0.6 * up()) for _ in range(n)]
    return mu, L, psi


def _sample(mu, L, psi, n, R, count, seed):
    """Draw `count` scenarios of the n processing times. Deterministic."""
    u = _rng(seed)
    rows = []
    for _ in range(count):
        z = _normals(u, R)
        e = _normals(u, n)
        row = []
        for j in range(n):
            v = mu[j] + psi[j] * e[j]
            Lj = L[j]
            for r in range(R):
                v += Lj[r] * z[r]
            if v < 0.05:
                v = 0.05
            row.append(round(v, 4))
        rows.append(row)
    return rows


def _specs():
    """Deterministic instance family (n_instances = 10).
    hetero: heterogeneous means, weak anti  -> LPT >> round-robin, strong ~ LPT.
    trap:   equal means, strong ALIGNED anti -> LPT lands far from covariance-aware.
    mixed:  spread means + strong anti       -> LPT commits to mean-balance, misses it."""
    return [
        dict(seed=1101, n=15, k=3, R=2, mlo=4, mhi=22, A=1.0, psi=1.1, Acom=1.3, pat=[1, -1, 0], type="hetero"),
        dict(seed=1102, n=18, k=3, R=2, mlo=3, mhi=20, A=1.1, psi=1.0, Acom=1.2, pat=[1, -1, 0], type="hetero"),
        dict(seed=2201, n=15, k=3, R=2, mlo=2, mhi=2, A=6.0, psi=0.30, Acom=0.0, pat=[1, -1, 0], type="trap"),
        dict(seed=2202, n=15, k=3, R=2, mlo=2, mhi=2, A=6.5, psi=0.30, Acom=0.0, pat=[1, -1, 0], type="trap"),
        dict(seed=2204, n=15, k=3, R=2, mlo=2, mhi=2, A=6.5, psi=0.30, Acom=0.0, pat=[1, -1, 0], type="trap"),
        dict(seed=3301, n=15, k=3, R=2, mlo=2, mhi=7, A=6.0, psi=0.40, Acom=0.3, pat=[1, -1, 0], type="mixed"),
        dict(seed=3302, n=15, k=3, R=2, mlo=2, mhi=8, A=6.0, psi=0.40, Acom=0.3, pat=[1, -1, 0], type="mixed"),
        dict(seed=3303, n=15, k=3, R=2, mlo=2, mhi=7, A=5.5, psi=0.40, Acom=0.3, pat=[1, -1, 0], type="mixed"),
        dict(seed=3304, n=15, k=3, R=2, mlo=2, mhi=8, A=6.0, psi=0.40, Acom=0.3, pat=[1, -1, 0], type="mixed"),
        dict(seed=3305, n=15, k=3, R=2, mlo=2, mhi=7, A=6.0, psi=0.40, Acom=0.3, pat=[1, -1, 0], type="mixed"),
    ]


def _build_instances():
    insts = []
    for sp in _specs():
        n = sp["n"]; R = sp["R"]
        mu, L, psi = _build_model(sp)
        probe = _sample(mu, L, psi, n, R, 80, sp["seed"] * 977 + 13)      # TRAIN
        held = _sample(mu, L, psi, n, R, 300, sp["seed"] * 6151 + 101)    # HELD-OUT
        insts.append({"name": f"fleet{sp['seed']}", "n": n, "k": sp["k"],
                      "s": 80, "probe": probe, "held": held})
    return insts


# ----------------------------- references / scoring ------------------------
def _expected_makespan(scenarios, assign, n, k):
    total = 0.0
    for row in scenarios:
        loads = [0.0] * k
        for i in range(n):
            loads[assign[i]] += row[i]
        m = loads[0]
        for x in loads[1:]:
            if x > m:
                m = x
        total += m
    return total / len(scenarios)


def _mean_load_lb(scenarios, k):
    total = 0.0
    for row in scenarios:
        total += sum(row) / k
    return total / len(scenarios)


def _round_robin(n, k):
    return [i % k for i in range(n)]


def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    n = inst["n"]; k = inst["k"]
    if len(assign) != n:
        return None
    for g in assign:
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < 0 or g >= k:
            return None
    return assign


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n = inst["n"]; k = inst["k"]
        held = inst["held"]
        q_base = _expected_makespan(held, _round_robin(n, k), n, k)
        q_lb = _mean_load_lb(held, k)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": n, "k": k,
                  "s": inst["s"], "probe": inst["probe"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            assign = _validate(inst, ans)
        except Exception:
            assign = None
        if assign is None:
            vec.append(0.0)
            continue
        q_cand = _expected_makespan(held, assign, n, k)
        r = 0.1 + GAIN * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
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
