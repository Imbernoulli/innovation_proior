#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0961 -- "SOC Sensor Fusion: Budgeting the 1% Alert
Allowance" (family: stealth-frontier-detector-tuning; format B, quality-metric).

THEME.  A SOC (security operations center) watches K independent telemetry
SENSORS (netflow entropy, auth timing, DNS query rate, ...).  Every event --
benign background traffic, or one of several attack FAMILIES graded from overt
to near-benign STEALTH levels -- produces one real-valued reading per sensor.
The SOC must publish, per sensor, an alert THRESHOLD, plus FUSION weights and a
vote threshold that combine per-sensor alerts into one verdict.  A hard SLA
caps the FALSE-POSITIVE rate on a fixed benign event set at 1%; violate it and
the whole layout is worthless (score 0).

MECHANISM.  The 1% false-positive allowance is a single, SHARED, divisible
resource: making one sensor more sensitive (to catch a stealthier family) uses
up some of it; making another sensor stricter frees some back.  Each attack
family has its own "fingerprint" sensor where its signal survives deepest into
stealth; other sensors show it nothing but ordinary background noise.  A naive
tuner treats the 1% as a constraint to just barely satisfy uniformly (split it
evenly over every sensor -- a "common operating point"); an insightful tuner
treats it as a currency to be REBUDGETED non-uniformly, starving sensors that
guard already-easy families to fund the sensor guarding the hardest one.  We
score the WORST-served family's stealth-weighted detection, so a layout that
looks good on average but abandons one family scores poorly.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the
          exact schema (channels, fp_cap, benign matrix, families/levels,
          attacks[f][l] = list of per-family per-stealth-level sensor-reading
          vectors).
  stdout: ONE JSON object:
            {"theta": [t_0..t_{K-1}], "w": [w_0..w_{K-1}], "tau": T}
          theta_c is sensor c's alert threshold; w_c >= 0 is its fusion
          weight; an event is flagged iff sum_c w_c * 1[reading_c > theta_c]
          >= T.  Any malformed/non-finite/wrong-length answer, a crash, a
          timeout, or non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Let FP = fraction of the fixed benign
set flagged.  If FP > fp_cap (1%) the instance scores 0.0.  Otherwise, for
each family f, let det(f, l) = fraction of that family's stealth-level-l
attack samples flagged; the family's stealth-weighted detection quality is
    q_f = sum_l (l+1)*det(f,l)  /  sum_l (l+1)          (l = 0..levels-1)
(stealthier levels carry more weight, matching "graded from overt to
near-benign").  The instance's raw quality is q = min_f q_f -- the worst-served
family, so hoarding the allowance on the easy families cannot hide a starved
hard one.  We normalize with an affine anchor against two references computed
by THIS evaluator, never seen by the candidate:
    q_base = q achieved using ONLY sensor 0, thresholded at the full private
             1% allowance (a weak, single-sensor reference)
    q_ub   = q achieved giving EVERY sensor its own FULL private 1% allowance
             simultaneously (ignores that the allowance must be shared --
             a generous, unreachable ceiling: no config that truly respects
             one shared 1% cap can ever beat it)
    r = clamp( 0.1 + 0.9 * (q - q_base) / max(1e-9, q_ub - q_base), 0, 1 )
Matching the weak single-sensor reference scores ~0.1; approaching the
unreachable ceiling approaches 1.0 (never reached in practice, real headroom
remains).  The reported Ratio is the mean of r over 10 instances; Vector lists
the per-instance r.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance.  q_base/q_ub and
the full attack/benign data live only in the PARENT process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return u


L_LEVELS = 5
M_SAMPLES = 120
SCALE = 100.0
FP_CAP = 0.01
GAMMA = 5.0  # power-transform exponent: compresses benign/background noise
             # toward 0 so the extreme upper tail (where alert thresholds
             # live) is SPARSE -- a fixed slice of the FP allowance buys a
             # materially different threshold depending how it's spent.

TIERS = {
    "hard":   (10.0, 16.0, 0.32, 0.42),
    "medium": (26.0, 36.0, 0.12, 0.20),
    "easy":   (55.0, 70.0, 0.015, 0.04),
}

FAMILY_NAMES = ["c2-beacon", "credential-stuffing", "dns-tunnel", "lateral-movement",
                "data-exfil", "priv-escalation", "web-shell", "living-off-land"]
CHANNEL_NAMES = ["netflow-entropy", "auth-timing", "dns-query-rate", "proc-tree-depth",
                  "syscall-burst", "tls-fingerprint-drift", "endpoint-mem-delta"]

# spec: (seed, K, tiers_list, n_benign)
SPECS = [
    (101, 4, ["hard", "easy", "easy", "easy"], 1800),
    (102, 5, ["hard", "medium", "easy", "easy", "easy"], 1800),
    (103, 4, ["hard", "hard", "easy", "easy", "easy"], 2000),           # F>K: channel reuse
    (104, 6, ["hard", "medium", "medium", "easy", "easy", "easy"], 2000),
    (105, 6, ["hard", "easy", "easy", "easy"], 2000),                    # K>F: idle channels
    (106, 4, ["hard", "medium", "easy", "easy"], 1800),
    (107, 7, ["hard", "hard", "medium", "easy", "easy"], 2000),
    (108, 5, ["hard", "medium", "medium", "easy", "easy"], 2200),
    (109, 7, ["hard", "medium", "medium", "easy", "easy", "easy"], 2400),         # held-out, larger
    (110, 6, ["hard", "hard", "medium", "medium", "easy", "easy", "easy", "easy"], 2400),  # held-out, F>K
]


def _build_instance(seed, K, tiers, n_benign):
    u = _rng(seed)
    F = len(tiers)
    sig = [f % K for f in range(F)]  # family f's fingerprint sensor (round-robin; reused when F>K)
    A = []; dsig = []
    for f in range(F):
        lo_a, hi_a, lo_d, hi_d = TIERS[tiers[f]]
        A.append(lo_a + u() * (hi_a - lo_a))
        dsig.append(lo_d + u() * (hi_d - lo_d))
    benign = [[SCALE * (u() ** GAMMA) for _ in range(K)] for _ in range(n_benign)]
    attacks = []
    for f in range(F):
        levels = []
        for l in range(L_LEVELS):
            samples = []
            for _ in range(M_SAMPLES):
                vec = []
                for c in range(K):
                    base = SCALE * (u() ** GAMMA)
                    if c == sig[f]:
                        decay = max(0.0, 1.0 - l * dsig[f])
                        vec.append(base + A[f] * decay)
                    else:
                        vec.append(base)
                samples.append(vec)
            levels.append(samples)
        attacks.append(levels)
    return {"name": f"soc{seed}", "channels": K, "families": F, "levels": L_LEVELS,
            "n_benign": n_benign, "fp_cap": FP_CAP, "benign": benign, "attacks": attacks,
            "family_names": FAMILY_NAMES[:F], "channel_names": CHANNEL_NAMES[:K]}


def _quantile_theta(vals, p, n):
    """Smallest-count threshold theta such that at most floor(p*n) of vals exceed it."""
    srt = sorted(vals)
    k = int(p * n)
    if k <= 0:
        return srt[-1] + 1.0
    if k >= n:
        return srt[0] - 1.0
    return srt[n - 1 - k]


def _eval_config(inst, theta, w, tau):
    """Exact (union) evaluation of a full sensor-fusion config against the
    FULL instance data. Returns (fp_rate, qmin, qfs)."""
    K = inst["channels"]; benign = inst["benign"]; attacks = inst["attacks"]
    F = inst["families"]; L = inst["levels"]; n = inst["n_benign"]
    fp = 0
    for row in benign:
        s = 0.0
        for c in range(K):
            if row[c] > theta[c]:
                s += w[c]
        if s >= tau:
            fp += 1
    fp_rate = fp / n
    qfs = []
    for f in range(F):
        num = 0.0; den = 0.0
        for l in range(L):
            samples = attacks[f][l]
            cnt = 0
            for vec in samples:
                s = 0.0
                for c in range(K):
                    if vec[c] > theta[c]:
                        s += w[c]
                if s >= tau:
                    cnt += 1
            det = cnt / len(samples)
            weight = l + 1
            num += weight * det; den += weight
        qfs.append(num / den if den > 0 else 0.0)
    qmin = min(qfs) if qfs else 0.0
    return fp_rate, qmin, qfs


def _q_base(inst):
    """Weak reference: sensor 0 only, thresholded at the full 1% allowance."""
    K = inst["channels"]; benign = inst["benign"]; n = inst["n_benign"]; fp_cap = inst["fp_cap"]
    col0 = [row[0] for row in benign]
    theta0 = _quantile_theta(col0, fp_cap, n)
    theta = [theta0] + [1e9] * (K - 1)
    w = [1.0] + [0.0] * (K - 1)
    _, qmin, _ = _eval_config(inst, theta, w, 1.0)
    return qmin


def _q_ub(inst):
    """Generous, unreachable ceiling: EVERY sensor gets its own full private 1%
    allowance simultaneously (ignores that the allowance is shared). No config
    that genuinely respects one shared 1% cap can score above this, since
    thresholds here are at least as lenient as any shared-budget alternative."""
    K = inst["channels"]; benign = inst["benign"]; n = inst["n_benign"]; fp_cap = inst["fp_cap"]
    theta = []
    for c in range(K):
        col = [row[c] for row in benign]
        theta.append(_quantile_theta(col, fp_cap, n))
    w = [1.0] * K
    _, qmin, _ = _eval_config(inst, theta, w, 1.0)
    return qmin


def _safe_float(x):
    """Convert an int/float to a Python float, rejecting anything that isn't
    finite -- including JSON integers too large to represent as a float
    (float() raises OverflowError on those rather than returning inf)."""
    try:
        v = float(x)
    except (OverflowError, ValueError):
        return None
    if not (v == v) or v in (float("inf"), float("-inf")):
        return None
    return v


def _validate_answer(K, answer):
    if not isinstance(answer, dict):
        return None
    theta = answer.get("theta"); w = answer.get("w"); tau = answer.get("tau")
    if not isinstance(theta, list) or len(theta) != K:
        return None
    if not isinstance(w, list) or len(w) != K:
        return None
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        return None
    tau = _safe_float(tau)
    if tau is None:
        return None
    th = []
    for x in theta:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        x = _safe_float(x)
        if x is None:
            return None
        th.append(x)
    ws = []
    for x in w:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        x = _safe_float(x)
        if x is None or x < 0.0:
            return None
        ws.append(x)
    return th, ws, tau


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]

    vec = []
    for seed, K, tiers, n_benign in SPECS:
        inst = _build_instance(seed, K, tiers, n_benign)
        q_base = _q_base(inst)
        q_ub = _q_ub(inst)
        denom = q_ub - q_base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "channels": inst["channels"], "fp_cap": inst["fp_cap"],
                  "n_benign": inst["n_benign"], "benign": inst["benign"],
                  "families": inst["families"], "levels": inst["levels"],
                  "family_names": inst["family_names"], "channel_names": inst["channel_names"],
                  "attacks": inst["attacks"]}

        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0); continue

        try:
            parsed = _validate_answer(inst["channels"], ans)
        except Exception:
            parsed = None
        if parsed is None:
            vec.append(0.0); continue
        theta, w, tau = parsed

        try:
            fp_rate, qmin, _ = _eval_config(inst, theta, w, tau)
        except Exception:
            vec.append(0.0); continue

        if fp_rate > inst["fp_cap"] + 1e-9:
            vec.append(0.0); continue

        r = 0.1 + 0.9 * (qmin - q_base) / denom
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
