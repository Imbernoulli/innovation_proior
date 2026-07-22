#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0503 -- "WardWatch: Robust Early-Warning Triage Thresholds".

Family: ml-robust-threshold-policy; format B, quality-metric.

The candidate designs a deterministic threshold table for a hospital early-warning
score.  It receives a labeled calibration stream from a source period and an unlabeled
recent target stream that reveals the score/stratum distribution after a shift.  It
must return finite alert thresholds for unit-by-acuity strata.  The evaluator applies
that policy to a hidden deployment stream with labels held only in this parent process.

Scoring is deterministic and CPU-only.  Utility rewards true early alerts, heavily
penalizes missed deteriorations (asymmetric false-negative cost), charges every alert,
adds a quadratic fatigue penalty when a unit-block exceeds its alert budget, and adds a
calibration-bin penalty when alert rates by score decile deviate from the hidden
event-rate-implied target.  The reported Ratio is a bounded maximize-utility affine
normalization:

    r = clamp(0.10 + 0.70 * (u_cand - u_base) / (u_ref - u_base), 0, 1)

where u_base is the fixed conservative threshold 0.50 policy and u_ref is an internal
public-data heuristic (stratum threshold search plus target-score quantile adjustment).
Thus the trivial policy scores about 0.10, the bundled strong policy scores about 0.80,
and better target-robust policies have real headroom.

The candidate is untrusted and is run via isorun.run_candidate.  Only inst["public"] is
sent to the sandbox.  Hidden deployment labels, scoring state, and reference utilities
never leave this process.
"""
import json
import math
import sys

import isorun


_MASK64 = (1 << 64) - 1
UNITS = ("ED", "WARD", "ICU")
ACUITY = ("low", "med", "high")
STRATA = tuple("%s|%s" % (u, a) for u in UNITS for a in ACUITY)
GRID = [round(0.05 + 0.025 * i, 3) for i in range(31)]  # 0.050 .. 0.800
BASELINE_THRESHOLD = 0.50


class Rng:
    """Small deterministic splitmix64 RNG with Box-Muller normals."""
    def __init__(self, seed):
        self.state = seed & _MASK64
        self.spare = None

    def _u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        return z & _MASK64

    def uniform(self):
        return (self._u64() + 0.5) / 18446744073709551616.0

    def normal(self):
        if self.spare is not None:
            v = self.spare
            self.spare = None
            return v
        u1 = self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self.spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)


def _clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p):
    p = _clip(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _choice(rng, weighted):
    total = sum(w for _, w in weighted)
    x = rng.uniform() * total
    acc = 0.0
    for value, weight in weighted:
        acc += weight
        if x <= acc:
            return value
    return weighted[-1][0]


def _scenario_adjust(kind, key):
    unit, acuity = key.split("|")
    risk = 0.0
    score = 0.0
    if kind == "ed_surge":
        if unit == "ED":
            risk += 0.22 if acuity != "high" else 0.38
            score -= 0.10
        if unit == "WARD" and acuity == "low":
            risk -= 0.08
    elif kind == "ward_frailty":
        if unit == "WARD":
            risk += 0.28 if acuity != "low" else 0.14
            score -= 0.08
        if unit == "ICU":
            score += 0.05
    elif kind == "icu_alert_fatigue":
        if unit == "ICU":
            risk += 0.20
            score += 0.12
        if acuity == "low":
            score += 0.05
    elif kind == "night_undertriage":
        if acuity in ("med", "high"):
            risk += 0.20
            score -= 0.12
    elif kind == "score_overcall":
        if acuity == "low":
            score += 0.18
        if acuity == "high":
            risk += 0.12
    elif kind == "mixed":
        if unit == "ED" and acuity == "high":
            risk += 0.32
        if unit == "WARD" and acuity != "low":
            risk += 0.16
        if unit == "ICU":
            score += 0.10
        else:
            score -= 0.05
    return risk, score


def _unit_weights(kind, target):
    if not target:
        return (("ED", 0.46), ("WARD", 0.38), ("ICU", 0.16))
    if kind == "ed_surge":
        return (("ED", 0.60), ("WARD", 0.28), ("ICU", 0.12))
    if kind == "ward_frailty":
        return (("ED", 0.30), ("WARD", 0.54), ("ICU", 0.16))
    if kind == "icu_alert_fatigue":
        return (("ED", 0.34), ("WARD", 0.34), ("ICU", 0.32))
    if kind == "night_undertriage":
        return (("ED", 0.42), ("WARD", 0.42), ("ICU", 0.16))
    if kind == "score_overcall":
        return (("ED", 0.50), ("WARD", 0.34), ("ICU", 0.16))
    return (("ED", 0.43), ("WARD", 0.40), ("ICU", 0.17))


def _acuity_weights(unit, kind, target):
    if unit == "ICU":
        base = [("low", 0.18), ("med", 0.42), ("high", 0.40)]
    elif unit == "ED":
        base = [("low", 0.48), ("med", 0.36), ("high", 0.16)]
    else:
        base = [("low", 0.55), ("med", 0.33), ("high", 0.12)]
    if not target:
        return tuple(base)
    out = []
    for a, w in base:
        if kind in ("ed_surge", "night_undertriage") and a == "high":
            w *= 1.35
        if kind == "ward_frailty" and unit == "WARD" and a != "low":
            w *= 1.30
        if kind == "score_overcall" and a == "low":
            w *= 1.25
        out.append((a, w))
    return tuple(out)


def _make_records(rng, n, kind, target, include_label, block_offset):
    records = []
    unit_eff = {"ED": 0.00, "WARD": 0.16, "ICU": 0.48}
    acuity_eff = {"low": -0.72, "med": 0.14, "high": 0.98}
    age_eff = {"adult": 0.00, "older": 0.38}
    unit_score_bias = {"ED": 0.02, "WARD": -0.03, "ICU": 0.06}
    acuity_score_bias = {"low": 0.06, "med": 0.00, "high": -0.04}
    for i in range(n):
        unit = _choice(rng, _unit_weights(kind, target))
        acuity = _choice(rng, _acuity_weights(unit, kind, target))
        older_p = 0.27 + (0.10 if acuity == "high" else 0.0)
        if target and kind == "ward_frailty" and unit == "WARD":
            older_p += 0.18
        age = "older" if rng.uniform() < older_p else "adult"
        key = "%s|%s" % (unit, acuity)
        block = block_offset + i // 40
        circ = 0.14 if (block % 6) in (4, 5) else 0.0
        risk_shift, score_shift = _scenario_adjust(kind, key) if target else (0.0, 0.0)
        latent = (-3.02 + unit_eff[unit] + acuity_eff[acuity] + age_eff[age] +
                  circ + risk_shift + 0.34 * rng.normal())
        p_true = _sigmoid(latent)
        y = 1 if rng.uniform() < p_true else 0
        slope = 0.90 if target else 0.98
        raw = (-0.06 + slope * _logit(p_true) + unit_score_bias[unit] +
               acuity_score_bias[acuity] + score_shift + 0.48 * rng.normal())
        score = _clip(_sigmoid(raw), 0.001, 0.999)
        rec = {
            "unit": unit,
            "acuity": acuity,
            "age_band": age,
            "score": round(score, 6),
            "score_bin": int(min(9, max(0, int(score * 10.0)))),
            "block": block,
        }
        if include_label:
            rec["label"] = y
        records.append(rec)
    return records


def make_instances():
    specs = [
        (5030101, 560, 180, 620, "ed_surge", 4.2),
        (5030102, 560, 180, 640, "ward_frailty", 4.0),
        (5030103, 540, 170, 620, "icu_alert_fatigue", 3.5),
        (5030104, 540, 170, 620, "night_undertriage", 4.1),
        (5030105, 560, 180, 640, "score_overcall", 3.7),
        (5030106, 580, 190, 660, "night_undertriage", 3.9),
        (5030107, 620, 200, 700, "ed_surge", 3.6),
        (5030108, 620, 200, 700, "ed_surge", 3.7),
        # harder held-out shifts: fewer public target hints and tighter fatigue budgets
        (5030201, 520, 120, 720, "night_undertriage", 3.2),
        (5030202, 520, 120, 720, "icu_alert_fatigue", 3.0),
        (5030203, 500, 110, 740, "mixed", 3.1),
        (5030204, 500, 110, 740, "score_overcall", 3.0),
    ]
    out = []
    for seed, n_cal, n_recent, n_hidden, kind, fatigue_budget in specs:
        rng = Rng(seed)
        calibration = _make_records(rng, n_cal, kind, False, True, 0)
        recent = _make_records(rng, n_recent, kind, True, False, 1000)
        hidden = _make_records(rng, n_hidden, kind, True, True, 2000)
        public = {
            "instance_id": seed,
            "scenario": kind,
            "strata": list(STRATA),
            "threshold_grid": list(GRID),
            "baseline_threshold": BASELINE_THRESHOLD,
            "score_bins": [round(0.1 * i, 1) for i in range(11)],
            "costs": {
                "fn_cost_by_acuity": {"low": 8.0, "med": 12.0, "high": 17.0},
                "alert_cost": 1.0,
                "tp_credit": 2.0,
                "fatigue_weight": 0.42,
                "fatigue_budget_per_40": fatigue_budget,
                "calibration_weight": 0.34,
                "monotone_weight": 2.0,
            },
            "calibration": calibration,
            "recent_unlabeled": recent,
            "note": "Return finite alert thresholds by unit|acuity. Hidden deployment labels are not public.",
        }
        out.append({"public": public, "hidden": {"deployment": hidden}})
    return out


def _is_num(x):
    return (not isinstance(x, bool)) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _validate_policy(public, answer):
    if not isinstance(answer, dict):
        return False, None
    if "default_threshold" not in answer:
        return False, None
    default = answer["default_threshold"]
    if not _is_num(default):
        return False, None
    default = float(default)
    if default < 0.02 or default > 0.98:
        return False, None
    table = answer.get("thresholds", {})
    if not isinstance(table, dict):
        return False, None
    allowed = set(public["strata"])
    clean = {}
    for key, value in table.items():
        if not isinstance(key, str) or key not in allowed:
            return False, None
        if not _is_num(value):
            return False, None
        value = float(value)
        if value < 0.02 or value > 0.98:
            return False, None
        clean[key] = value
    return True, {"default_threshold": default, "thresholds": clean}


def _threshold_for(policy, rec):
    key = "%s|%s" % (rec["unit"], rec["acuity"])
    return policy["thresholds"].get(key, policy["default_threshold"])


def _utility(public, records, policy):
    costs = public["costs"]
    fn_cost = costs["fn_cost_by_acuity"]
    alert_cost = float(costs["alert_cost"])
    tp_credit = float(costs["tp_credit"])
    utility = 0.0
    block_alerts = {}
    block_counts = {}
    bins = {i: {"n": 0, "y": 0, "a": 0, "fn": 0.0} for i in range(10)}
    alerts = []

    for rec in records:
        y = int(rec.get("label", 0))
        alert = 1 if rec["score"] >= _threshold_for(policy, rec) else 0
        alerts.append(alert)
        fc = float(fn_cost[rec["acuity"]])
        if y:
            if alert:
                utility += tp_credit
            else:
                utility -= fc
        elif alert:
            utility -= alert_cost
        if alert:
            utility -= alert_cost
        bk = (rec["unit"], rec["block"])
        block_counts[bk] = block_counts.get(bk, 0) + 1
        block_alerts[bk] = block_alerts.get(bk, 0) + alert
        b = int(rec["score_bin"])
        bins[b]["n"] += 1
        bins[b]["y"] += y
        bins[b]["a"] += alert
        bins[b]["fn"] += fc

    # Alert fatigue: unit-blocks can absorb a few escalations, then handoff quality drops.
    fatigue_budget = float(costs["fatigue_budget_per_40"])
    fatigue_weight = float(costs["fatigue_weight"])
    for bk, cnt in block_counts.items():
        budget = max(1.0, fatigue_budget * cnt / 40.0)
        excess = block_alerts.get(bk, 0) - budget
        if excess > 0.0:
            utility -= fatigue_weight * excess * excess

    # Calibration-bin term: within score deciles, alert rates should roughly track the
    # hidden event-rate-implied treatment load, not just a brittle global cutoff.
    calib_weight = float(costs["calibration_weight"])
    prev_alert_rate = None
    prev_n = 0
    for b in range(10):
        st = bins[b]
        n = st["n"]
        if n < 8:
            continue
        event_rate = st["y"] / n
        alert_rate = st["a"] / n
        mean_fn = st["fn"] / n
        target = (1.15 * event_rate * mean_fn) / (event_rate * mean_fn + 2.5 * alert_cost + 1e-9)
        target = _clip(target, 0.0, 0.92)
        diff = alert_rate - target
        utility -= calib_weight * n * diff * diff
        if prev_alert_rate is not None and alert_rate + 0.06 < prev_alert_rate:
            drop = prev_alert_rate - alert_rate - 0.06
            utility -= float(costs["monotone_weight"]) * min(prev_n, n) * drop * drop
        prev_alert_rate = alert_rate
        prev_n = n

    return utility


def _baseline_policy(public):
    return {"default_threshold": BASELINE_THRESHOLD, "thresholds": {}}


def _quantile_threshold(scores, tail_rate):
    if not scores:
        return None
    s = sorted(scores)
    tail_rate = _clip(tail_rate, 0.02, 0.80)
    idx = int(math.floor((1.0 - tail_rate) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, idx))]


def _patient_proxy(records, threshold, costs):
    fn_cost = costs["fn_cost_by_acuity"]
    alert_cost = float(costs["alert_cost"])
    tp_credit = float(costs["tp_credit"])
    util = 0.0
    alerts = 0
    for rec in records:
        y = int(rec.get("label", 0))
        alert = rec["score"] >= threshold
        if alert:
            alerts += 1
        fc = float(fn_cost[rec["acuity"]])
        if y:
            util += tp_credit if alert else -fc
        elif alert:
            util -= alert_cost
        if alert:
            util -= alert_cost
    rate = alerts / max(1, len(records))
    if rate > 0.28:
        util -= 0.35 * len(records) * (rate - 0.28) * (rate - 0.28)
    return util


def _best_threshold(records, costs, fallback):
    if not records:
        return fallback
    best_t = fallback
    best_u = None
    for t in GRID:
        u = _patient_proxy(records, t, costs)
        if best_u is None or u > best_u + 1e-12:
            best_u = u
            best_t = t
    return best_t


def _fit_reference_policy(public):
    cal = public["calibration"]
    recent = public["recent_unlabeled"]
    costs = public["costs"]
    global_t = _best_threshold(cal, costs, BASELINE_THRESHOLD)
    by_key = {k: [] for k in public["strata"]}
    recent_by_key = {k: [] for k in public["strata"]}
    for rec in cal:
        by_key["%s|%s" % (rec["unit"], rec["acuity"])].append(rec)
    for rec in recent:
        recent_by_key["%s|%s" % (rec["unit"], rec["acuity"])].append(rec["score"])

    table = {}
    all_recent_scores = [rec["score"] for rec in recent]
    all_cal_scores = [rec["score"] for rec in cal]
    for key in public["strata"]:
        rows = by_key[key]
        local = _best_threshold(rows, costs, global_t)
        alpha = min(0.78, max(0.25, len(rows) / 160.0))
        t = alpha * local + (1.0 - alpha) * global_t
        source_scores = [rec["score"] for rec in rows]
        if source_scores:
            source_tail = sum(1 for x in source_scores if x >= t) / len(source_scores)
        else:
            source_tail = sum(1 for x in all_cal_scores if x >= t) / max(1, len(all_cal_scores))
        target_scores = recent_by_key[key] or all_recent_scores
        qt = _quantile_threshold(target_scores, source_tail)
        if qt is not None:
            # Partial alert-load adaptation: use target score distribution without trusting
            # it completely, since a prevalence shift may warrant more alerts.
            t = 0.72 * t + 0.28 * qt
        table[key] = round(_clip(t, 0.04, 0.92), 6)
    return {"default_threshold": round(global_t, 6), "thresholds": table}


def score(inst, answer):
    public = inst["public"]
    ok, policy = _validate_policy(public, answer)
    if not ok:
        return False, None
    return True, _utility(public, inst["hidden"]["deployment"], policy)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    vec = []
    for inst in make_instances():
        public = inst["public"]
        hidden = inst["hidden"]["deployment"]
        u_base = _utility(public, hidden, _baseline_policy(public))
        u_ref = _utility(public, hidden, _fit_reference_policy(public))
        if u_ref <= u_base + 1e-9:
            u_ref = u_base + max(1.0, 0.05 * abs(u_base))
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, u_cand = score(inst, ans)
        except Exception:
            ok, u_cand = False, None
        if not ok or u_cand is None or not math.isfinite(float(u_cand)):
            vec.append(0.0)
            continue
        r = 0.10 + 0.70 * (float(u_cand) - u_base) / (u_ref - u_base)
        if not math.isfinite(r):
            r = 0.0
        r = _clip(r, 0.0, 1.0)
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
