#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1310 -- "Pointing a Telescope That Cannot Be Everywhere"
(family: satellite-tasking-policy; format B, quality-metric).

THEME.  An Earth-observation satellite crosses a strip of ground targets during a
mission made of TWO revisit opportunities ("passes") over the same strip: an early
pass and a later pass, separated by an orbital gap. Before a pass, the telescope
sits at pointing position 0. Slewing from position `a` to position `b` costs
`|a-b| * slew_rate + settle` seconds (SLEW-TIME-COST). Imaging a target then costs
its `dwell` seconds. A pass has a hard time budget; once the running clock would
exceed it, that pass stops taking new images.

Each target's collected value decays with the ABSOLUTE mission clock at the moment
it is imaged: `value * max(0, 1 - decay_rate * mission_time)` (OBSERVATION-DECAY-
VALUE) -- whatever the target is showing (fire front, flood extent, snow line...)
is evolving, so a late image is worth less than an early one of the same target.
Mission time in the first pass is just the pass-1 clock; in the second pass it is
`pass_gap + pass-2 clock`, so second-pass images are systematically discounted.

Every target also carries a per-pass CLOUD FORECAST: `cloud_forecast_p1/p2` is the
probability the target is cloud-covered during that pass (CLOUD-COVER-UNCERTAINTY).
The candidate sees only this forecast. The actual cloud state realized for each
pass is hidden ground truth: if a target is actually clouded during the pass in
which it is imaged, that observation returns ZERO value (the slew/dwell time is
still spent -- burned for nothing). A target can be imaged in AT MOST one pass
total (once collected -- or wasted on cloud -- there is no third look).

The obvious strategy -- image the highest-value target next -- ignores both how
expensive it is to reach (a rich but far-off target burns the whole pass slewing to
it, crowding out several closer, still-valuable targets) and whether it is likely
to be clouded out (a rich target with a high pass-1 cloud forecast and a much lower
pass-2 forecast is worth imaging LATER despite the decay penalty, not gambled on
now). The insight the strong tier exploits: route by VALUE DENSITY (expected value
per slew-second, not raw value) within each pass, and DEFER targets whose forecast
says pass-2 is much safer than pass-1 even though a deferred image is worth less.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "slew_rate": float, "settle": float,
             "pass1_budget": float, "pass2_budget": float, "pass_gap": float,
             "targets": [
                {"id": int, "x": float, "value": int, "dwell": float,
                 "decay_rate": float,
                 "cloud_forecast_p1": float, "cloud_forecast_p2": float},
                ...
             ]}
  stdout: ONE JSON object:
            {"pass1": [id, id, ...], "pass2": [id, id, ...]}
          Each list is a VISITING ORDER (satellite images them in that order,
          starting from pointing position 0 at the start of that pass). A target id
          may appear in AT MOST one of the two lists (it can be attempted only
          once, in only one pass); lists may be any length (including empty) and
          may name targets that end up not fitting the time budget -- those are
          simply not imaged (this is not a validity error).

  A plan is VALID iff `pass1`/`pass2` are lists of integers, every id refers to an
  existing target, and no id appears twice (within a list or across both lists). A
  crash, timeout, non-JSON output, or any of the above violations makes the
  instance score 0.0.

SCORING (deterministic; no wall-time).  For a valid plan, each pass is simulated in
list order from pointing position 0 with a running clock: slewing + settling to the
next target, then dwelling to image it; a target that would push the clock past the
pass's budget (slew+settle+dwell included) is skipped, and the pass simply moves on
to the next id in the list without spending that time. An imaged target contributes
`0` if the HIDDEN actual cloud state for that pass says it is clouded, else the
decayed value at the mission-time it was imaged. `y_cand` is the sum over both
passes.

Per instance the evaluator also computes, itself, with full information:
    y_base = the value-SORTED (desc), pass-1-ONLY, first-fit-in-that-order plan --
             i.e. exactly "image the highest-value target next", ignoring slew
             cost, cloud forecast, decay, and the second pass entirely. This is the
             weak reference the innovation hook names, and the normalization
             anchor (scores ~0.1).
    y_ub   = sum of every target's raw `value`, ignoring cost/decay/cloud -- a
             loose, generally unreachable upper bound, so strong siters have
             headroom below 1.0.
and normalizes with the same affine anchor used across this corpus:
    r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC view (forecasts, never the
hidden actual cloud state). All references and validation happen in THIS parent
process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed & ((1 << 64) - 1)]

    def _raw():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state[0]

    def nxt_int(lo, hi):
        return lo + (_raw() >> 17) % (hi - lo + 1)

    def nxt_float(lo, hi):
        v = (_raw() >> 11) / float(1 << 53)  # in [0, 1)
        return lo + v * (hi - lo)

    return nxt_int, nxt_float


SLEW_RATE = 1.0
SETTLE = 3.0


def _random_targets(seed, n, x_max, val_lo, val_hi):
    ni, nf = _rng(seed)
    out = []
    for _ in range(n):
        x = nf(0.0, x_max)
        value = ni(val_lo, val_hi)
        dwell = nf(3.0, 9.0)
        decay_rate = nf(0.0003, 0.0009)
        p1 = nf(0.05, 0.38)
        p2 = max(0.03, p1 - nf(0.05, 0.25))
        a1 = nf(0.0, 1.0) < p1   # hidden actual cloud state, correlated with forecast
        a2 = nf(0.0, 1.0) < p2
        out.append(dict(x=round(x, 2), value=value, dwell=round(dwell, 2),
                         decay_rate=round(decay_rate, 6),
                         cloud_forecast_p1=round(p1, 3), cloud_forecast_p2=round(p2, 3),
                         cloud_actual_p1=bool(a1), cloud_actual_p2=bool(a2)))
    return out


def _forced_target(kind, x, value, dwell, decay_rate):
    """Hand-planted trap targets: their hidden truth is FIXED (not drawn), so the
    trap is guaranteed to fire every run, independent of the RNG stream."""
    if kind == "far_decoy":
        # rich but expensive to reach; cheap-to-check, never clouded -- pure
        # slew-cost trap: value-first tasking burns the pass slewing to it.
        return dict(x=x, value=value, dwell=dwell, decay_rate=decay_rate,
                    cloud_forecast_p1=0.05, cloud_forecast_p2=0.05,
                    cloud_actual_p1=False, cloud_actual_p2=False)
    if kind == "cloud_trap":
        # rich, cheap to reach, high pass-1 cloud risk that DOES realize, low
        # pass-2 risk that does NOT -- pure deferral trap.
        return dict(x=x, value=value, dwell=dwell, decay_rate=decay_rate,
                    cloud_forecast_p1=0.88, cloud_forecast_p2=0.10,
                    cloud_actual_p1=True, cloud_actual_p2=False)
    raise ValueError(kind)


def _make_instance(name, seed, n, x_max, val_lo, val_hi,
                    pass1_budget, pass2_budget, pass_gap, forced=()):
    targets = []
    for spec in forced:
        targets.append(_forced_target(*spec))
    targets.extend(_random_targets(seed, n, x_max, val_lo, val_hi))
    for i, t in enumerate(targets):
        t["id"] = i
    return dict(name=name, slew_rate=SLEW_RATE, settle=SETTLE,
                pass1_budget=float(pass1_budget), pass2_budget=float(pass2_budget),
                pass_gap=float(pass_gap), targets=targets)


def _build_instances():
    """10 instances: 4 plain (RNG only) + 6 with hand-planted traps (slew-cost,
    cloud-deferral, or both compounded), including 2 larger held-out ones."""
    specs = [
        ("orbit01", 1001, 10, 260, 40, 160, 140, 140, 380, ()),
        ("orbit02", 1002, 12, 300, 40, 170, 160, 150, 400, ()),
        ("orbit03_slewtrap", 1003, 9, 260, 40, 150, 140, 140, 380,
         (("far_decoy", 258, 260, 6.0, 0.0004),)),
        ("orbit04_cloudtrap", 1004, 9, 260, 40, 150, 140, 140, 380,
         (("cloud_trap", 40, 240, 6.0, 0.0004),)),
        ("orbit05_compound", 1005, 9, 280, 40, 150, 150, 140, 380,
         (("far_decoy", 276, 260, 6.0, 0.0004), ("cloud_trap", 30, 230, 6.0, 0.0004))),
        ("orbit06", 1006, 13, 320, 40, 170, 170, 160, 420, ()),
        ("orbit07_doublecloud", 1007, 11, 300, 40, 160, 160, 150, 400,
         (("cloud_trap", 25, 220, 6.0, 0.0004), ("cloud_trap", 55, 200, 6.0, 0.0004),
          ("far_decoy", 296, 250, 6.0, 0.0004))),
        ("orbit08", 1008, 15, 340, 40, 170, 190, 170, 420, ()),
        ("orbit09_held_compound", 2001, 16, 380, 40, 190, 200, 180, 450,
         (("far_decoy", 372, 300, 6.0, 0.0003), ("cloud_trap", 25, 260, 6.0, 0.0003))),
        ("orbit10_held_doublecloud", 2002, 14, 360, 40, 180, 190, 170, 440,
         (("cloud_trap", 20, 250, 6.0, 0.0003), ("cloud_trap", 45, 230, 6.0, 0.0003))),
    ]
    return [_make_instance(*s[:9], forced=s[9]) for s in specs]


# ----------------------------- simulation -----------------------------------
def _value_at(target, mission_time):
    v = target["value"] * (1.0 - target["decay_rate"] * mission_time)
    return v if v > 0.0 else 0.0


def _run_pass(tmap, ids, budget, gap_offset, cloud_key):
    """Simulate one pass in list order from pointing position 0. Returns
    (value_collected, set_of_ids_used)."""
    pos = 0.0
    elapsed = 0.0
    used = set()
    total = 0.0
    for tid in ids:
        t = tmap[tid]
        slew = abs(t["x"] - pos) * SLEW_RATE + SETTLE
        finish = elapsed + slew + t["dwell"]
        if finish > budget:
            continue  # doesn't fit in remaining time -- skip, keep clock as-is
        elapsed = finish
        pos = t["x"]
        used.add(tid)
        mission_time = gap_offset + elapsed
        if t[cloud_key]:
            continue  # clouded out: time spent, zero value
        total += _value_at(t, mission_time)
    return total, used


def _simulate(inst, pass1_ids, pass2_ids):
    tmap = {t["id"]: t for t in inst["targets"]}
    v1, used1 = _run_pass(tmap, pass1_ids, inst["pass1_budget"], 0.0, "cloud_actual_p1")
    remaining2 = [tid for tid in pass2_ids if tid not in used1]
    v2, _ = _run_pass(tmap, remaining2, inst["pass2_budget"], inst["pass_gap"], "cloud_actual_p2")
    return v1 + v2


def _naive_value_order_baseline(inst):
    """The evaluator's own weak reference: sort by raw value desc, pass-1 ONLY,
    first-fit in that order -- exactly "image the highest-value target next"."""
    order = sorted(inst["targets"], key=lambda t: (-t["value"], t["id"]))
    ids = [t["id"] for t in order]
    return _simulate(inst, ids, [])


def baseline(inst):
    return _naive_value_order_baseline(inst)


def _upper_bound(inst):
    return float(sum(t["value"] for t in inst["targets"]))


# ----------------------------- answer validation -----------------------------
def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    p1 = answer.get("pass1")
    p2 = answer.get("pass2")
    if not isinstance(p1, list) or not isinstance(p2, list):
        return None
    n = len(inst["targets"])
    seen = set()
    out = {}
    for key, lst in (("pass1", p1), ("pass2", p2)):
        ids = []
        for v in lst:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            if v < 0 or v >= n:
                return None
            if v in seen:
                return None
            seen.add(v)
            ids.append(v)
        out[key] = ids
    return out


def score(inst, answer):
    plan = _validate(inst, answer)
    if plan is None:
        return False, 0.0
    y = _simulate(inst, plan["pass1"], plan["pass2"])
    if not (y == y) or y in (float("inf"), float("-inf")):
        return False, 0.0
    return True, y


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        y_base = baseline(inst)
        y_ub = _upper_bound(inst)
        denom = max(1e-9, y_ub - y_base)
        public = {"name": inst["name"], "slew_rate": inst["slew_rate"],
                  "settle": inst["settle"], "pass1_budget": inst["pass1_budget"],
                  "pass2_budget": inst["pass2_budget"], "pass_gap": inst["pass_gap"],
                  "targets": [{"id": t["id"], "x": t["x"], "value": t["value"],
                               "dwell": t["dwell"], "decay_rate": t["decay_rate"],
                               "cloud_forecast_p1": t["cloud_forecast_p1"],
                               "cloud_forecast_p2": t["cloud_forecast_p2"]}
                              for t in inst["targets"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, y_cand = score(inst, ans)
        except Exception:
            ok, y_cand = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (y_cand - y_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
