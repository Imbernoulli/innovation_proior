#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1118 -- "The Unseen Tide: Cargo Admission Policy"
(family: resource-augmented-online-pack; format B, quality-metric).

THEME.  A freight exchange loads cargo lots onto one barge of fixed capacity C.  Lots
arrive one at a time and each admission decision is IRREVOCABLE (accept or reject on
sight, no undo).  Before today's tide, the dock master is shown a PREVIEW manifest: a
similar, but NOT identical, historical arrival log drawn from the same generative
process (same size range, same value-density law, same rough oversubscription regime).
Today's ACTUAL arrivals (the "replay" stream) are held back -- the dock master commits
to an admission POLICY before the first real lot shows up, and that policy is then
applied, causally, lot by lot, to the unseen real stream.

MECHANISM COMPOSITION.
  * online-packing-policy:   admission is causal / irrevocable, replayed lot-by-lot by
    this evaluator (never by the candidate) over a stream the candidate never sees.
  * dual-fitting-threshold:  the policy is a THRESHOLD on value-density, and the
    correct threshold is a *shadow price on remaining capacity* -- it must climb as
    capacity gets scarce, not sit at one static density cutoff.
  * value-drift-adaptation:  the policy also reacts to the OBSERVED average density of
    lots seen so far, because the preview's implied "how busy will today be" guess can
    be wrong (today may be a surge or a lull relative to the historical log).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int),
             "preview_n": int,
             "preview": [[size_0, value_0], ..., [size_{m-1}, value_{m-1}]]}
          `preview` is a historical manifest from the SAME distribution as today's
          actual arrivals, but is NOT today's stream -- sizes/values are integers,
          1 <= size <= capacity, value >= 1.
  stdout: ONE JSON object:
            {"policy": {"base": float, "cap_gain": float,
                        "drift_gain": float, "time_gain": float}}
          Four finite real numbers.  The candidate does NOT output per-lot decisions;
          it outputs the four coefficients of a fixed admission-threshold formula.

REPLAY (this evaluator; deterministic; the candidate never runs this code or sees its
inputs).  Today's TRUE stream (a fresh draw from the same law, usually a different
length / total volume than the preview) is processed lot by lot, i in 0..N-1, tracking
remaining capacity `rem` (starts at C) and the mean density of lots SEEN SO FAR:

    density_i   = value_i / size_i
    time_frac   = i / (N - 1)                       # 0 at first lot, 1 at last
    cap_used    = 1 - rem / C                        # fraction of capacity spent so far
    running_avg = mean(density_0 .. density_{i-1})   # 0 before the first lot
    threshold_i = base
                + cap_gain   * cap_used ** 2          # shadow price: quadratic ramp
                + drift_gain * running_avg            # reacts to observed density level
                + time_gain  * time_frac              # anticipates a known drift trend

    ADMIT lot i iff density_i >= threshold_i AND size_i <= rem.
    If admitted: rem -= size_i, total_value += value_i.

A STATIC density cutoff (cap_gain = drift_gain = time_gain = 0, `base` fixed from the
preview) is the "obvious" online-knapsack recipe.  It is calibrated for the load the
preview implied and has no way to notice, mid-stream, that today diverged from that
guess -- it either strands capacity (today is quieter than expected) or squanders it on
mediocre early lots (today is busier, or trending up, than expected).  A policy whose
`cap_gain`/`drift_gain`/`time_gain` make the bar climb as capacity actually depletes and
as the observed density level shifts recovers much more of the achievable value.

SCORING (deterministic; no wall-time).  Per instance, this evaluator computes on the
TRUE (hidden) stream:
    v_base = value collected by "admit every lot that still fits" (no filtering at all)
    v_hi   = the fractional-knapsack (LP-relaxation) upper bound: sort lots by density
             descending and fill C, allowing the boundary lot to be taken fractionally.
             v_hi is an UPPER BOUND on ANY feasible 0/1 selection, causal or not, so no
             submitted policy can ever fully reach it -- built-in headroom.
    v_cand = value collected by REPLAYing the candidate's policy over the true stream.
and normalizes with an affine anchor (weak "accept everything" -> 0.1, the loose
fractional ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (v_cand - v_base) / max(1e-9, v_hi - v_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees `preview`, never the true stream, never the
references above.  Those are computed entirely in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
MASK64 = (1 << 64) - 1


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & MASK64

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
        return state

    def nxt_int(lo, hi):
        return lo + (nxt() >> 17) % (hi - lo + 1)

    def nxt_float():
        return (nxt() >> 11) / (1 << 53)

    return nxt_int, nxt_float


# ----------------------------- stream generation ----------------------------
def _build_stream(seed, n, size_lo, size_hi, mu, noise_frac, mode,
                   start_mult=1.0, end_mult=1.0, spike_frac=0.15, spike_mult=3.0):
    """Deterministic list of (size, value) integer pairs, arrival order matters."""
    nxt_int, nxt_float = _rng(seed)
    items = []
    for i in range(n):
        t = i / max(1, n - 1)
        noise = (nxt_float() - 0.5) * 2 * noise_frac
        if mode == "flat":
            dens = mu * (1 + noise)
        elif mode in ("drift_up", "drift_down"):
            mult = start_mult + (end_mult - start_mult) * t
            dens = mu * mult * (1 + noise)
        elif mode == "late_spike":
            if t >= 1 - spike_frac:
                dens = mu * spike_mult * (1 + noise * 0.5)
            else:
                dens = mu * (1 + noise)
        else:
            dens = mu * (1 + noise)
        dens = max(0.5, dens)
        size = nxt_int(size_lo, size_hi)
        value = max(1, round(dens * size))
        items.append((size, value))
    return items


# ----------------------------- instance family -----------------------------
# (id, mode, mu, noise, size_lo, size_hi, n_preview, oversub_preview,
#  oversub_actual_target, start_mult, end_mult, spike_frac, spike_mult)
_SPECS = [
    dict(id=1, mode="flat", mu=12.0, noise=0.45, size_lo=5, size_hi=30,
         n_prev=70, oversub_prev=2.2, oversub_act=3.6),
    dict(id=2, mode="flat", mu=10.0, noise=0.45, size_lo=5, size_hi=30,
         n_prev=70, oversub_prev=2.0, oversub_act=5.0),
    dict(id=3, mode="flat", mu=11.0, noise=0.5, size_lo=5, size_hi=30,
         n_prev=80, oversub_prev=2.0, oversub_act=3.0),
    dict(id=4, mode="drift_up", mu=9.0, noise=0.3, size_lo=5, size_hi=30,
         n_prev=70, oversub_prev=2.0, oversub_act=3.4, start_mult=0.6, end_mult=2.2),
    dict(id=5, mode="drift_up", mu=8.0, noise=0.3, size_lo=5, size_hi=35,
         n_prev=90, oversub_prev=2.2, oversub_act=4.0, start_mult=0.5, end_mult=3.0),
    dict(id=6, mode="drift_down", mu=10.0, noise=0.3, size_lo=5, size_hi=30,
         n_prev=70, oversub_prev=2.0, oversub_act=3.2, start_mult=2.0, end_mult=0.6),
    dict(id=7, mode="drift_up", mu=8.0, noise=0.3, size_lo=5, size_hi=30,
         n_prev=80, oversub_prev=2.2, oversub_act=6.0, start_mult=0.7, end_mult=1.8),
    dict(id=8, mode="flat", mu=12.0, noise=0.55, size_lo=5, size_hi=30,
         n_prev=70, oversub_prev=2.2, oversub_act=3.5),
    dict(id=9, mode="late_spike", mu=9.0, noise=0.3, size_lo=5, size_hi=30,
         n_prev=90, oversub_prev=2.2, oversub_act=4.0, spike_frac=0.15, spike_mult=3.2),
    dict(id=10, mode="drift_down", mu=9.0, noise=0.3, size_lo=5, size_hi=40,
         n_prev=100, oversub_prev=2.2, oversub_act=3.9, start_mult=1.8, end_mult=0.6),
]


def _build_instances():
    out = []
    for sp in _SPECS:
        seed_prev = 1000 + sp["id"] * 7
        seed_rep = 5000 + sp["id"] * 13
        preview = _build_stream(seed_prev, sp["n_prev"], sp["size_lo"], sp["size_hi"],
                                 sp["mu"], sp["noise"], sp["mode"],
                                 sp.get("start_mult", 1.0), sp.get("end_mult", 1.0),
                                 sp.get("spike_frac", 0.15), sp.get("spike_mult", 3.0))
        total_prev = sum(s for s, v in preview)
        capacity = max(1, round(total_prev / sp["oversub_prev"]))
        mean_size = (sp["size_lo"] + sp["size_hi"]) / 2.0
        target_total = capacity * sp["oversub_act"]
        n_rep = max(10, round(target_total / mean_size))
        replay_items = _build_stream(seed_rep, n_rep, sp["size_lo"], sp["size_hi"],
                                      sp["mu"], sp["noise"], sp["mode"],
                                      sp.get("start_mult", 1.0), sp.get("end_mult", 1.0),
                                      sp.get("spike_frac", 0.15), sp.get("spike_mult", 3.0))
        out.append({
            "name": f"tide{sp['id']:02d}",
            "capacity": capacity,
            "preview": preview,
            "replay": replay_items,
        })
    return out


# ----------------------------- references ----------------------------------
def _v_base(items, capacity):
    """Weak baseline: admit every lot that still fits, arrival order, no filtering."""
    rem = capacity
    v = 0
    for size, value in items:
        if size <= rem:
            rem -= size
            v += value
    return v


def _v_hi(items, capacity):
    """Fractional-knapsack (LP relaxation) upper bound: unreachable-by-0/1 ideal."""
    order = sorted(range(len(items)), key=lambda i: items[i][1] / items[i][0], reverse=True)
    rem = capacity
    v = 0.0
    for i in order:
        size, value = items[i]
        if size <= rem:
            rem -= size
            v += value
        else:
            if rem > 0:
                v += value * (rem / size)
            break
    return v


CAP_POWER = 2.0


def _replay_policy(items, capacity, policy):
    """Deterministically replay a 4-coefficient threshold policy over `items`."""
    base = policy["base"]
    cap_gain = policy["cap_gain"]
    drift_gain = policy["drift_gain"]
    time_gain = policy["time_gain"]
    N = len(items)
    rem = capacity
    seen_sum = 0.0
    seen_n = 0
    total = 0
    for i, (size, value) in enumerate(items):
        density = value / size
        time_frac = i / max(1, N - 1)
        cap_used = 1.0 - rem / capacity
        running_avg = (seen_sum / seen_n) if seen_n > 0 else 0.0
        threshold = (base + cap_gain * (cap_used ** CAP_POWER)
                     + drift_gain * running_avg + time_gain * time_frac)
        if density >= threshold and size <= rem:
            rem -= size
            total += value
        seen_sum += density
        seen_n += 1
    return total


# ----------------------------- validation -----------------------------------
def _extract_policy(answer):
    if not isinstance(answer, dict):
        return None
    policy = answer.get("policy")
    if not isinstance(policy, dict):
        return None
    out = {}
    for key in ("base", "cap_gain", "drift_gain", "time_gain"):
        if key not in policy:
            return None
        v = policy[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if abs(v) > 1e6:
            return None
        out[key] = v
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        capacity = inst["capacity"]
        replay_items = inst["replay"]
        v_base = _v_base(replay_items, capacity)
        v_hi = _v_hi(replay_items, capacity)
        denom = v_hi - v_base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "capacity": capacity,
                  "preview_n": len(inst["preview"]),
                  "preview": [list(p) for p in inst["preview"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            policy = _extract_policy(ans)
        except Exception:
            policy = None
        if policy is None:
            vec.append(0.0)
            continue
        try:
            v_cand = _replay_policy(replay_items, capacity, policy)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (v_cand - v_base) / denom
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
