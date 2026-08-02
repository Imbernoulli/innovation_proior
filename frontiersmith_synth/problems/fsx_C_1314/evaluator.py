#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1314 -- "Ensemble Cue Policy Through a Rubato"
(family: music-ensemble-cue-policy; eval_form: quality-metric; wave3-lens:policy-simulator shape).

A small chamber ensemble reads through a passage of T beats. At the start of
every beat the CONDUCTOR (the candidate) broadcasts one tempo-multiplier
"cue" to the whole ensemble. Each player has their own reaction LATENCY
(beats before a cue reaches their bow/breath) and INERTIA (how much of the
gap to the cue they close per beat) -- known, public numbers, since they come
from rehearsal. Because the conductor only gets to look at the score and a
noisy first read-through BEFORE the performance, the candidate submits its
ENTIRE cue schedule for the passage up front (a conductor who has studied the
whole piece knows the whole phrase shape in advance); the evaluator then
plays that schedule against each player's OWN response dynamics and scores
the result -- run as an ISOLATED subprocess (isorun) so it never sees hidden
ground truth.

Public instance JSON (what the candidate reads on stdin):
    {
      "n_players": int,
      "T":         int,                  # number of beats
      "role_weight": [[float,...]*n]*T,  # per-beat score annotation: who is
                                          # currently carrying the melodic
                                          # phrase (sums to 1 each beat)
      "latency":   [int,...],            # per-player cue reaction delay (beats)
      "inertia":   [float,...],          # per-player per-beat closing fraction
      "observed":  [[float,...]*n]*T,    # noisy first read-through: each
                                          # player's own inclination tempo at
                                          # every beat, with NO conductor cueing
      "seed": int
    }

Answer JSON (what the candidate writes on stdout):
    [float, float, ..., float]   # length T; cue[t] = tempo multiplier
                                  # broadcast to the whole ensemble at beat t

Quality is measured by re-simulating the ensemble's ACTUAL per-player tempo
trajectory under the candidate's cue schedule (same latency/inertia as given)
and comparing the phrase-weighted ensemble tempo against the HIDDEN true
phrase tempo curve (tracking error) plus how tightly the players stay
together, weighted by who currently carries the phrase (tightness error).
Passages come in two flavors: metronomic WARM-UPS (near-constant tempo,
where simple mean-tracking is basically fine) and RUBATO passages (real
tempo swells) where a policy that (a) treats every player's noisy reading
equally instead of trusting the phrase-carrier's channel, and (b) reacts to
the cue it should be broadcasting NOW instead of anticipating the reaction
latency, both damps the swell and arrives late.

Per-instance normalization is an affine anchor against the evaluator's own
internal baseline (the "never move" flat cue == constant 1.0 for the whole
passage):

    r = clamp( 0.1 + 0.9 * (err_base - err_cand) / max(err_base, FLOOR), 0, 1 )

so a candidate that reproduces the flat baseline maps to ~0.1 and reducing
combined error to (near) zero maps toward 1.0. An instance where the
candidate raises, returns the wrong shape/length, emits a non-finite or
out-of-range cue, or times out scores exactly 0.0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

CAND_TIMEOUT = 20
CUE_LO, CUE_HI = 0.1, 4.0
DENOM_FLOOR = 1e-4
W_TEMPO = 1.0        # weight on phrase-tracking error
W_TIGHT = 0.3        # weight on ensemble-tightness error

SIGMA_LEAD = 0.010          # noise on the phrase-carrier's own reading
FOLLOWER_REVERT = 0.9       # mean-reversion of a non-carrying player's inner pulse
FOLLOWER_PROCESS_NOISE = 0.010
FOLLOWER_OBS_NOISE = 0.030
FOLLOWER_TRACK_FRAC = 0.35  # how much of the true phrase a non-carrier partially senses
RAMP = 3                    # beats over which a phrase handoff ramps


# ============================ role-weight schedule ==========================
def _role_schedule(rng, n, T):
    """Partition the passage into n segments (one per player, shuffled order);
    each segment's lead gets most of the role weight, ramped smoothly across
    handoffs. Returns (role_weight: T x n array, lead_idx: length-T int array)."""
    order = rng.permutation(n)
    seg_len = T / float(n)
    bounds = [int(round(k * seg_len)) for k in range(n + 1)]
    bounds[0], bounds[-1] = 0, T

    def vec_for(lead):
        v = np.full(n, 0.35 / max(n - 1, 1))
        v[lead] = 0.65
        return v

    seg_of_t = np.zeros(T, dtype=np.int64)
    for k in range(n):
        seg_of_t[bounds[k]:bounds[k + 1]] = k

    role = np.zeros((T, n))
    lead_idx = np.zeros(T, dtype=np.int64)
    for t in range(T):
        k = seg_of_t[t]
        cur_vec = vec_for(order[k])
        # ramp into this segment from the previous one over RAMP beats
        dist_in = t - bounds[k]
        if k > 0 and dist_in < RAMP:
            prev_vec = vec_for(order[k - 1])
            frac = (dist_in + 1) / float(RAMP + 1)
            v = (1 - frac) * prev_vec + frac * cur_vec
        else:
            v = cur_vec
        v = v / v.sum()
        role[t] = v
        lead_idx[t] = int(np.argmax(v))
    return role, lead_idx


# ============================ true tempo curves ==============================
def _true_tempo_warmup(rng, T):
    amp = rng.uniform(0.02, 0.04)
    phase = rng.uniform(0.0, 2 * math.pi)
    t = np.arange(T)
    return 1.0 + amp * np.sin(2 * math.pi * t / T + phase)


def _hann_bump(t, center, width, amp, sign):
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.zeros_like(t, dtype=np.float64)
    mask = (t >= lo) & (t <= hi)
    phase = (t[mask] - lo) / width
    out[mask] = amp * sign * 0.5 * (1 - np.cos(2 * math.pi * phase))
    return out


def _true_tempo_rubato(rng, T, n_swells):
    t = np.arange(T, dtype=np.float64)
    curve = np.ones(T)
    for _ in range(n_swells):
        center = rng.uniform(0.22 * T, 0.78 * T)
        width = rng.uniform(8.0, 16.0)
        amp = rng.uniform(0.18, 0.35)
        sign = float(rng.choice([-1.0, 1.0]))
        curve = curve + _hann_bump(t, center, width, amp, sign)
    curve = curve + 0.01 * np.sin(2 * math.pi * t / max(T, 1) + rng.uniform(0, 2 * math.pi))
    return curve


# ============================ observed sensor generation =====================
def _observed(rng, n, T, true_tempo, lead_idx):
    """Per-beat, per-player noisy self-read. The CURRENT phrase carrier reads
    the true phrase tempo closely (small noise) -- they are playing the
    melodic line and feel the rubato directly. Every other player only
    PARTIALLY senses the phrase (FOLLOWER_TRACK_FRAC) and otherwise reports
    their own steady, mean-reverting inner pulse, plus sensor noise -- they
    are reading accompaniment figures and only loosely track the expressive
    push/pull, so a plain average keeps real (if diluted and noisy) signal
    on EVERY instance, but the cleanest, most amplified read of the phrase
    always lives in the single carrier channel that role_weight identifies."""
    obs = np.zeros((T, n))
    f = np.ones(n)  # each follower's own mean-reverting inner-pulse state
    for t in range(T):
        for i in range(n):
            if i == lead_idx[t]:
                obs[t, i] = true_tempo[t] + rng.normal(0.0, SIGMA_LEAD)
            else:
                f[i] = 1.0 + FOLLOWER_REVERT * (f[i] - 1.0) + rng.normal(0.0, FOLLOWER_PROCESS_NOISE)
                inclination = FOLLOWER_TRACK_FRAC * true_tempo[t] + (1 - FOLLOWER_TRACK_FRAC) * f[i]
                obs[t, i] = inclination + rng.normal(0.0, FOLLOWER_OBS_NOISE)
    return obs


# ============================ instance construction ===========================
def _build_instance(seed, kind, n, T, **kw):
    rng = np.random.default_rng(seed)
    role, lead_idx = _role_schedule(rng, n, T)
    if kind == "warmup":
        true_tempo = _true_tempo_warmup(rng, T)
    else:
        true_tempo = _true_tempo_rubato(rng, T, kw.get("n_swells", 1))
    latency = rng.integers(kw.get("lat_lo", 1), kw.get("lat_hi", 5), size=n).astype(np.int64)
    inertia = rng.uniform(0.15, 0.55, size=n)
    obs = _observed(rng, n, T, true_tempo, lead_idx)
    return {
        "name": f"{kind}{seed}",
        "n": n, "T": T,
        "role_weight": role, "latency": latency, "inertia": inertia,
        "observed": obs, "true_tempo": true_tempo, "seed": seed,
    }


def _build_instances():
    specs = [
        ("warmup", dict(seed=401, n=4, T=32)),
        ("warmup", dict(seed=402, n=5, T=40)),
        ("warmup", dict(seed=403, n=6, T=48)),
        ("rubato", dict(seed=411, n=4, T=36, n_swells=1)),
        ("rubato", dict(seed=412, n=5, T=40, n_swells=1)),
        ("rubato", dict(seed=413, n=5, T=44, n_swells=2)),
        ("rubato", dict(seed=414, n=6, T=48, n_swells=1, lat_lo=1, lat_hi=6)),   # held-out: wider latency spread
        ("rubato", dict(seed=415, n=6, T=52, n_swells=2, lat_lo=2, lat_hi=6)),   # held-out: harder, high latency
        ("rubato", dict(seed=416, n=7, T=50, n_swells=2)),                       # held-out: more players
        ("rubato", dict(seed=417, n=5, T=56, n_swells=2, lat_lo=1, lat_hi=6)),   # held-out: longer passage
    ]
    return [_build_instance(**{**p, "kind": k}) for k, p in specs]


# ============================ dynamics / scoring ==============================
def _simulate(cue, latency, inertia, n, T):
    v = np.ones((T, n))
    for t in range(1, T):
        src = t - latency
        src = np.where(src >= 0, src, 0)
        c_seen = cue[src]
        v[t] = v[t - 1] + inertia * (c_seen - v[t - 1])
    return v


def _combined_error(cue, inst):
    n, T = inst["n"], inst["T"]
    role, latency, inertia = inst["role_weight"], inst["latency"], inst["inertia"]
    true_tempo = inst["true_tempo"]
    v = _simulate(cue, latency, inertia, n, T)
    w_tempo = (role * v).sum(axis=1)
    tempo_err = math.sqrt(float(np.mean((w_tempo[1:] - true_tempo[1:]) ** 2)))
    dev2 = (v - w_tempo[:, None]) ** 2
    var_t = (role * dev2).sum(axis=1)
    tight_err = math.sqrt(float(np.mean(var_t[1:])))
    return W_TEMPO * tempo_err + W_TIGHT * tight_err


def baseline(inst):
    flat = np.ones(inst["T"])
    return _combined_error(flat, inst)


def _valid_cue(ans, T):
    if isinstance(ans, dict):
        ans = ans.get("cue", None)
    if ans is None:
        return None
    try:
        c = np.asarray(ans, dtype=np.float64)
    except Exception:
        return None
    if c.ndim != 1 or c.shape[0] != T:
        return None
    if not np.all(np.isfinite(c)):
        return None
    if np.any(c < CUE_LO) or np.any(c > CUE_HI):
        return None
    return c


def score(inst, answer):
    cue = _valid_cue(answer, inst["T"])
    if cue is None:
        return False, 0.0
    err = _combined_error(cue, inst)
    return True, err


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {
            "n_players": int(inst["n"]),
            "T": int(inst["T"]),
            "role_weight": inst["role_weight"].tolist(),
            "latency": [int(x) for x in inst["latency"]],
            "inertia": [float(x) for x in inst["inertia"]],
            "observed": inst["observed"].tolist(),
            "seed": int(20240000 + inst["seed"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, err_cand = score(inst, ans)
        except Exception:
            ok, err_cand = False, 0.0
        if not ok:
            vec.append(0.0)
            continue

        err_base = baseline(inst)
        denom = max(err_base, DENOM_FLOOR)
        r = 0.1 + 0.9 * (err_base - err_cand) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
