#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0840 -- "Blindfolded Blacksmith: Tuning the Quench"
(family: annealing-schedule-design; format B, quality-metric).

THEME.  A blindfolded blacksmith runs the SAME frozen single-bit-flip Metropolis
searcher over a fixed pseudo-boolean landscape.  She cannot see the landscape and
cannot change the mechanism -- she can only tell the smith HOW HOT TO KEEP THE
FORGE AND WHEN TO PUMP THE BELLOWS AGAIN.  That control policy (temperature
schedule + a reheat/restart rule keyed off the smith's own observed hit-rate) is
the candidate's entire output.

MECHANISM COMPOSITION (why this is not one textbook algorithm in a costume):
  1. frozen-solver-harness -- the move generator (uniform random single-bit flip)
     and the Metropolis accept rule are simulated ENTIRELY by this evaluator; the
     candidate never touches moves, only the temperature-over-time control policy.
  2. temperature-schedule-design -- the candidate sets an initial temperature and
     a per-window geometric decay; the landscape's actual reward-step size per
     flip ("slope") is a HIDDEN per-instance constant never shown to the
     candidate, so a schedule computed only from public, structural fields (block
     sizes, step budget) can be calibrated for the wrong temperature scale.
  3. ruggedness-adaptive-restart -- every `window` moves the smith reports her own
     realised acceptance rate and whether the best score has improved recently;
     the candidate's policy may use that observed feedback to COMPOUND-escalate
     the temperature (reheat in place, or restart from the best state found so
     far, or from a fresh random state) a bounded number of times. A policy that
     reacts to what it actually observed self-calibrates to the hidden scale; a
     policy that only ever decays once cannot recover from a bad initial guess.

LANDSCAPE.  A public instance's N bits are partitioned into consecutive blocks.
For block b_k with population count u (0..b_k) among its bits:
  - "onemax" block:  reward(u) = u * slope           (more 1s is always better)
  - "trap" block:     reward(u) = (b_k - u) * slope    for u < b_k
                       reward(b_k) = b_k * slope * BONUS   (BONUS > 1, unique peak)
  A trap block's reward DECREASES monotonically as bits are flipped from 0 to 1,
  all the way until the LAST bit completes it, at which point reward jumps far
  above the u=0 "consolation" value. Single-bit-flip hill-climbing therefore
  drives every trap block straight to u=0 and sits there forever, unless the
  searcher is willing to accept a long run of locally-worsening flips.
  `slope` and `BONUS` are fixed per instance but withheld from the public view;
  only block sizes/types, N, and the step budget are public. Total reward is the
  sum of all blocks' rewards; the true global optimum is every bit set to 1.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER INSTANCE).
See statement.md for the exact schema. In short: the candidate emits ONE control
policy {T0, alpha, window, stagnation_window, accept_floor, reheat_factor,
restart_mode, max_events}. The evaluator then runs its OWN frozen Metropolis
kernel for `steps` moves using that policy and reports the best total reward the
kernel ever visited.

SCORING (deterministic, no wall-time). For instance i: baseline_i = best reward a
policy with T0=0 (pure greedy hill-climbing, i.e. the "do nothing clever" search)
reaches; max_i = the closed-form true optimum (all bits set). r_i = clamp(0.1 +
0.9*(obj_i - baseline_i)/(max_i - baseline_i), 0, 1). Ratio = mean(r_i).

ISOLATION.  The candidate runs OS-sandboxed via isorun.run_candidate and only
ever sees the public instance; slope/BONUS, the move RNG stream, and the scoring
machinery live only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:  Ratio: <mean r>   /   Vector: [r_1, ..., r_n]
"""
import sys, json, math, random
import isorun

# ============================== instance family =============================

def _specs():
    # (name, seed, trap_block_sizes, onemax_block_sizes, slope, bonus, steps)
    # `slope` spans a deliberately WIDE range across instances (2.0 -> 36.0, an
    # 18x spread) while T0_max stays fixed at 15.0 (see _build_instances). A
    # trap block's crossing barrier scales with slope*(size-1), so the
    # "inferno" tier's barrier is many times the "breeze"/"ember" tiers'. Per-
    # attempt crossing probability falls off ~exp(-barrier/T); on the inferno
    # tier at any T <= T0_max that probability is small enough that the
    # available step budget gives it very little chance of ever happening. A
    # policy whose temperature never exceeds T0_max (any constant-T or single
    # monotone-decay policy, however well the initial guess is chosen) is
    # therefore stuck near the do-nothing baseline on that tier. Only
    # compounding escalation (leg_T0 = Tcur * reheat_factor, applied repeatedly
    # and explicitly UNBOUNDED by T0_max) reaches temperatures where crossing
    # becomes likely within the given budget -- this closes the loophole where
    # one lucky fixed constant dominates every instance.
    return [
        ("breeze-easy-A",   3101, [3, 4, 4],          [3, 3, 4],       2.0, 2.5, 12000),
        ("breeze-easy-B",   3102, [3, 4, 5],          [3, 4],          3.0, 2.5, 12000),
        ("ember-mid-A",     3201, [4, 5, 5],          [3, 4],          5.0, 2.5, 13000),
        ("ember-mid-B",     3202, [5, 5, 6],          [3, 3, 4],       6.0, 2.5, 13000),
        ("ember-mid-C",     3301, [4, 5, 6],          [4, 4],          7.0, 2.5, 14000),
        ("inferno-hard-A",  3401, [5, 6, 6],          [3, 4],          20.0, 2.5, 8000),
        ("inferno-hard-B",  3402, [5, 6, 7],          [3, 3, 4],       24.0, 2.5, 8500),
        ("inferno-held-C",  3501, [4, 5, 6, 6],       [3, 4],          28.0, 2.5, 9000),
        ("inferno-held-D",  3502, [5, 6, 6, 7],       [3, 4, 4],       32.0, 2.5, 9500),
        ("inferno-held-E",  3601, [5, 5, 6, 6, 7],    [3, 3, 4],       36.0, 2.5, 10000),
    ]


def _block_reward(typ, b, u, slope, bonus):
    if typ == "one":
        return u * slope
    if u == b:
        return b * slope * bonus
    return (b - u) * slope


def _total_F(blocks, slope, bonus, x, offsets):
    s = 0.0
    for (typ, b), off in zip(blocks, offsets):
        u = sum(x[off:off + b])
        s += _block_reward(typ, b, u, slope, bonus)
    return s


def _max_F(blocks, slope, bonus):
    return sum(_block_reward(typ, b, b, slope, bonus) for typ, b in blocks)


def _build_instances():
    out = []
    for name, seed, trap_bs, one_bs, slope, bonus, steps in _specs():
        rng = random.Random(seed)
        blocks = [("trap", b) for b in trap_bs] + [("one", b) for b in one_bs]
        rng.shuffle(blocks)
        N = sum(b for _, b in blocks)
        offsets = []
        o = 0
        for _, b in blocks:
            offsets.append(o); o += b
        x0 = [rng.randint(0, 1) for _ in range(N)]
        kernel_seed = seed * 7919 + 13
        max_events_cap = 80
        public = {
            "name": name,
            "N": N,
            "blocks": [{"type": ("trap" if t == "trap" else "onemax"), "size": b} for t, b in blocks],
            "steps": steps,
            "T0_max": 15.0,
            "reheat_factor_max": 3.0,
            "max_events_cap": max_events_cap,
        }
        hidden = {"blocks": blocks, "offsets": offsets, "x0": x0, "slope": slope,
                  "bonus": bonus, "steps": steps, "kernel_seed": kernel_seed,
                  "N": N, "max_events_cap": max_events_cap,
                  "T0_max": 15.0, "reheat_factor_max": 3.0}
        out.append({"public": public, "hidden": hidden})
    return out


# =============================== answer schema ===============================

_RESTART_MODES = ("reheat", "restart_best", "restart_random")


def _validate_policy(answer, hidden):
    if not isinstance(answer, dict):
        return None
    T0 = answer.get("T0")
    alpha = answer.get("alpha")
    window = answer.get("window")
    stag_w = answer.get("stagnation_window")
    accept_floor = answer.get("accept_floor")
    reheat_factor = answer.get("reheat_factor")
    restart_mode = answer.get("restart_mode")
    max_events = answer.get("max_events")

    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    if not is_num(T0) or not (0.0 <= T0 <= hidden["T0_max"] + 1e-9):
        return None
    if not is_num(alpha) or not (0.0 < alpha <= 1.0):
        return None
    if isinstance(window, bool) or not isinstance(window, int) or not (5 <= window <= hidden["steps"]):
        return None
    if isinstance(stag_w, bool) or not isinstance(stag_w, int) or stag_w < 1:
        return None
    if not is_num(accept_floor) or not (0.0 <= accept_floor <= 1.0):
        return None
    if not is_num(reheat_factor) or not (1.0 <= reheat_factor <= hidden["reheat_factor_max"] + 1e-9):
        return None
    if restart_mode not in _RESTART_MODES:
        return None
    if isinstance(max_events, bool) or not isinstance(max_events, int) or not (0 <= max_events <= hidden["max_events_cap"]):
        return None
    return dict(T0=float(T0), alpha=float(alpha), window=window, stagnation_window=stag_w,
                accept_floor=float(accept_floor), reheat_factor=float(reheat_factor),
                restart_mode=restart_mode, max_events=max_events)


# ============================ frozen SA kernel ===============================

def _run_kernel(hidden, policy):
    blocks = hidden["blocks"]; offsets = hidden["offsets"]; slope = hidden["slope"]
    bonus = hidden["bonus"]; N = hidden["N"]; steps = hidden["steps"]
    rng = random.Random(hidden["kernel_seed"])

    bit_block = [0] * N
    for bi, ((typ, b), off) in enumerate(zip(blocks, offsets)):
        for i in range(off, off + b):
            bit_block[i] = bi

    x = list(hidden["x0"])
    u_of_block = [sum(x[off:off + b]) for (typ, b), off in zip(blocks, offsets)]
    F = _total_F(blocks, slope, bonus, x, offsets)
    best = F
    best_x = list(x)

    T0 = policy["T0"]; alpha = policy["alpha"]; window = policy["window"]
    stag_w = policy["stagnation_window"]; accept_floor = policy["accept_floor"]
    reheat_factor = policy["reheat_factor"]; restart_mode = policy["restart_mode"]
    max_events = policy["max_events"]

    leg_T0 = T0
    k_in_leg = 0
    events_used = 0
    steps_since_improve = 0
    accepts_in_window = 0
    Tcur = leg_T0

    for t in range(steps):
        i = rng.randrange(N)
        bi = bit_block[i]
        typ, b = blocks[bi]
        off = offsets[bi]
        u_old = u_of_block[bi]
        old_r = _block_reward(typ, b, u_old, slope, bonus)
        u_new = u_old - 1 if x[i] == 1 else u_old + 1
        new_r = _block_reward(typ, b, u_new, slope, bonus)
        delta = new_r - old_r
        accept = delta >= 0
        if not accept and Tcur > 1e-12:
            accept = rng.random() < math.exp(delta / Tcur)
        if accept:
            x[i] = 1 - x[i]
            u_of_block[bi] = u_new
            F += delta
            accepts_in_window += 1
            if F > best + 1e-9:
                best = F; best_x = list(x); steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            steps_since_improve += 1

        if (t + 1) % window == 0:
            acc_rate = accepts_in_window / window
            accepts_in_window = 0
            stagnant = steps_since_improve >= stag_w
            if acc_rate < accept_floor and stagnant and events_used < max_events:
                events_used += 1
                leg_T0 = Tcur * reheat_factor
                k_in_leg = 0
                if restart_mode == "restart_best":
                    x = list(best_x); F = best
                    for bi2, ((typ2, b2), off2) in enumerate(zip(blocks, offsets)):
                        u_of_block[bi2] = sum(x[off2:off2 + b2])
                elif restart_mode == "restart_random":
                    x = [rng.randint(0, 1) for _ in range(N)]
                    for bi2, ((typ2, b2), off2) in enumerate(zip(blocks, offsets)):
                        u_of_block[bi2] = sum(x[off2:off2 + b2])
                    F = _total_F(blocks, slope, bonus, x, offsets)
                    if F > best + 1e-9:
                        best = F; best_x = list(x); steps_since_improve = 0
            else:
                k_in_leg += 1
            Tcur = leg_T0 * (alpha ** k_in_leg)
    return best


_TRIVIAL_POLICY = dict(T0=0.0, alpha=1.0, window=10 ** 9, stagnation_window=10 ** 9,
                        accept_floor=0.0, reheat_factor=1.0, restart_mode="reheat", max_events=0)


def score(inst, answer):
    hidden = inst["hidden"]
    policy = _validate_policy(answer, hidden)
    if policy is None:
        return False, 0.0
    obj = _run_kernel(hidden, policy)
    if not math.isfinite(obj):
        return False, 0.0
    return True, obj


def baseline(inst):
    trivial = dict(_TRIVIAL_POLICY)
    trivial["window"] = min(trivial["window"], inst["hidden"]["steps"])
    return _run_kernel(inst["hidden"], trivial)


def _max_val(inst):
    h = inst["hidden"]
    return _max_F(h["blocks"], h["slope"], h["bonus"])


# =================================== main =====================================

def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>", file=sys.stderr)
        sys.exit(2)
    cand = sys.argv[1]
    insts = _build_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False; obj = 0.0
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        m = _max_val(inst)
        denom = max(m - b, 1e-9)
        r = 0.1 + 0.9 * (obj - b) / denom
        r = max(0.0, min(1.0, r))
        vec.append(r if (r == r) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
