# TIER: strong
# The insight: don't fix this round's acceptance bar from today's economics
# alone -- ROLL FORWARD a short lookahead window under "keep using today's
# break-even bar" and check whether mix health would dip below the cascade
# threshold within that window. If so, this round's bar is raised (several
# candidate margins are tried and the one with the best PROJECTED cumulative
# expected value over the lookahead wins); once the projected danger has
# passed, the bar relaxes back to break-even to recover volume/throughput.
# This is a genuine reformulation -- a short-horizon rollout/planning view
# over the SAME public dynamics -- not "greedy with a bigger constant."
import sys, json, math

CAND_MARGINS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
LOOKAHEAD_K = 11


def sigmoid(x):
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def p_good(s, h, temp):
    return sigmoid((s - (1.0 - h)) / temp)


def step_update(h, r, vol_mult, bad_frac, approval_rate, in_spiral, p):
    h_recover = p["h_recover"] * (p["spiral_h_recover_mult"] if in_spiral else 1.0)
    r_recover = p["r_recover"] * (p["spiral_r_recover_mult"] if in_spiral else 1.0)
    h_new = h + h_recover * (1 - h) * (1 - bad_frac) - p["h_decay_bad"] * bad_frac
    h_new = min(p["h_ceiling"], max(p["h_floor"], h_new))
    r_new = r + r_recover * (1 - r) * (1 - bad_frac) - p["r_decay_bad"] * bad_frac
    r_new = min(1.0, max(p["r_floor"], r_new))
    new_spiral = in_spiral or (h_new < p["cascade_threshold"])
    if approval_rate < p["vol_shrink_threshold"]:
        vol_mult = max(p["vol_min_frac"], vol_mult - p["vol_shrink_step"])
    else:
        vol_mult = min(1.0, vol_mult + p["vol_grow_step"])
    return h_new, r_new, vol_mult, new_spiral


def breakeven_pg(r, p, in_spiral):
    c_eff = p["c_bad"] * (p["spiral_c_bad_mult"] if in_spiral else 1.0) * (1 + p["bad_amp"] * (1 - r))
    return c_eff / (p["v_good"] * r + c_eff)


def rollout_value(score, p, t0, B, h, r, vol_mult, in_spiral, threshold_now, horizon):
    """Expected cumulative value from round t0..horizon-1, using threshold_now
    for round t0 and break-even thereafter. Belief-based (no hidden draws)."""
    total = 0.0
    for t in range(t0, horizon):
        vol_t = max(1, round(B * vol_mult))
        be_now = threshold_now if t == t0 else breakeven_pg(r, p, in_spiral)
        n_appr = 0
        exp_bad = 0.0
        for j in range(B):
            i = t * B + j
            if j >= vol_t:
                continue
            pg = p_good(score[i], h, p["temp"])
            if pg > be_now:
                n_appr += 1
                c_eff = p["c_bad"] * (p["spiral_c_bad_mult"] if in_spiral else 1.0)
                total += pg * p["v_good"] * r - (1 - pg) * c_eff * (1 + p["bad_amp"] * (1 - r))
                exp_bad += (1 - pg)
        approval_rate = n_appr / vol_t if vol_t else 0.0
        bad_frac = exp_bad / vol_t if vol_t else 0.0
        h, r, vol_mult, in_spiral = step_update(h, r, vol_mult, bad_frac, approval_rate, in_spiral, p)
    return total


inst = json.load(sys.stdin)
T, B, N = inst["T"], inst["B"], inst["N"]
score = inst["score"]
p = inst["params"]
h, r, vol_mult, in_spiral = inst["h0"], inst["r0"], 1.0, False

decisions = [0] * N
for t in range(T):
    vol_t = max(1, round(B * vol_mult))
    be = breakeven_pg(r, p, in_spiral)
    horizon = min(T, t + LOOKAHEAD_K)

    best_val = None
    best_thr = be
    for m in CAND_MARGINS:
        thr = be + m
        val = rollout_value(score, p, t, B, h, r, vol_mult, in_spiral, thr, horizon)
        if best_val is None or val > best_val + 1e-9:
            best_val = val
            best_thr = thr

    n_appr = 0
    exp_bad = 0.0
    for j in range(B):
        i = t * B + j
        if j >= vol_t:
            continue
        pg = p_good(score[i], h, p["temp"])
        a = 1 if pg > best_thr else 0
        decisions[i] = a
        if a:
            n_appr += 1
            exp_bad += (1 - pg)
    approval_rate = n_appr / vol_t if vol_t else 0.0
    bad_frac = exp_bad / vol_t if vol_t else 0.0
    h, r, vol_mult, in_spiral = step_update(h, r, vol_mult, bad_frac, approval_rate, in_spiral, p)

print(json.dumps({"decisions": decisions}))
