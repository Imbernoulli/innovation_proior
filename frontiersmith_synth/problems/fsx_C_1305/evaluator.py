#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1305 -- "Two-Sided Quotes on a Tape That Talks Back"
(family: market-maker-policy; format B, quality-metric).

THEME.  You run a small market-making desk on one instrument for a fixed session of
T ticks.  Every tick you post a bid and an ask around the last observed price.  Two
kinds of counterparties can trade against you: UNINFORMED (noise) flow that trades
for liquidity reasons and does not predict anything, and INFORMED flow that only
shows up when it has a short-horizon edge -- and only trades against you when your
quote is mispriced relative to where the price is *about* to settle.  Quoting a
tight symmetric spread maximizes how much noise flow you capture, but it also lets
informed flow pick you off for free exactly when the price is about to move against
your resulting position.  You must additionally manage inventory risk: holding a
large position (in either direction) is itself costly.

Your policy is a SINGLE linear rule, committed once per session (no interaction):
    skew         = inv_coef * (inventory / Qmax)  +  ofi_coef * ofi_t
    bid_t        = last_price - half_spread - skew
    ask_t        = last_price + half_spread - skew
`ofi_t` is a public, causally-lagged order-flow-imbalance feature (recent net
signed order flow, using only ticks strictly BEFORE t) that the evaluator computes
internally each tick and feeds through your committed coefficients; you never see
the live session's tick-by-tick data (that would leak the future).  Instead you are
given a CALIBRATION window: a resolved, independent past session (same regime) with
its own `ofi` and `next_ret` (the realized `ret_horizon`-tick-ahead return) arrays,
from which you must infer how predictive order flow is in this regime before
committing your three numbers for the live session.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the schema.
  stdout: ONE JSON object: {"half_spread": h, "inv_coef": a, "ofi_coef": b}
          h must be a finite number in [0, 1e6]; a, b finite numbers in [-1e4, 1e4].
          Anything else (wrong type, NaN/Inf, missing key, crash, timeout, non-JSON)
          scores that instance 0.0.

SCORING (deterministic; no wall-time).  The evaluator itself re-simulates the LIVE
session tick by tick with your (h, a, b), computing realized cash, terminal
mark-to-market, and an inventory-risk penalty (see `simulate`).  This gives
`pnl_cand`.  It also computes, itself, `pnl_oracle`: the best PnL over a fixed grid
of (h, a, b) triples run on the SAME live session (the evaluator has full access to
the hidden tick data; this is a reference, not something the candidate can see).
Normalized score (never trading -> pnl = 0 is the 0.1 anchor):
    r = clamp(0.1 + 0.9 * pnl_cand / max(pnl_oracle, 1e-6), 0, 1)
The reported Ratio is the mean of r over 10 seeded instances (3 pure-noise warm-ups
+ 7 mixed noise/informed sessions, some larger/held-out).  Because the grid is
coarse and finite, pnl_oracle is a strong but not perfect reference -> headroom.

ISOLATION.  The candidate runs OS-sandboxed via `isorun.run_candidate`; it only
ever sees the PUBLIC instance (session metadata + the resolved calibration window).
The live session's hidden tick data (prices, order flow, informed/noise labels)
never leave this process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

MASK64 = (1 << 64) - 1
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407

FULL_FILL_HS = 0.15    # half-spread at/below which noise flow fills in FULL
ZERO_FILL_HS = 0.45    # half-spread at/above which noise flow fills NOT AT ALL
MAX_SKEW = 5.0        # internal safety clamp on |skew| applied to every policy
LAM = 6e-5             # inventory-risk coefficient (risk += LAM * inventory^2 per tick)
W_OFI = 5               # causal order-flow-imbalance rolling window (ticks)
LEAD = 6               # ticks of informed pre-positioning before a regime ramp completes
RET_HORIZON = LEAD     # calibration next_ret is the H-tick-ahead return, H=RET_HORIZON

H_GRID = [0.02, 0.05, 0.09, 0.15, 0.25]
A_GRID = [0.0, 1.0, 2.5, 5.0, 9.0]
B_GRID = [-30.0, -15.0, -6.0, 0.0, 6.0, 15.0, 30.0]


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed & MASK64]

    def u01():
        state[0] = (state[0] * LCG_A + LCG_C) & MASK64
        return ((state[0] >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u01


def _gauss(u01):
    u1 = u01()
    if u1 < 1e-12:
        u1 = 1e-12
    u2 = u01()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _uint(u01, lo, hi):
    span = hi - lo + 1
    v = int(u01() * span)
    if v >= span:
        v = span - 1
    if v < 0:
        v = 0
    return lo + v


# ----------------------------- session generator ----------------------------
def _gen_session(seed, T, n_jumps, jump_mag_lo, jump_mag_hi, informed_lo, informed_hi,
                  base_vol, noise_scale, informed_flag):
    """One deterministic tick session. Returns dict with hidden tick arrays."""
    u01 = _rng(seed)

    # 1. choose regime-ramp completion ticks (spaced far enough apart to never overlap)
    jump_times = []
    if informed_flag and n_jumps > 0:
        lo_b, hi_b = LEAD + 5, max(LEAD + 6, T - 3)
        attempts = 0
        while len(jump_times) < n_jumps and attempts < 4000:
            attempts += 1
            cand = _uint(u01, lo_b, hi_b)
            if all(abs(cand - e) >= LEAD + 3 for e in jump_times):
                jump_times.append(cand)
        jump_times.sort()
    jump_sign = {}
    jump_mag = {}
    for tj in jump_times:
        jump_sign[tj] = 1.0 if u01() < 0.5 else -1.0
        jump_mag[tj] = jump_mag_lo + u01() * (jump_mag_hi - jump_mag_lo)

    # 2. owning-ramp map (which future jump a tick t is ramping toward), pure function
    owning = {}
    for tj in jump_times:
        for t in range(max(1, tj - LEAD), tj + 1):
            owning[t] = tj

    # 3. price path: background diffusion + evenly-spread ramp increment on ramp ticks
    p = [100.0]
    for t in range(1, T + 1):
        step = _gauss(u01) * base_vol
        if t in owning:
            tj = owning[t]
            step += jump_sign[tj] * jump_mag[tj] / (LEAD + 1)
        p.append(p[-1] + step)

    # 4. noise order flow (both sides, every tick)
    noise_buy = [0] * (T + 1)
    noise_sell = [0] * (T + 1)
    for t in range(1, T + 1):
        noise_buy[t] = _uint(u01, 0, noise_scale)
        noise_sell[t] = _uint(u01, 0, noise_scale)

    # 5. informed order flow: only active strictly BEFORE ramp completion, one side only
    informed_buy = [0] * (T + 1)
    informed_sell = [0] * (T + 1)
    for tj in jump_times:
        s = jump_sign[tj]
        for t in range(max(1, tj - LEAD), tj):
            sz = _uint(u01, informed_lo, informed_hi)
            if s > 0:
                informed_buy[t] = sz
            else:
                informed_sell[t] = sz

    # 6. fair value informed traders act on: the ramp's completion price
    fair = {}
    for tj in jump_times:
        for t in range(max(1, tj - LEAD), tj):
            fair[t] = p[tj]

    # 7. causal order-flow-imbalance feature (window ending strictly before t)
    raw = [0] * (T + 1)
    for t in range(1, T + 1):
        raw[t] = (noise_buy[t] + informed_buy[t]) - (noise_sell[t] + informed_sell[t])
    ofi = [0.0] * (T + 1)
    for t in range(1, T + 1):
        s = 0
        for t2 in range(max(1, t - W_OFI), t):
            s += raw[t2]
        ofi[t] = float(s)

    return {"p": p, "ofi": ofi, "noise_buy": noise_buy, "noise_sell": noise_sell,
            "informed_buy": informed_buy, "informed_sell": informed_sell, "fair": fair, "T": T}


# ----------------------------- forward simulator -----------------------------
def simulate(h, a, b, sess, Qmax):
    p = sess["p"]; ofi = sess["ofi"]
    nb = sess["noise_buy"]; ns = sess["noise_sell"]
    ib = sess["informed_buy"]; isl = sess["informed_sell"]
    fair = sess["fair"]
    T = sess["T"]
    hs = h if h > 1e-6 else 1e-6
    q = 0.0
    cash = 0.0
    risk = 0.0
    for t in range(1, T + 1):
        mid_prev = p[t - 1]
        skew = a * (q / Qmax) + b * ofi[t]
        if skew > MAX_SKEW:
            skew = MAX_SKEW
        elif skew < -MAX_SKEW:
            skew = -MAX_SKEW
        bid = mid_prev - hs - skew
        ask = mid_prev + hs - skew

        if hs <= FULL_FILL_HS:
            frac = 1.0
        elif hs >= ZERO_FILL_HS:
            frac = 0.0
        else:
            frac = (ZERO_FILL_HS - hs) / (ZERO_FILL_HS - FULL_FILL_HS)
        f_nb = nb[t] * frac
        f_ns = ns[t] * frac
        f_ib = 0.0
        f_is = 0.0
        ft = fair.get(t)
        if ib[t] > 0 and ft is not None and ask < ft:
            f_ib = float(ib[t])
        if isl[t] > 0 and ft is not None and bid > ft:
            f_is = float(isl[t])

        desired_buy = f_nb + f_ib      # counterparties BUYING from us (hit our ask)
        desired_sell = f_ns + f_is     # counterparties SELLING to us (hit our bid)

        room_sell = q + Qmax           # we can sell (q decreases) down to -Qmax
        buy_fill = desired_buy if desired_buy <= room_sell else room_sell
        if buy_fill < 0.0:
            buy_fill = 0.0
        q -= buy_fill
        cash += ask * buy_fill

        room_buy = Qmax - q            # we can buy (q increases) up to +Qmax
        sell_fill = desired_sell if desired_sell <= room_buy else room_buy
        if sell_fill < 0.0:
            sell_fill = 0.0
        q += sell_fill
        cash -= bid * sell_fill

        risk += LAM * q * q

    final_val = cash + q * p[T]
    return final_val - risk


def _grid_best(sess, Qmax):
    best = -1e18
    for h in H_GRID:
        for a in A_GRID:
            for b in B_GRID:
                v = simulate(h, a, b, sess, Qmax)
                if v > best:
                    best = v
    return best


# ----------------------------- instance family -----------------------------
def _build_instance(name, seed_base, T, Qmax, n_jumps, jmag_lo, jmag_hi,
                     info_lo, info_hi, base_vol, noise_scale, informed_flag, T_calib):
    live = _gen_session(seed_base * 1000003 + 11, T, n_jumps, jmag_lo, jmag_hi,
                         info_lo, info_hi, base_vol, noise_scale, informed_flag)
    calib = _gen_session(seed_base * 1000003 + 97, T_calib, n_jumps, jmag_lo, jmag_hi,
                          info_lo, info_hi, base_vol, noise_scale, informed_flag)
    cp = calib["p"]
    next_ret = [cp[min(T_calib, t + RET_HORIZON)] - cp[t] for t in range(1, T_calib + 1)]
    public = {
        "name": name,
        "T": T,
        "Qmax": Qmax,
        "hs_bounds": [0.005, 50.0],
        "max_skew": MAX_SKEW,
        "fill_band": [FULL_FILL_HS, ZERO_FILL_HS],
        "ofi_window": W_OFI,
        "ret_horizon": RET_HORIZON,
        "vol_hint": base_vol,
        "calibration": {"n": T_calib, "ofi": calib["ofi"][1:T_calib + 1], "next_ret": next_ret},
    }
    return {"public": public, "sess": live, "Qmax": Qmax}


def make_instances():
    specs = [
        # name, seed, T, Qmax, n_jumps, jmag_lo, jmag_hi, info_lo, info_hi, base_vol, noise_scale, informed, T_calib
        ("warm_a", 301, 140, 40, 0, 0, 0, 0, 0, 0.050, 5, False, 160),
        ("warm_b", 302, 150, 45, 0, 0, 0, 0, 0, 0.060, 6, False, 170),
        ("warm_c", 303, 130, 35, 0, 0, 0, 0, 0, 0.045, 4, False, 150),
        ("mix_a",  411, 160, 40, 3, 0.5, 1.0, 3, 6, 0.050, 5, True, 160),
        ("mix_b",  412, 170, 40, 4, 0.6, 1.2, 3, 7, 0.050, 5, True, 170),
        ("mix_c",  413, 160, 45, 3, 0.5, 1.1, 3, 6, 0.055, 5, True, 160),
        ("mix_d",  414, 180, 40, 4, 0.7, 1.4, 4, 8, 0.045, 4, True, 180),
        ("mix_e",  415, 190, 50, 5, 0.9, 1.8, 4, 9, 0.055, 5, True, 190),
        # harder / larger held-out mixed instances
        ("mix_f",  416, 230, 55, 6, 1.0, 2.0, 5, 10, 0.050, 5, True, 220),
        ("mix_g",  417, 260, 60, 7, 1.1, 2.2, 5, 11, 0.055, 6, True, 240),
    ]
    out = []
    for (name, seed, T, Qmax, nj, jlo, jhi, ilo, ihi, bv, ns, inf, Tc) in specs:
        out.append(_build_instance(name, seed, T, Qmax, nj, jlo, jhi, ilo, ihi, bv, ns, inf, Tc))
    return out


# ----------------------------- answer validation -----------------------------
def _validate_answer(ans):
    if not isinstance(ans, dict):
        return False, 0.0, 0.0, 0.0
    for k in ("half_spread", "inv_coef", "ofi_coef"):
        if k not in ans:
            return False, 0.0, 0.0, 0.0
    h, a, b = ans["half_spread"], ans["inv_coef"], ans["ofi_coef"]
    for v in (h, a, b):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0, 0.0, 0.0
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            return False, 0.0, 0.0, 0.0
    h, a, b = float(h), float(a), float(b)
    if h < 0.0 or h > 1e6:
        return False, 0.0, 0.0, 0.0
    if abs(a) > 1e4 or abs(b) > 1e4:
        return False, 0.0, 0.0, 0.0
    return True, h, a, b


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        pub = inst["public"]; sess = inst["sess"]; Qmax = inst["Qmax"]
        pnl_oracle = _grid_best(sess, Qmax)
        denom = pnl_oracle if pnl_oracle > 1e-6 else 1e-6

        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        ok, h, a, b = _validate_answer(ans)
        if not ok:
            vec.append(0.0)
            continue
        try:
            pnl_cand = simulate(h, a, b, sess, Qmax)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * pnl_cand / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
