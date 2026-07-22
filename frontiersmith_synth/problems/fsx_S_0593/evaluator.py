#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0593 -- "Skew Ahead: Quoting a Market Book Against
Adverse Scripted Order Flow" (family: quote-flow-inventory-control; format B,
quality-metric; theme: market maker quoting against a scripted order flow).

THEME.  You run a market maker on a single asset over T discrete steps.  At each
step t you post a two-sided quote around the public mid price m_t: a BID a distance
hb_t below mid (you buy there) and an ASK a distance ha_t above mid (you sell
there), each backed by a maximum quoted SIZE (zb_t on the bid, za_t on the ask).

SCRIPTED ORDER FLOW (mechanism 1: adversarial-scripted-orderflow).  Each step a
deterministic, SEEDED order-flow script releases S_t units of marketable SELL
interest (which can hit your bid, so YOU BUY and grow long) and B_t units of
marketable BUY interest (which can lift your ask, so YOU SELL and grow short).
Only a fraction of that interest actually trades against you, and the fraction
RISES as your quote tightens: the units filled on the bid are
    S_t * max(0, 1 - hb_t / W)   capped by your quoted size zb_t,
and on the ask
    B_t * max(0, 1 - ha_t / W)   capped by your quoted size za_t.
The script is ADVERSARIAL: in the run-up to a seeded price move it leans its
interest INTO the move -- a burst of BUY interest ahead of an up-move (lifting
your ask, forcing you short right before the price rises) and a burst of SELL
interest ahead of a down-move.  So the flow tries to load you with exactly the
inventory that the coming move will punish.

INVENTORY RISK CARRYOVER (mechanism 2: inventory-risk-carryover).  Fills change a
signed inventory q that CARRIES OVER across steps.  Inventory is bounded to
[-Qmax, +Qmax] (a fill that would breach the cap is clamped).  Every step you pay a
convex holding charge lam * q^2 on the inventory you are left carrying, and at the
end a terminal charge mu * q_T^2.  Your book is marked to the final mid m_T.

OBJECTIVE (maximize) per instance:
    PnL = cash_T + q_T * m_T  -  lam * sum_t q_t^2  -  mu * q_T^2
where cash accumulates -qb*(m_t-hb_t) on each buy and +qs*(m_t+ha_t) on each sell.
Quoting nothing (never filling) yields PnL = 0.

INNOVATION HOOK (what `strong` exploits).  The mid path m and the flow S_t, B_t are
PUBLIC, so the coming move is ANTICIPATED.  The insight is to SKEW the quotes to
lean AGAINST the inventory the flow will otherwise force you to hold through the
move: ahead of an up-move, widen/withdraw the ask (refuse to be pushed short) and
tighten the bid to deliberately build a LONG book, so the rise pays you -- then
unwind after the move.  A myopic maker that always posts the spread that maximizes
per-step capture (tight, symmetric, full size) instead lets the adverse burst load
it the WRONG way and bleeds the move, trading a sliver of spread for a large
directional loss.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "W": float, "Qmax": float,
             "lam": float, "mu": float,
             "m": [m_0 ... m_T],        # T+1 mid prices (public; the move is visible)
             "S": [S_0 ... S_{T-1}],    # sell interest per step (hits your bid)
             "B": [B_0 ... B_{T-1}]}    # buy interest per step (lifts your ask)
  stdout: ONE JSON object:
            {"hb": [...T...], "ha": [...T...],   # bid / ask half-spreads (>=0)
             "zb": [...T...], "za": [...T...]}   # bid / ask quoted sizes    (>=0)
  VALID iff each of hb,ha,zb,za is a list of exactly T finite numbers >= 0
  (no NaN/inf/bool/negative).  Any violation, crash, timeout, or non-JSON -> 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    PnL_cand = objective above for the candidate's quotes.
    A generous, UNREACHABLE upper bound hi is formed from the instance's own
    parameters (full directional capture at the cap on every move, plus the
    textbook maximum spread capture on all flow), scaled by GAIN:
       hi = GAIN * ( Qmax * Dsum  +  0.25 * W * sum_t (S_t + B_t) )
    where Dsum is the total absolute size of the seeded moves.  Then
       r = clamp( 0.1 + 0.9 * PnL_cand / hi , 0, 1 ).
  Quoting nothing scores exactly 0.1; the loose bound keeps even a strong,
  correctly-skewed book below 1.0 (headroom is intentional).  The final score is
  the mean of r over 10 fixed seeded instances (adverse-move traps, calm
  spread-capture instances, and held-out twin-move regimes).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED
SUBPROCESS via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The
objective and the normalization are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

GAIN = 1.15   # normalization slack: larger -> lower ratios / more headroom


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- transition / objective ----------------------
def simulate(inst, hb, ha, zb, za):
    """Replay the deterministic quoting episode; return the PnL objective."""
    T = inst["T"]
    m = inst["m"]
    S = inst["S"]
    B = inst["B"]
    W = inst["W"]
    Qmax = inst["Qmax"]
    lam = inst["lam"]
    mu = inst["mu"]

    q = 0.0
    cash = 0.0
    pen = 0.0
    for t in range(T):
        # --- bid side: sell interest S_t hits your bid; you BUY, grow long ---
        fb = 1.0 - hb[t] / W
        if fb < 0.0:
            fb = 0.0
        qb = S[t] * fb
        if zb[t] < qb:
            qb = zb[t]
        room_up = Qmax - q          # cannot exceed +Qmax
        if qb > room_up:
            qb = room_up
        if qb < 0.0:
            qb = 0.0
        q += qb
        cash -= qb * (m[t] - hb[t])
        # --- ask side: buy interest B_t lifts your ask; you SELL, grow short ---
        fa = 1.0 - ha[t] / W
        if fa < 0.0:
            fa = 0.0
        qs = B[t] * fa
        if za[t] < qs:
            qs = za[t]
        room_dn = q + Qmax          # cannot fall below -Qmax
        if qs > room_dn:
            qs = room_dn
        if qs < 0.0:
            qs = 0.0
        q -= qs
        cash += qs * (m[t] + ha[t])
        # --- convex holding charge on carried inventory ---
        pen += lam * q * q
    obj = cash + q * m[T] - pen - mu * q * q
    return obj


def _hi(inst):
    Fsum = 0.0
    for t in range(inst["T"]):
        Fsum += inst["S"][t] + inst["B"][t]
    return GAIN * (inst["Qmax"] * inst["dsum"] + 0.25 * inst["W"] * Fsum)


# ----------------------------- instance family -----------------------------
def _base_flow(seed, T, S0, B0):
    ni = _rng(seed)
    S = [float(S0 + ni(-2, 2)) for _ in range(T)]
    B = [float(B0 + ni(-2, 2)) for _ in range(T)]
    return ni, S, B


def _adverse(seed, T, direction, delta, Hsurge, surge):
    """A single anticipated move at te = T//2.  In the H steps BEFORE it, the flow
    leans INTO the move so a myopic maker is loaded the wrong way; after it the flow
    returns to base so a skewed book can unwind.  This is a TRAP family."""
    ni, S, B = _base_flow(seed, T, 20, 20)
    m = [100.0] * (T + 1)
    te = T // 2
    d = delta * direction
    for t in range(te, T + 1):
        m[t] += d
    for t in range(max(0, te - Hsurge), te):
        if direction > 0:                       # up-move: buyers burst -> pushes you short
            B[t] += surge
            S[t] = max(4.0, S[t] - surge * 0.5)
        else:                                   # down-move: sellers burst -> pushes you long
            S[t] += surge
            B[t] = max(4.0, B[t] - surge * 0.5)
    return {"T": T, "W": 1.0, "Qmax": 100.0, "lam": 0.0025, "mu": 0.02,
            "m": m, "S": S, "B": B, "dsum": abs(d)}


def _calm(seed, T):
    """No move (Dsum contribution tiny): symmetric flow, flat price.  Positioning
    cannot help; only spread capture pays, so the myopic recipe is near-optimal
    here -- these instances keep greedy honestly above do-nothing."""
    ni, S, B = _base_flow(seed, T, 22, 22)
    m = [100.0] * (T + 1)
    # a whisper of drift so the family is not perfectly degenerate, tiny vs spread
    for t in range(1, T + 1):
        m[t] = m[t - 1] + (0.02 if (t % 2 == 0) else -0.02)
    dsum = 0.0
    for t in range(1, T + 1):
        dsum += abs(m[t] - m[t - 1])
    return {"T": T, "W": 1.0, "Qmax": 100.0, "lam": 0.0025, "mu": 0.02,
            "m": m, "S": S, "B": B, "dsum": dsum * 0.0 + 0.5}  # tiny bound floor


def _twin(seed, T, d1, d2, Hsurge, surge):
    """Held-out: two opposite moves (up then down) at te1=T//3 and te2=2T//3, each
    preceded by an adverse burst.  You must build long, flip through the cap to
    short, and partly unwind -- the position limit and holding cost make even a good
    skew imperfect, so this generalization family leaves the most headroom."""
    ni, S, B = _base_flow(seed, T, 20, 20)
    m = [100.0] * (T + 1)
    te1 = T // 3
    te2 = (2 * T) // 3
    for t in range(te1, T + 1):
        m[t] += d1                              # first move (up)
    for t in range(te2, T + 1):
        m[t] += d2                              # second move (down)
    for t in range(max(0, te1 - Hsurge), te1):  # up-move burst: buyers
        B[t] += surge
        S[t] = max(4.0, S[t] - surge * 0.5)
    for t in range(max(te1, te2 - Hsurge), te2):  # down-move burst: sellers
        S[t] += surge
        B[t] = max(4.0, B[t] - surge * 0.5)
    return {"T": T, "W": 1.0, "Qmax": 100.0, "lam": 0.0025, "mu": 0.02,
            "m": m, "S": S, "B": B, "dsum": abs(d1) + abs(d2)}


def _build_instances():
    T = 12
    out = []
    specs = [
        ("adv_up1",  "adv", 59301, +1, 5.0, 3, 40.0),
        ("adv_dn1",  "adv", 59302, -1, 5.0, 3, 40.0),
        ("adv_up2",  "adv", 59303, +1, 6.0, 3, 44.0),
        ("adv_dn2",  "adv", 59304, -1, 4.5, 3, 38.0),
        ("calm1",    "calm", 59311, 0, 0.0, 0, 0.0),
        ("calm2",    "calm", 59312, 0, 0.0, 0, 0.0),
        ("calm3",    "calm", 59313, 0, 0.0, 0, 0.0),
        ("twin1",    "twin", 59321, 0, 0.0, 3, 40.0),
        ("twin2",    "twin", 59322, 0, 0.0, 3, 42.0),
        ("adv_up3",  "adv", 59305, +1, 5.5, 3, 42.0),
    ]
    for name, kind, seed, direction, delta, H, surge in specs:
        if kind == "adv":
            inst = _adverse(seed, T, direction, delta, H, surge)
        elif kind == "calm":
            inst = _calm(seed, T)
        else:
            inst = _twin(seed, T, 5.0, -5.0, H, surge)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- validation ----------------------------------
def _vec(answer, key, T):
    v = answer.get(key)
    if not isinstance(v, list) or len(v) != T:
        return None
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")) or x < 0.0:
            return None
        out.append(x)
    return out


def _valid_quotes(inst, answer):
    if not isinstance(answer, dict):
        return None
    T = inst["T"]
    hb = _vec(answer, "hb", T)
    ha = _vec(answer, "ha", T)
    zb = _vec(answer, "zb", T)
    za = _vec(answer, "za", T)
    if hb is None or ha is None or zb is None or za is None:
        return None
    return hb, ha, zb, za


# ----------------------------- scoring driver ------------------------------
def _public(inst):
    return {"name": inst["name"], "T": inst["T"], "W": inst["W"],
            "Qmax": inst["Qmax"], "lam": inst["lam"], "mu": inst["mu"],
            "m": list(inst["m"]), "S": list(inst["S"]), "B": list(inst["B"])}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        hi = _hi(inst)
        if hi <= 1e-9:
            hi = 1e-9
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            parsed = _valid_quotes(inst, ans)
            if parsed is None:
                vec.append(0.0)
                continue
            hb, ha, zb, za = parsed
            pnl = simulate(inst, hb, ha, zb, za)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * pnl / hi
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
