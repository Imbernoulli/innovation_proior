#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1143 -- "Three Hundred Lots: The Ledbury Room"
(family: pacing-duel-bidder; format B, quality-metric).

THEME.  An antique buyer sits through a single-session auction of THREE HUNDRED
sequential lots.  Three other bidders are in the room, each a PUBLISHED,
deterministic, budget-paced rule-follower (their exact formulas are printed
below and repeated, with per-instance constants, inside the public instance).
Every lot is a sealed-bid SECOND-PRICE auction among all four bidders: the
highest bid wins the lot and pays the value of the SECOND-highest bid; every
other bid, including a losing bid, is otherwise inert -- except that a losing
bid that is the new second-highest RAISES what the winner must pay.  That is
the whole mechanism: bidding you never intend to win with can still cost an
opponent money, and -- because the three opponents are budget-paced -- money an
opponent loses now shrinks every bid formula they will use for the rest of the
session.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the
          exact schema (n_lots, buyer_budget, lots[], opponents[]).
  stdout: ONE JSON object: {"bids": [b_0, ..., b_{n_lots-1}]}
          b_i is your (the buyer's) sealed bid on lot i: a finite, non-negative
          number.  Bids are NOT pre-clamped to your remaining budget -- you may
          bid above what you could pay if you are confident (by simulating the
          fully deterministic opponents yourself) that the bid will not win.
          If a bid DOES win and you cannot cover the second price out of your
          remaining budget, the WHOLE instance scores 0 (an unsafe bluff).

SCORING (deterministic; no wall-time).  The evaluator replays all 300 lots in
order, computing each opponent's bid from ITS OWN published formula and
remaining budget, resolving each lot's second-price auction, and updating
budgets.  Your objective is total utility = sum over lots YOU win of
(your value for the lot - the price you paid).  We normalize with a fixed
affine anchor:
    r = clamp( 0.1 + 0.9 * utility / U_ref , 0, 1 )
  where U_ref = sum of ALL lot values (the unreachable ideal of winning every
  lot for free).  Winning nothing (utility 0) scores exactly 0.1; U_ref is a
  very loose ceiling (you compete against three funded bidders and your own
  finite budget), so even excellent buyers stay well below 1.0 -- headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance (which happens to
be the FULL instance here -- this is a full-information deterministic game,
so there is nothing to hide; the isolation guarantee is about the judge
process/filesystem, not about withholding instance data).

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MASK = (1 << 64) - 1
EPS = 1e-6


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed & MASK]

    def nxt():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & MASK
        return (state[0] >> 11) / float(1 << 53)

    return nxt


def _fint(nxt, lo, hi):
    return lo + int(nxt() * (hi - lo + 1))


def _ffloat(nxt, lo, hi):
    return lo + nxt() * (hi - lo)


# ----------------------------- opponent rules -------------------------------
# Every opponent's bid is a pure function of: the lot's published estimate E_i,
# the catalogue-wide max estimate (known in advance -- it's a printed
# pre-sale catalogue), the opponent's OWN remaining budget, and how many lots
# (including this one) remain. This is what "published, deterministic" means:
# a solver can replicate these formulas exactly and simulate the whole session.
def _opp_bid(kind, params, E_i, remaining_budget, catalogue_max, remaining_lots):
    if remaining_budget <= 0:
        return 0.0
    if kind == "pacer":
        share = remaining_budget / remaining_lots
        raw = params["k"] * min(share, E_i)
    elif kind == "sniper":
        if E_i >= params["thresh"] * catalogue_max:
            raw = params["spike"] * E_i
        else:
            raw = params["low"] * E_i
    elif kind == "capper":
        raw = min(params["cap_mult"] * E_i, params["cap_frac"] * params["init_budget"])
    else:
        raw = 0.0
    if raw < 0.0:
        raw = 0.0
    if raw > remaining_budget:
        raw = remaining_budget
    return raw


# ----------------------------- instance family ------------------------------
def _build_lots(nxt, n_lots, decoy_idx, sleeper_idx):
    lots = []
    for i in range(n_lots):
        if i in decoy_idx:
            E = _ffloat(nxt, 2200.0, 4900.0)
            v = _ffloat(nxt, 60.0, 220.0)          # flashy but nearly worthless to the buyer
        elif i in sleeper_idx:
            E = _ffloat(nxt, 70.0, 230.0)
            v = _ffloat(nxt, 950.0, 1900.0)        # undervalued gem
        else:
            E = _ffloat(nxt, 90.0, 520.0)
            v = E * _ffloat(nxt, 0.7, 1.3)
        lots.append({"estimate": round(E, 2), "value": round(v, 2)})
    return lots


def _spread_idx(nxt, n_lots, count, lo_frac, hi_frac):
    lo, hi = int(n_lots * lo_frac), int(n_lots * hi_frac)
    idx = set()
    guard = 0
    while len(idx) < count and guard < count * 50:
        idx.add(_fint(nxt, lo, min(hi, n_lots - 1)))
        guard += 1
    return idx


def _make_instance(seed, n_lots, buyer_budget, n_decoy, n_sleeper,
                    pacer, sniper, capper, name):
    nxt = _rng(seed)
    decoy_idx = _spread_idx(nxt, n_lots, n_decoy, 0.0, 0.45)
    sleeper_idx = _spread_idx(nxt, n_lots, n_sleeper, 0.15, 1.0) - decoy_idx
    lots = _build_lots(nxt, n_lots, decoy_idx, sleeper_idx)
    opponents = [
        {"type": "pacer", "budget": pacer["budget"], "k": pacer["k"]},
        {"type": "sniper", "budget": sniper["budget"], "thresh": sniper["thresh"],
         "spike": sniper["spike"], "low": sniper["low"]},
        {"type": "capper", "budget": capper["budget"], "cap_mult": capper["cap_mult"],
         "cap_frac": capper["cap_frac"]},
    ]
    return {"name": name, "n_lots": n_lots, "buyer_budget": float(buyer_budget),
            "lots": lots, "opponents": opponents}


def _build_instances():
    specs = [
        # (seed, n_lots, buyer_budget, n_decoy, n_sleeper, pacer, sniper, capper)
        # -- TRAP sessions: many high-estimate decoy lots + a sniper whose budget a
        #    few bluffed decoys can cripple, followed by many undervalued sleeper
        #    lots the crippled sniper would otherwise have kept contesting.
        (1001, 300, 21000, 8, 30, dict(budget=27000, k=0.85),
         dict(budget=21000, thresh=0.45, spike=3.6, low=0.12), dict(budget=26000, cap_mult=1.2, cap_frac=0.05)),
        (1002, 300, 20000, 9, 28, dict(budget=26000, k=0.9),
         dict(budget=22000, thresh=0.42, spike=3.8, low=0.10), dict(budget=25000, cap_mult=1.15, cap_frac=0.045)),
        (1004, 300, 22000, 8, 32, dict(budget=25000, k=0.95),
         dict(budget=23000, thresh=0.44, spike=3.5, low=0.09), dict(budget=27000, cap_mult=1.1, cap_frac=0.04)),
        (1008, 300, 23000, 9, 34, dict(budget=24000, k=0.92),
         dict(budget=24000, thresh=0.4, spike=3.9, low=0.08), dict(budget=26000, cap_mult=1.15, cap_frac=0.04)),
        (1009, 300, 24000, 10, 34, dict(budget=30000, k=0.9),
         dict(budget=25000, thresh=0.42, spike=3.7, low=0.09), dict(budget=29000, cap_mult=1.2, cap_frac=0.045)),
        # -- CONTROL sessions: few/no decoys, a mild sniper -- honest pacing is
        #    already close to competitive, so the bluff advantage is smaller.
        (1003, 300, 19000, 1, 12, dict(budget=29000, k=0.8),
         dict(budget=20000, thresh=0.7, spike=1.7, low=0.15), dict(budget=24000, cap_mult=1.3, cap_frac=0.06)),
        (1005, 300, 18000, 1, 10, dict(budget=24000, k=0.9),
         dict(budget=16000, thresh=0.72, spike=1.5, low=0.2), dict(budget=22000, cap_mult=1.25, cap_frac=0.07)),
        (1006, 300, 20500, 2, 14, dict(budget=27500, k=0.88),
         dict(budget=25000, thresh=0.68, spike=1.8, low=0.11), dict(budget=25500, cap_mult=1.2, cap_frac=0.05)),
        (1007, 300, 17000, 1, 8, dict(budget=22000, k=0.95),
         dict(budget=12000, thresh=0.75, spike=1.4, low=0.25), dict(budget=20000, cap_mult=1.4, cap_frac=0.09)),
        (1010, 300, 21500, 2, 12, dict(budget=25000, k=0.86),
         dict(budget=18000, thresh=0.7, spike=1.6, low=0.16), dict(budget=23000, cap_mult=1.25, cap_frac=0.06)),
    ]
    out = []
    for i, (seed, n_lots, bud, nd, ns, pac, sni, cap) in enumerate(specs):
        out.append(_make_instance(seed, n_lots, bud, nd, ns, pac, sni, cap, f"session{seed}"))
    return out


# ----------------------------- simulation -----------------------------------
def _simulate(inst, bids):
    """Replay the whole session with the buyer committed to `bids` (validated,
    finite, non-negative, length n_lots). Return buyer utility, or None if a
    bid the buyer WINS cannot be covered by its remaining budget (unsafe
    bluff -> caller must invalidate the whole instance)."""
    n_lots = inst["n_lots"]
    lots = inst["lots"]
    opps = inst["opponents"]  # [pacer, sniper, capper] fixed order
    catalogue_max = max(l["estimate"] for l in lots)
    rem = {"pacer": opps[0]["budget"], "sniper": opps[1]["budget"], "capper": opps[2]["budget"]}
    buyer_rem = inst["buyer_budget"]
    utility = 0.0
    for i in range(n_lots):
        remaining_lots = n_lots - i
        E, v = lots[i]["estimate"], lots[i]["value"]
        bp = _opp_bid("pacer", {"k": opps[0]["k"]}, E, rem["pacer"], catalogue_max, remaining_lots)
        bs = _opp_bid("sniper", {"thresh": opps[1]["thresh"], "spike": opps[1]["spike"], "low": opps[1]["low"]},
                      E, rem["sniper"], catalogue_max, remaining_lots)
        bc = _opp_bid("capper", {"cap_mult": opps[2]["cap_mult"], "cap_frac": opps[2]["cap_frac"],
                                  "init_budget": opps[2]["budget"]}, E, rem["capper"], catalogue_max, remaining_lots)
        bb = bids[i]
        entries = [("pacer", bp, 0), ("sniper", bs, 1), ("capper", bc, 2), ("buyer", bb, 3)]
        entries.sort(key=lambda t: (-t[1], t[2]))
        winner, price = entries[0][0], entries[1][1]
        if winner == "buyer":
            if price > buyer_rem + EPS:
                return None
            buyer_rem -= price
            utility += (v - price)
        else:
            rem[winner] -= price
    return utility


def _validate_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    bids = answer.get("bids")
    n_lots = inst["n_lots"]
    if not isinstance(bids, list) or len(bids) != n_lots:
        return None
    out = []
    for x in bids:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        fx = float(x)
        if fx != fx or fx in (float("inf"), float("-inf")) or fx < 0.0:
            return None
        out.append(fx)
    return out


def score(inst, answer):
    bids = _validate_answer(inst, answer)
    if bids is None:
        return False, 0.0
    util = _simulate(inst, bids)
    if util is None:
        return False, 0.0
    return True, util


def baseline(inst):
    """Trivial-construction reference the evaluator computes itself: bid
    nothing on every lot (utility 0)."""
    return 0.0


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        u_ref = sum(l["value"] for l in inst["lots"])
        if u_ref < 1e-9:
            u_ref = 1e-9
        public = {"name": inst["name"], "n_lots": inst["n_lots"],
                  "buyer_budget": inst["buyer_budget"], "lots": inst["lots"],
                  "opponents": inst["opponents"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj / u_ref)
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
