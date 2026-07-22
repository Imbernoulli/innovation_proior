#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0932 -- "Ten Rounds to Close: Multi-Supplier Procurement
Under Timed Concessions" (family: concession-timed-bargaining; format B, quality-metric).

THEME.  You are the buyer for a procurement desk that must source a total quantity Q
of one commodity across at most T negotiation rounds, sourcing from M seeded suppliers
plus one always-available "outside option" (a backstop spot-market source).

Each supplier i posts an opening ask p0_i and will never go below a floor ask
pfloor_i. Its CURRENT ask depends on a concession level c_i in [0,1]:

    ask_i(c_i) = pfloor_i + (p0_i - pfloor_i) * (1 - c_i)          # c_i=0 -> p0_i, c_i=1 -> pfloor_i

Each round you take exactly ONE action:
  * "negotiate" a chosen supplier i (before its own deadline_i) -- this ADVANCES its
    concession: c_i += base_step_i, boosted by soften_mult_i (softens: the concession
    step is multiplied by soften_mult_i) whenever the round is within window_i rounds
    of that supplier's OWN deadline_i. You may optionally buy `qty` units from i in the
    SAME round at the resulting ask, up to i's remaining capacity cap_i and your
    remaining need.
  * "outside": buy `qty` units instantly, no negotiation, at the outside price
    outside_price(t) = outside0 * outside_growth^(t-1)  (outside_growth > 1, so the
    outside option DECAYS -- it only gets worse the longer you wait).
  * "wait": do nothing.

HARDENING (the trap the statement warns about honestly): any supplier you do NOT
negotiate with this round, that you HAD previously engaged (c_i > 0), REGRESSES --
it hardens because you walked away: c_i -= harden_step_i (floored at 0). Chasing
whichever supplier currently quotes the best price, round after round, means every
OTHER supplier you have touched keeps hardening while unengaged ones sit at their
(often unappealing) opening price p0_i forever -- so a myopic "press whoever is
cheapest right now" policy never discovers suppliers whose curve SHAPE (fast step,
low floor) would have paid off with sustained commitment, and it treats the outside
option purely as a last resort instead of as a decaying resource to lock in early
for whatever shortfall you already know you will not close with suppliers.

Any need still unmet after T rounds is force-filled at a heavy penalty price
outside_price(T+1) * penalty_mult.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance -- everything above is fully known
          upfront, this is a full-information planning problem; no per-case label
          is sent, so a solution must work from the parameters, not a lookup):
            {"T": int, "Q": float, "M": int,
             "suppliers": [ {"p0","pfloor","base_step","harden_step","soften_mult",
                              "window","deadline","cap"}, ... ],   # M entries
             "outside0": float, "outside_growth": float, "penalty_mult": float}
  stdout: ONE JSON object:
            {"actions": [a_1, ..., a_k]}     # k <= T; a round-by-round action plan
          each a_t one of:
            {"type":"negotiate","supplier": int, "qty": float}   # qty optional, default 0
            {"type":"outside","qty": float}
            {"type":"wait"}
          Missing trailing rounds (k<T) are treated as "wait".

SCORING (deterministic; no wall-time).  The evaluator simulates the round loop above
in the PARENT process against the answer, computing total procurement cost. Two
references, both computed by the evaluator itself:
    cost_base = Q * outside0          (buy everything from the outside option at
                                        round 1 -- the do-nothing reference)
    cost_ub   = Q * min_i(pfloor_i)   (optimistic, generally UNREACHABLE bound: every
                                        unit at the single best floor price, ignoring
                                        capacity/deadlines/rounds-to-concede)
    r = clamp( 0.1 + 0.9 * (cost_base - cost_cand) / (cost_base - cost_ub), 0, 1 )
A do-nothing buyer scores ~0.1; the unreachable all-floor bound scores 1.0; real
plans that respect capacity/deadlines/rounds-needed stay well under 1.0 -> headroom.
Any malformed answer, invalid supplier index, non-finite/negative qty, unknown
action type, or a crash/timeout scores that instance 0.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the public instance. The references and
the simulation/validation are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def uniform(lo, hi):
        return lo + (nxt() >> 11) / float(1 << 53) * (hi - lo)

    def randint(lo, hi):
        return lo + nxt() % (hi - lo + 1)

    return uniform, randint


# ----------------------------- instance family -----------------------------
def _rand_instance(seed):
    uni, ri = _rng(seed)
    M = ri(2, 4)
    T = ri(14, 22)
    sups = []
    for _ in range(M):
        p0 = uni(85.0, 150.0)
        pfloor = max(p0 - uni(20.0, 90.0), 15.0)
        sups.append({
            "p0": round(p0, 2), "pfloor": round(pfloor, 2),
            "base_step": round(uni(0.04, 0.22), 3),
            "harden_step": round(uni(0.03, 0.25), 3),
            "soften_mult": round(uni(1.2, 4.0), 2),
            "window": ri(1, 4),
            "deadline": ri(max(6, T - 8), T),
            "cap": round(uni(25.0, 55.0), 2),
        })
    Q = round(uni(60.0, sum(s["cap"] for s in sups) * 0.9), 2)
    outside0 = round(uni(0.95, 1.15) * max(s["p0"] for s in sups) * 0.85, 2)
    outside_growth = round(uni(1.02, 1.08), 4)
    return {"name": f"rand{seed}", "T": T, "Q": Q, "M": M, "suppliers": sups,
            "outside0": outside0, "outside_growth": outside_growth, "penalty_mult": 1.5}


def _build_instances():
    insts = [
        # --- trap 1: one supplier is cheap-but-slow (small floor gap), the other is
        # expensive-to-open but concedes fast to a much lower floor -- a one-round-
        # lookahead policy locks onto the former forever and never "pays" the one
        # costlier round it would take to discover the latter's floor.
        {"name": "twin_slow_fast", "T": 16, "Q": 100.0, "M": 2, "suppliers": [
            {"p0": 100.0, "pfloor": 90.0, "base_step": 0.05, "harden_step": 0.05,
             "soften_mult": 1.5, "window": 2, "deadline": 16, "cap": 60.0},
            {"p0": 140.0, "pfloor": 40.0, "base_step": 0.25, "harden_step": 0.10,
             "soften_mult": 1.5, "window": 2, "deadline": 16, "cap": 60.0},
        ], "outside0": 130.0, "outside_growth": 1.05, "penalty_mult": 1.5},

        # --- trap 2: a third supplier only becomes attractive right at its own
        # (early) deadline via a big soften multiplier in a short window; a policy
        # that never commits ahead of time to be there for it misses the payoff.
        {"name": "soften_window_timing", "T": 14, "Q": 90.0, "M": 3, "suppliers": [
            {"p0": 95.0, "pfloor": 80.0, "base_step": 0.07, "harden_step": 0.04,
             "soften_mult": 1.2, "window": 1, "deadline": 14, "cap": 40.0},
            {"p0": 98.0, "pfloor": 78.0, "base_step": 0.06, "harden_step": 0.04,
             "soften_mult": 1.2, "window": 1, "deadline": 14, "cap": 40.0},
            {"p0": 120.0, "pfloor": 30.0, "base_step": 0.06, "harden_step": 0.15,
             "soften_mult": 6.0, "window": 2, "deadline": 8, "cap": 60.0},
        ], "outside0": 115.0, "outside_growth": 1.04, "penalty_mult": 1.5},

        # --- trap 3: two near-tied suppliers. A one-step-lookahead policy locks
        # onto whichever is marginally cheaper first and rides it to exhaustion
        # before touching the other, so it captures neither supplier's true
        # floor as well as committing to the better one from round 1 would.
        # (This particular pair does not itself trigger a walk-away/return
        # cycle for the reference greedy below -- that dynamic shows up on
        # "soften_window_timing" above, where a third supplier's deadline
        # temporarily pulls attention away from supplier 0 and back; the
        # harden_step=0.30 here still applies to and penalizes any policy that
        # DOES waver between these two, which a hardening-blind candidate
        # discovers the hard way when it deviates from either reference.)
        {"name": "harden_punish", "T": 18, "Q": 80.0, "M": 2, "suppliers": [
            {"p0": 100.0, "pfloor": 55.0, "base_step": 0.09, "harden_step": 0.30,
             "soften_mult": 1.3, "window": 2, "deadline": 18, "cap": 50.0},
            {"p0": 102.0, "pfloor": 50.0, "base_step": 0.10, "harden_step": 0.30,
             "soften_mult": 1.3, "window": 2, "deadline": 18, "cap": 50.0},
        ], "outside0": 110.0, "outside_growth": 1.03, "penalty_mult": 1.5},

        # --- trap 4: capacity-limited suppliers (cap well under Q) look
        # attractive early (their opening asks beat the outside price), so a
        # policy that presses them round after round exhausts BOTH before it
        # is ever forced to reach for the outside option -- by which point,
        # after many rounds of outside_growth compounding, the outside price
        # has grown far past even the do-nothing round-1 reference. Locking
        # the known shortfall in at round 1 (before any decay) avoids this.
        {"name": "outside_urgency", "T": 16, "Q": 100.0, "M": 2, "suppliers": [
            {"p0": 80.0, "pfloor": 70.0, "base_step": 0.06, "harden_step": 0.06,
             "soften_mult": 1.4, "window": 2, "deadline": 16, "cap": 25.0},
            {"p0": 82.0, "pfloor": 68.0, "base_step": 0.05, "harden_step": 0.06,
             "soften_mult": 1.4, "window": 2, "deadline": 16, "cap": 25.0},
        ], "outside0": 100.0, "outside_growth": 1.10, "penalty_mult": 1.6},
    ]
    for seed in (9001, 9002, 9003, 9004, 9005, 9006):
        insts.append(_rand_instance(seed))
    return insts


# ----------------------------- simulation / scoring -------------------------
def _ask(sup, c):
    return sup["pfloor"] + (sup["p0"] - sup["pfloor"]) * (1.0 - c)


def _outside_price(inst, t):
    return inst["outside0"] * (inst["outside_growth"] ** (t - 1))


def _finite_nonneg(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and \
        x not in (float("inf"), float("-inf")) and x >= 0.0


def simulate(inst, answer):
    """Validate `answer` strictly against `inst`; return total cost, or None if
    the answer is malformed/infeasible in any way (type/shape/range violation).
    ALL structural checks -- including per-round deadline legality -- happen in
    this single up-front pass over the full (possibly short) action list, so an
    illegal action can never be skipped just because the need was already met by
    an earlier round (the round-effects loop below stops early on completion,
    but validity must not depend on how far that loop actually runs)."""
    if not isinstance(answer, dict):
        return None
    actions = answer.get("actions")
    if not isinstance(actions, list):
        return None
    T = inst["T"]; M = inst["M"]; Q = inst["Q"]
    sup = inst["suppliers"]
    if len(actions) > T:
        return None
    for idx, a in enumerate(actions):
        if not isinstance(a, dict) or "type" not in a:
            return None
        t = idx + 1
        typ = a["type"]
        if typ not in ("negotiate", "outside", "wait"):
            return None
        if typ == "negotiate":
            j = a.get("supplier")
            if isinstance(j, bool) or not isinstance(j, int) or not (0 <= j < M):
                return None
            qty = a.get("qty", 0.0)
            if not _finite_nonneg(qty):
                return None
            if t > sup[j]["deadline"]:
                return None   # illegal: negotiating a supplier past its own deadline
        elif typ == "outside":
            qty = a.get("qty", 0.0)
            if not _finite_nonneg(qty):
                return None

    c = [0.0] * M
    cap_rem = [s["cap"] for s in sup]
    remaining = Q
    total_cost = 0.0
    for t in range(1, T + 1):
        if remaining <= 1e-9:
            break
        a = actions[t - 1] if t - 1 < len(actions) else {"type": "wait"}
        focus = None
        typ = a["type"]
        if typ == "negotiate":
            j = a["supplier"]
            qty_req = float(a.get("qty", 0.0))
            # (deadline legality already verified above for every listed action)
            focus = j
            near = (sup[j]["deadline"] - t) < sup[j]["window"]
            step = sup[j]["base_step"] * (sup[j]["soften_mult"] if near else 1.0)
            c[j] = min(1.0, c[j] + step)
            price = _ask(sup[j], c[j])
            if qty_req > 0:
                q = min(qty_req, cap_rem[j], remaining)
                if q > 0:
                    total_cost += q * price
                    remaining -= q
                    cap_rem[j] -= q
        elif typ == "outside":
            qty_req = float(a.get("qty", 0.0))
            price = _outside_price(inst, t)
            q = min(qty_req, remaining)
            if q > 0:
                total_cost += q * price
                remaining -= q
        for j in range(M):
            if j != focus and c[j] > 0:
                c[j] = max(0.0, c[j] - sup[j]["harden_step"])

    if remaining > 1e-9:
        penalty = _outside_price(inst, T + 1) * inst["penalty_mult"]
        total_cost += remaining * penalty
        remaining = 0.0
    return total_cost


def _cost_base(inst):
    return inst["Q"] * inst["outside0"]


def _cost_ub(inst):
    return inst["Q"] * min(s["pfloor"] for s in inst["suppliers"])


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        # NOTE: deliberately do NOT send inst["name"] to the candidate -- it is an
        # internal bookkeeping label for the 10 fixed instances used to reproduce
        # this docstring's design discussion, and leaking it would let a candidate
        # branch on a per-case identifier instead of solving the instance from its
        # actual (T, Q, suppliers, outside...) parameters.
        public = {"T": inst["T"], "Q": inst["Q"], "M": inst["M"],
                  "suppliers": [dict(s) for s in inst["suppliers"]],
                  "outside0": inst["outside0"], "outside_growth": inst["outside_growth"],
                  "penalty_mult": inst["penalty_mult"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            cost = simulate(inst, ans)
        except Exception:
            cost = None
        if cost is None:
            vec.append(0.0)
            continue
        cb = _cost_base(inst)
        cu = _cost_ub(inst)
        denom = max(cb - cu, 1e-9)
        r = 0.1 + 0.9 * (cb - cost) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
