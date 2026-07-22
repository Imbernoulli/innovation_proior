#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0960 -- "Commit-Cut: Streaming Panel Orders on a Kerf-Locked Sheet"
(family: commit-cut-stream-policy; format B, quality-metric).

THEME.  A sheet-metal / panel shop receives orders one at a time.  Each order is a
rectangular panel (w x h) with a value.  The shop must decide IMMEDIATELY, in arrival
order, and IRREVOCABLY: either cut the panel out of the sheet right now, or refuse it
(a small fixed forfeiture cost -- the "option" the shop paid to keep flexibility for
later, higher-value orders). Once a panel is cut, the saw needs KERF CLEARANCE `k`
around every side of the cut that is not already the sheet's outer edge (an edge cut
needs no clearance; an interior cut through the middle of remaining stock does) -- and
because the cut is permanent, that clearance is PERMANENTLY unusable, exactly like the
placed panel itself. There is no combining, no undo, no re-cutting.

This composes three mechanisms into one objective:
  - irrevocable-online-placement: decisions are emitted positionally in arrival order
    and validated by REPLAYING them causally against a single mutable sheet -- an
    answer that is only legal "if you could go back and redo it" is rejected outright.
  - kerf-clearance-geometry: every accepted cut locks its footprint *plus* a k-wide
    interior buffer (clipped at the sheet's true edges), so two touching pieces always
    burn 2k of dead space between them unless one of them hugs the sheet boundary.
  - discard-option-valuation: refusing an order is not free (fixed forfeiture cost)
    but not catastrophic either -- it is a genuine tool for preserving contiguous
    room for a value-dense arrival you can already see is coming (the FULL arrival
    list is given up front, as is standard for this corpus's Format-B contract; the
    difficulty is geometric/combinatorial, not informational).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance -- no name/seed identifier is sent,
          only raw geometry, so a candidate cannot special-case a known id):
            {"W": int, "H": int, "k": int, "discard_cost": float,
             "n": int, "items": [{"w":int,"h":int,"value":float}, ...]}   # arrival order
  stdout: ONE JSON object:
            {"decisions": [ {"action":"place","x":int,"y":int,"rot":0|1}
                             | {"action":"discard"}, ... ]}   # length n, positional
          rot=0 keeps (w,h); rot=1 uses (h,w).

  VALIDITY (replayed causally against a fresh W x H grid, item by item, in order):
  a "place" decision is legal iff its (possibly rotated) footprint lies in bounds and
  its footprint-plus-kerf-buffer (clipped to the sheet) does not overlap any cell
  already locked by an earlier decision; on acceptance BOTH the footprint and its
  clipped buffer become permanently locked.  ANY illegal "place" (out of bounds,
  overlap, bad types, rot not in {0,1}), a malformed `decisions` list (wrong length,
  wrong length, unknown action), a crash, a timeout, or non-JSON output makes the
  WHOLE instance score 0.0 -- a submitted cutting plan is either entirely executable
  or it is not a real plan. A "discard" simply forfeits `discard_cost` and moves on.

SCORING (deterministic; no wall-time).  Per instance:
    gained = sum(value of placed items) - discard_cost * (# discarded items)
    relax  = sum(value_i for i where the panel fits SOME orientation on an *empty*
                 sheet, i.e. (w<=W and h<=H) or (h<=W and w<=H))
  `relax` is a valid (necessary-condition) upper bound: no order that cannot fit on an
  empty sheet by itself can ever be accepted under any policy whatsoever.  It is also
  deliberately loose: it ignores that panels compete for the same sheet, that kerf
  eats real space between neighbours, and that the arrival ORDER makes some subsets of
  otherwise-compatible panels mutually unreachable.  Score = clamp(gained/relax, 0, 1),
  averaged over 10 fixed seeded instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  `relax` and all
grading are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def make_rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def randint(lo, hi):
        r = nxt()
        return lo + (r >> 17) % (hi - lo + 1)

    def rand01():
        r = nxt()
        return (r >> 11) / (1 << 53)

    return randint, rand01


# ----------------------------- arrival mixture ------------------------------
# name: (w_lo, w_hi, h_lo, h_hi, density_lo, density_hi); value = w*h*density
CATS = {
    "filler": (1, 2, 3, 4, 0.9, 1.2),   # plentiful, cheap, elongated (rotation matters)
    "decoy":  (2, 3, 5, 7, 1.2, 1.6),   # tempting mid-value, still not the point
    "tail":   (4, 6, 8, 10, 4.2, 5.0),  # rare, value-dense -- the arrivals worth a corridor
}


def gen_item(randint, rand01, cat):
    wlo, whi, hlo, hhi, dlo, dhi = CATS[cat]
    w = randint(wlo, whi)
    h = randint(hlo, hhi)
    dens = dlo + rand01() * (dhi - dlo)
    value = round(w * h * dens, 2)
    return {"w": w, "h": h, "value": value}


def _reorder(items, cats, order_mode):
    if order_mode == "trap":
        head = [it for it, c in zip(items, cats) if c != "tail"]
        tail = [it for it, c in zip(items, cats) if c == "tail"]
        return head + tail
    if order_mode == "reverse_trap":
        head = [it for it, c in zip(items, cats) if c != "tail"]
        tail = [it for it, c in zip(items, cats) if c == "tail"]
        return tail + head
    return items  # "mixed": keep sampled arrival order as-is


INSTANCE_SPECS = [
    # seed, W,  H,  k, n,  mix (filler,decoy,tail probs),                    order_mode,     discard_cost
    (101, 22, 16, 1, 18, [("filler", .55), ("decoy", .27), ("tail", .18)], "trap",         0.50),
    (102, 22, 16, 1, 20, [("filler", .55), ("decoy", .27), ("tail", .18)], "trap",         0.35),
    (103, 22, 16, 1, 18, [("filler", .50), ("decoy", .30), ("tail", .20)], "trap",         0.50),
    (104, 22, 16, 1, 18, [("filler", .55), ("decoy", .27), ("tail", .18)], "mixed",        0.50),
    (105, 22, 16, 1, 18, [("filler", .50), ("decoy", .30), ("tail", .20)], "mixed",        0.40),
    (106, 22, 14, 1, 16, [("filler", .45), ("decoy", .35), ("tail", .20)], "mixed",        0.55),
    (107, 22, 16, 1, 24, [("filler", .55), ("decoy", .25), ("tail", .20)], "trap",         0.35),
    (108, 24, 14, 1, 20, [("filler", .40), ("decoy", .35), ("tail", .25)], "mixed",        0.50),
    (109, 26, 14, 1, 24, [("filler", .55), ("decoy", .25), ("tail", .20)], "trap",         0.40),
    (110, 22, 18, 1, 22, [("filler", .35), ("decoy", .35), ("tail", .30)], "reverse_trap", 0.50),
]


def build_instances():
    out = []
    for seed, W, H, k, n, mix, order_mode, discard_cost in INSTANCE_SPECS:
        randint, rand01 = make_rng(seed)
        cats = []
        for _ in range(n):
            r = rand01()
            c = 0.0
            chosen = mix[-1][0]
            for name, p in mix:
                c += p
                if r <= c:
                    chosen = name
                    break
            cats.append(chosen)
        items = [gen_item(randint, rand01, c) for c in cats]
        items = _reorder(items, cats, order_mode)
        out.append({"name": f"panel{seed}", "W": W, "H": H, "k": k, "n": n,
                    "discard_cost": discard_cost, "items": items})
    return out


# ----------------------------- geometry -------------------------------------
def make_grid(W, H):
    return [[False] * W for _ in range(H)]


def legal(grid, W, H, k, x, y, dw, dh):
    if x < 0 or y < 0 or dw <= 0 or dh <= 0 or x + dw > W or y + dh > H:
        return False
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            if row[xx]:
                return False
    return True


def commit(grid, W, H, k, x, y, dw, dh):
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            row[xx] = True


def relax_bound(inst):
    W, H = inst["W"], inst["H"]
    tot = 0.0
    for it in inst["items"]:
        w, h = it["w"], it["h"]
        if (w <= W and h <= H) or (h <= W and w <= H):
            tot += it["value"]
    return tot


# ----------------------------- validation -----------------------------------
def score_answer(inst, answer):
    """Replay `answer["decisions"]` causally. Return (ok, gained)."""
    if not isinstance(answer, dict):
        return False, 0.0
    decisions = answer.get("decisions")
    if not isinstance(decisions, list):
        return False, 0.0
    W, H, k = inst["W"], inst["H"], inst["k"]
    items = inst["items"]
    n = len(items)
    if len(decisions) != n:
        return False, 0.0
    grid = make_grid(W, H)
    gained = 0.0
    for it, dec in zip(items, decisions):
        if not isinstance(dec, dict):
            return False, 0.0
        act = dec.get("action")
        if act == "discard":
            gained -= inst["discard_cost"]
            continue
        if act != "place":
            return False, 0.0
        x, y, rot = dec.get("x"), dec.get("y"), dec.get("rot")
        if isinstance(x, bool) or isinstance(y, bool) or isinstance(rot, bool):
            return False, 0.0
        if not (isinstance(x, int) and isinstance(y, int) and isinstance(rot, int)):
            return False, 0.0
        if rot not in (0, 1):
            return False, 0.0
        dw, dh = (it["w"], it["h"]) if rot == 0 else (it["h"], it["w"])
        if not legal(grid, W, H, k, x, y, dw, dh):
            return False, 0.0
        commit(grid, W, H, k, x, y, dw, dh)
        gained += it["value"]
    return True, gained


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = build_instances()

    vec = []
    for inst in instances:
        # NOTE: no "name"/seed identifier is sent -- a candidate sees only the raw
        # geometry+value stream, so a policy cannot special-case these 10 fixed
        # instances via a cheap string/id lookup; it must actually read `items`.
        public = {"W": inst["W"], "H": inst["H"], "k": inst["k"],
                  "discard_cost": inst["discard_cost"], "n": inst["n"],
                  "items": [{"w": it["w"], "h": it["h"], "value": it["value"]} for it in inst["items"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, gained = score_answer(inst, ans)
        except Exception:
            ok, gained = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        relax = relax_bound(inst)
        r = gained / relax if relax > 1e-9 else 0.0
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
