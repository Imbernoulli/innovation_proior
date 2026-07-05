#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0495 -- "Airwaves: A Combinatorial Spectrum Auction"
(family: new-mechanism-auction; format B, quality-metric; theme: spectrum auction).

THEME.  A regulator runs a single-round SEALED-BID COMBINATORIAL auction for a block
of spectrum licenses (an item = one licence, e.g. a (region, band) tile).  Carriers
value bundles super-additively: a contiguous multi-region footprint is worth more than
the sum of its regions (coverage synergy), so each carrier submits several PACKAGE bids
under XOR semantics -- at most ONE of a carrier's packages may win.  The mechanism must
pick a feasible set of winning packages (every licence goes to at most one carrier; each
carrier wins at most one package) and post a per-winner PRICE.  The regulator's goal is
to MAXIMIZE total economic welfare = sum of the (declared) values of the winning
packages -- this is exactly the NP-hard Winner-Determination Problem of a combinatorial
auction, skinned as an FCC-style spectrum sale.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "m": int,                      # number of licences, indexed 0..m-1
             "n_bidders": int,              # carriers, indexed 0..n_bidders-1
             "bids": [                      # flat list of package bids (XOR per bidder)
                {"bidder": int, "items": [int,...], "value": int}, ... ]}
  stdout: ONE JSON object (the allocation + pricing rule's output):
            {"win":    [j0, j1, ...],       # indices (into bids) of the WINNING packages
             "prices": [p0, p1, ...]}       # p_k = price charged to winner of bids[win[k]]

  A plan is VALID iff:
    * `win` is a list of DISTINCT integers, each a valid index into `bids`;
    * no licence appears in two winning packages (single assignment);
    * no bidder owns two winning packages (XOR / at most one package per carrier);
    * `prices` is a list of the SAME length as `win`, every price finite with
      0 <= p_k <= value(bids[win[k]])   (individual rationality: a winner never pays
      more than it bid -- non-negative utility).
  Any violation, wrong shape, a non-finite number, a crash, a timeout, or non-JSON
  output -> that instance scores 0.0.  (Prices do NOT change the welfare score; they are
  a feasibility requirement that keeps the mechanism individually rational.)

SCORING (deterministic; no wall-time).  Per instance the evaluator computes two
references ITSELF (never revealed to the candidate):
    w_base = welfare of a weak VALUE-SORTED greedy allocator (accept packages by
             descending value, skipping any that collide)          -> anchors 0.10
    w_ref  = welfare of a strong internal solver (multi-key greedy + local search
             with restarts)                                        -> anchors 0.80
  and normalizes the candidate welfare w_cand with an affine anchor:
    r = clamp( 0.10 + 0.70 * (w_cand - w_base) / max(1e-9, w_ref - w_base), 0, 1 )
  Reproducing the weak greedy scores ~0.10; matching the strong internal solver scores
  ~0.80; genuinely BEATING the internal solver pushes toward 1.0 (headroom is real --
  the WDP is NP-hard and the internal solver is not optimal); doing worse than the weak
  greedy scores < 0.10.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  Both references are
computed by THIS parent process, so a frame-walking / source-reading candidate learns
nothing it could not recompute itself, and cannot forge the score.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _sample(ni, pool, k):
    """Deterministic sample of k DISTINCT elements from list `pool` (partial F-Y)."""
    a = list(pool)
    k = min(k, len(a))
    for i in range(k):
        j = i + ni(0, len(a) - 1 - i)
        a[i], a[j] = a[j], a[i]
    return a[:k]


# ----------------------------- instance family -----------------------------
def _build_instance(seed, m, nb, fs_lo, fs_hi, pk_lo, pk_hi, syn):
    """Build one combinatorial spectrum-auction instance.

    Each carrier gets a random licence FOOTPRINT and per-licence base values, then
    submits several XOR package bids (subsets of its footprint) whose value is
    super-additive in the bundle size (coverage synergy `syn` percent per extra tile).
    The full-footprint package is always offered, creating a big blocking bid."""
    ni = _rng(seed)
    bids = []
    for b in range(nb):
        fs = ni(fs_lo, fs_hi)
        footprint = _sample(ni, range(m), fs)
        bval = {i: ni(3, 12) for i in footprint}
        packages = []
        # always offer the full footprint (max synergy)
        packages.append(list(footprint))
        npk = ni(pk_lo, pk_hi)
        for _ in range(npk):
            ss = ni(1, len(footprint))
            packages.append(_sample(ni, footprint, ss))
        for S in packages:
            base = sum(bval[i] for i in S)
            val = (base * (100 + syn * (len(S) - 1))) // 100
            if val < 1:
                val = 1
            bids.append({"bidder": b, "items": sorted(S), "value": int(val)})
    return {"name": f"auction{seed}", "m": m, "n_bidders": nb, "bids": bids}


def _build_instances():
    specs = [
        # seed,  m, nb, fs_lo, fs_hi, pk_lo, pk_hi, syn
        (2201, 12, 5, 3, 6, 3, 5, 25),
        (2202, 14, 6, 3, 6, 3, 5, 15),
        (2203, 16, 6, 4, 7, 3, 6, 30),
        (2204, 12, 7, 3, 5, 2, 4, 20),   # high contention
        (2205, 18, 6, 4, 8, 3, 6, 10),
        (2206, 15, 7, 3, 6, 3, 5, 35),
        (2207, 20, 7, 4, 8, 4, 7, 20),
        (2208, 16, 8, 3, 6, 3, 5, 25),
        # harder / larger held-out instances
        (2311, 24, 9, 4, 9, 4, 7, 25),
        (2312, 22, 8, 4, 8, 4, 8, 30),
        (2313, 26, 10, 5, 10, 5, 8, 20),
        (2314, 20, 9, 3, 7, 3, 6, 40),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- greedy / local-search references -------------
def _greedy(bids, order):
    """Greedy allocation over the given bid-index order: accept a package iff all its
    licences are still free AND its bidder has not already won.  Return (win_set, welfare)."""
    used_item = set()
    used_bidder = set()
    win = []
    welfare = 0
    for j in order:
        bd = bids[j]
        if bd["bidder"] in used_bidder:
            continue
        its = bd["items"]
        if any(i in used_item for i in its):
            continue
        used_item.update(its)
        used_bidder.add(bd["bidder"])
        win.append(j)
        welfare += bd["value"]
    return win, welfare


def _order_by(bids, key):
    idx = list(range(len(bids)))
    if key == "value":
        idx.sort(key=lambda j: (-bids[j]["value"], j))
    elif key == "density":
        idx.sort(key=lambda j: (-bids[j]["value"] / len(bids[j]["items"]), j))
    elif key == "sqrtdensity":
        idx.sort(key=lambda j: (-bids[j]["value"] / (len(bids[j]["items"]) ** 0.5), j))
    elif key == "small":
        idx.sort(key=lambda j: (len(bids[j]["items"]), -bids[j]["value"], j))
    return idx


def _welfare_of(bids, win):
    return sum(bids[j]["value"] for j in win)


def _local_search(bids, win, passes):
    """Improve a feasible winning set by (a) adding any compatible package and
    (b) 1-for-many swaps: drop the winners that conflict with a candidate package and
    add that package whenever it strictly increases welfare.  Deterministic."""
    win = list(win)
    winset = set(win)
    # item -> winning bid holding it; bidder -> winning bid
    item_owner = {}
    bidder_owner = {}
    for j in win:
        for i in bids[j]["items"]:
            item_owner[i] = j
        bidder_owner[bids[j]["bidder"]] = j

    def remove(j):
        winset.discard(j)
        for i in bids[j]["items"]:
            if item_owner.get(i) == j:
                del item_owner[i]
        if bidder_owner.get(bids[j]["bidder"]) == j:
            del bidder_owner[bids[j]["bidder"]]

    def add(j):
        winset.add(j)
        for i in bids[j]["items"]:
            item_owner[i] = j
        bidder_owner[bids[j]["bidder"]] = j

    n = len(bids)
    for _ in range(passes):
        improved = False
        for j in range(n):
            if j in winset:
                continue
            bd = bids[j]
            conflicts = set()
            ok = True
            # bidder conflict (XOR)
            bo = bidder_owner.get(bd["bidder"])
            if bo is not None:
                conflicts.add(bo)
            for i in bd["items"]:
                o = item_owner.get(i)
                if o is not None:
                    conflicts.add(o)
            gain = bd["value"] - sum(bids[c]["value"] for c in conflicts)
            if gain > 0:
                for c in list(conflicts):
                    remove(c)
                add(j)
                improved = True
        if not improved:
            break
    win = sorted(winset)
    return win, _welfare_of(bids, win)


def _base_welfare(bids):
    # weak reference: accept packages in SUBMISSION (arrival) order, no reordering
    _, w = _greedy(bids, list(range(len(bids))))
    return w


def _ref_welfare(bids):
    """Strong internal reference: best multi-key greedy start, then local search with a
    few seeded restart-perturbations.  Anchors the 0.80 point (NOT optimal -> headroom)."""
    best_win, best_w = None, -1
    for key in ("value", "density", "sqrtdensity", "small"):
        win, w = _greedy(bids, _order_by(bids, key))
        win, w = _local_search(bids, win, passes=8)
        if w > best_w:
            best_w, best_win = w, win
    # seeded restart perturbations: shuffle order, greedy, local-search, keep best
    # (deterministic seed derived from the bid values -- NO Python hash randomization)
    seed = 1315423911
    for b in bids:
        seed = (seed * 131 + b["value"] * 17 + len(b["items"])) & 0xFFFFFFFF
    ni = _rng(seed or 12345)
    n = len(bids)
    for _ in range(6):
        order = list(range(n))
        for i in range(n):
            j = i + ni(0, n - 1 - i)
            order[i], order[j] = order[j], order[i]
        win, w = _greedy(bids, order)
        win, w = _local_search(bids, win, passes=8)
        if w > best_w:
            best_w, best_win = w, win
    return best_w


# ----------------------------- validation ----------------------------------
def _score(inst, answer):
    """Validate the candidate answer. Return (ok, welfare)."""
    if not isinstance(answer, dict):
        return False, 0
    win = answer.get("win")
    prices = answer.get("prices")
    if not isinstance(win, list) or not isinstance(prices, list):
        return False, 0
    if len(win) != len(prices):
        return False, 0
    bids = inst["bids"]
    nbid = len(bids)
    seen_idx = set()
    used_item = set()
    used_bidder = set()
    welfare = 0
    for k, j in enumerate(win):
        if isinstance(j, bool) or not isinstance(j, int):
            return False, 0
        if j < 0 or j >= nbid or j in seen_idx:
            return False, 0
        seen_idx.add(j)
        bd = bids[j]
        if bd["bidder"] in used_bidder:
            return False, 0
        for i in bd["items"]:
            if i in used_item:
                return False, 0
            used_item.add(i)
        used_bidder.add(bd["bidder"])
        # price feasibility: finite, individually rational
        p = prices[k]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return False, 0
        if p != p or p in (float("inf"), float("-inf")):
            return False, 0
        if p < 0 or p > bd["value"] + 1e-9:
            return False, 0
        welfare += bd["value"]
    return True, welfare


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        bids = inst["bids"]
        w_base = _base_welfare(bids)
        w_ref = _ref_welfare(bids)
        denom = w_ref - w_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "m": inst["m"],
                  "n_bidders": inst["n_bidders"],
                  "bids": [{"bidder": b["bidder"], "items": list(b["items"]),
                            "value": b["value"]} for b in bids]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, w_cand = _score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.10 + 0.70 * (w_cand - w_base) / denom
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
