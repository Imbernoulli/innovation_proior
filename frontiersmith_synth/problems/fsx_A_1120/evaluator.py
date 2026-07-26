#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1120 -- "Reading-Hall Carousel Reshelving"
(family: wreath-macro-sorting-policy; format B, quality-metric).

THEME.  A round reading hall has K book carousels at physical bays 0..K-1;
each carousel has M shelf pockets 0..M-1 holding one book each (N = K*M books,
ids 0..N-1, book `id`'s home = carousel id//M, pocket id%M).  Only two rigid,
LINKED motions exist:
  - H+/H-  : the shared axle rotates the WHOLE ring of K carousels by one bay
             (every carousel keeps its own internal pocket order -- it just
             relocates).  This is the single global generator of the "outer"
             wreath-product factor (order K).
  - S{k}+/S{k}- (k=0..K-1): the belt inside whichever carousel currently sits
             at physical bay k advances that carousel's M pockets by one step.
             Only bay k is touched.  This is the "inner" factor (order M) --
             one independent copy per bay.

The reachable group is exactly the wreath product Z_M wr Z_K acting
imprimitively on the N books, with the K size-M "which-carousel-is-this"
sets as the (hidden but statement-visible) block system.  Because H is a
SINGLE global scalar (rotating every carousel at once) and each S{k} only
ever touches its own bay, the two levels never interfere with each other or
with themselves once corrected -- this is a genuine (if compact) two-level
Schreier-Sims stabilizer chain: fix the coset representative for the
hall-permutation quotient first (a book's id//M identifies which carousel it
truly belongs to, wherever it currently sits), then, independently per bay,
fix the pocket rotation (id%M).  A policy that reads off id//M and id%M
directly needs only O(K + N) moves; one that instead assumes "bay k already
holds carousel k" wastes a full M-cycle every time that assumption is wrong.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "K": int, "M": int, "N": K*M,
             "state": [book_id at physical slot 0, ..., N-1]}
          physical slot i = bay*M + pocket, bay = i//M, pocket = i%M.
  stdout: ONE JSON object:
            {"moves": [mv_0, mv_1, ...]}
          each mv is "H+", "H-", or "S<k><sign>" (0<=k<K, sign in "+-").

  A move list is VALID iff every entry matches this grammar (k in range) and,
  after applying all moves in order to the instance's `state`, slot i holds
  book i for every i (full restore).  Malformed entries, wrong types, a
  crash, a timeout, non-JSON output, or a state that isn't fully solved at
  the end -> that instance scores 0.0.  There is a generous move-count cap
  (50000) purely as an anti-DoS guard on the evaluator's own simulation.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, purely from `state` (never from the candidate's answer):
    q_base  = moves used by an internal WEAK reference policy ("assume bay k
              already holds carousel k; spin its shelf forward; if a full
              revolution doesn't match, nudge the hall once and move on") --
              always valid, always terminates, never reads the id//M stamp
              to jump straight to the right bay.
    q_ideal = 0.5 * true_min, where true_min = (minimal hall distance) +
              sum over bays of (minimal per-bay shelf distance) is the exact,
              provably optimal move count for the instance (each level is an
              independent degree of freedom, so this is a real lower bound,
              not merely an estimate) -- q_ideal is deliberately set BELOW
              that true optimum so even the optimal solver has headroom.
    q_cand  = moves used by the candidate's (validated) answer.
  and normalize with an affine anchor (weak baseline -> 0.1, optimistic ideal
  -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ideal), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  q_base and
q_ideal are computed by THIS parent process directly from the ground-truth
`state`, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, re
import isorun

MOVE_RE = re.compile(r"^S(\d+)([+-])$")
MAX_MOVES = 50000


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- core move engine ------------------------------
def _apply_move(state, K, M, mv):
    """Apply one legal move to `state` (list of length K*M) and return the
    new state.  `mv` MUST already be validated by _parse_move."""
    ns = state[:]
    if mv == "H+":
        for k in range(K):
            base_new = ((k + 1) % K) * M
            base_old = k * M
            for p in range(M):
                ns[base_new + p] = state[base_old + p]
    elif mv == "H-":
        for k in range(K):
            base_new = k * M
            base_old = ((k + 1) % K) * M
            for p in range(M):
                ns[base_new + p] = state[base_old + p]
    else:
        k = int(mv[1:-1])
        base = k * M
        if mv[-1] == '+':
            for p in range(M):
                ns[base + (p + 1) % M] = state[base + p]
        else:
            for p in range(M):
                ns[base + p] = state[base + (p + 1) % M]
    return ns


def _parse_move(mv, K):
    if not isinstance(mv, str):
        return None
    if mv == "H+" or mv == "H-":
        return mv
    m = MOVE_RE.match(mv)
    if not m:
        return None
    k = int(m.group(1))
    if k >= K:
        return None
    return mv


# ----------------------------- instance family -------------------------------
def _build_instance(seed, K, M, tag):
    N = K * M
    ni = _rng(seed)
    e = ni(1, K - 1) if K > 1 else 0
    rs = [ni(0, M - 1) for _ in range(K)]
    state = list(range(N))
    for _ in range(e):
        state = _apply_move(state, K, M, "H+")
    for k in range(K):
        for _ in range(rs[k]):
            state = _apply_move(state, K, M, f"S{k}+")
    return {"name": f"hall{seed}", "K": K, "M": M, "N": N, "state": state, "tag": tag}


def _build_instances():
    specs = [
        (201, 5, 5, "small-A"),
        (202, 6, 7, "small-B"),
        (203, 7, 6, "mid-A"),
        (204, 8, 8, "mid-B"),
        (205, 6, 10, "shelf-heavy-A"),
        (206, 10, 6, "hall-heavy-A"),
        (207, 9, 9, "mid-C"),
        (208, 11, 9, "held-out-large-A"),
        (209, 8, 12, "shelf-heavy-B"),
        (210, 13, 10, "held-out-large-scale"),
    ]
    return [_build_instance(seed, K, M, tag) for seed, K, M, tag in specs]


# ----------------------------- references -------------------------------------
def _naive_conflate_cost(state, K, M):
    """Weak internal reference: assumes bay k already holds carousel k, spins
    its shelf forward hoping to match; if a full M-cycle proves that guess
    wrong, nudges the hall once and re-tries.  Always valid & terminating."""
    N = K * M
    s = state[:]
    moves = 0
    cap = MAX_MOVES
    for i in range(N):
        target_bay = i // M
        hall_tries = 0
        while s[i] != i:
            if moves > cap:
                return None
            found = False
            for _ in range(M):
                if s[i] == i:
                    found = True
                    break
                s = _apply_move(s, K, M, f"S{target_bay}+")
                moves += 1
                if moves > cap:
                    return None
            if found:
                break
            s = _apply_move(s, K, M, "H+")
            moves += 1
            hall_tries += 1
            if hall_tries > K + 2:
                return None
    return moves if s == list(range(N)) else None


def _true_min_cost(state, K, M):
    """Exact optimal move count.  Hall offset and every bay's shelf offset
    are each an independent scalar (H is a single global generator; S{k}
    only ever touches bay k), so the minimal per-level distances sum to a
    real, tight lower bound -- and this policy achieves it exactly."""
    X = state[0] // M
    hall = min(X, K - X)
    shelf = 0
    for k in range(K):
        loc0 = state[k * M] % M
        shelf += min(loc0, M - loc0)
    return hall + shelf


# ----------------------------- validation --------------------------------------
def _score_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    raw_moves = answer.get("moves")
    if not isinstance(raw_moves, list):
        return None
    if len(raw_moves) > MAX_MOVES:
        return None
    K, M, N = inst["K"], inst["M"], inst["N"]
    parsed = []
    for mv in raw_moves:
        p = _parse_move(mv, K)
        if p is None:
            return None
        parsed.append(p)
    s = inst["state"][:]
    for mv in parsed:
        s = _apply_move(s, K, M, mv)
    if s != list(range(N)):
        return None
    return len(parsed)


# ----------------------------- scoring driver -----------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        K, M, N = inst["K"], inst["M"], inst["N"]
        q_base = _naive_conflate_cost(inst["state"], K, M)
        true_min = _true_min_cost(inst["state"], K, M)
        q_ideal = 0.5 * true_min
        if q_base is None or q_base <= q_ideal:
            vec.append(0.0)
            continue
        public = {"name": inst["name"], "K": K, "M": M, "N": N, "state": list(inst["state"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _score_answer(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        denom = max(1e-9, q_base - q_ideal)
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
