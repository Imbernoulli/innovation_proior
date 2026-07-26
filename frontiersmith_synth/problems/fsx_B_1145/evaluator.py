#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_1145 -- "One Guide, Three Kinds of Mountains".

Family: adaptive-rewrite-strategist. A candidate program receives a public instance
(a TERM -- a list of integer symbols -- plus a REWRITE RULESET: MERGE rules that fuse
an adjacent pair of symbols into one (length -1), and SPLIT rules that blow a single
symbol into an adjacent pair (length +1) -- and a move BUDGET). It outputs a sequence
of moves; the evaluator REPLAYS that sequence deterministically and scores the final
term length (mean ratio to a do-nothing baseline over 10 seeded instances).

The 10 instances are drawn from a frozen mixture of three planted landscape types
(mechanism: landscape-type-classification):
  - MONOTONE ("gentle slope"): only unambiguous same-symbol self-collapse merges are
    present. Any legal merge, applied in any order, drives the term down to its unique
    fully-reduced form. A single leftmost greedy pass is already optimal here.
  - CANYON ("descend before the summit"): the term contains "locked" pockets --
    color-run, LOCK symbol, color-run -- where the lock symbol matches no merge rule at
    all, only a SPLIT rule that turns it into a same-colored pair. Splitting the lock
    temporarily *grows* the term, but fuses the two flanking runs into one long run that
    then collapses far below what merge-only play can ever reach. A rule that never
    accepts a size-increasing move gets permanently stuck at the pre-split floor.
  - BRAIDED ("two guides, one trail"): some adjacent symbol pairs match TWO different
    merge rules at once (tagged "family" 0 and 1) with different outputs. Picking the
    family-0 rule looks equally good *this step* (both consume the same two symbols) but
    is a dead end; picking family-1 opens a bridge symbol that keeps absorbing the rest
    of the chain and finally reacts with a trailing symbol for one extra collapse. A
    "first rule in the list wins" recipe silently always takes the dead-end branch.

No single fixed recipe is near-optimal on all three (mechanism: per-instance-strategy-
dispatch) -- the evaluator rewards spending a little cheap analysis diagnosing which
landscape an instance is, THEN dispatching a tailored strategy, over recombining one
canned rule.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via `isorun`,
sees ONLY the public instance on stdin, and returns ONLY its answer on stdout.

Scoring (deterministic; no wall-time):
  baseline b = length of the ORIGINAL term (the "do nothing" / empty-move-sequence cost;
               always feasible).
  For a FEASIBLE move sequence producing a final term of length obj:
      r = min(1, 0.1 * b / obj)
  -> doing nothing maps to exactly 0.1; a final term k times shorter than the original
     maps to min(1, 0.1*k). Infeasible / malformed answers -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _gen_monotone(seed, n, k, repeat_p):
    """Unambiguous self-collapse system: merges = {(s,s)->s for s in 0..k-1}, no splits.
    Confluent -- any order of legal merges reaches the same fully-reduced term (one
    symbol per maximal run). `repeat_p` (0..99) controls how often the next symbol
    repeats the previous one (building runs worth collapsing)."""
    r = _rng(seed)
    term = [r(0, k - 1)]
    for _ in range(1, n):
        if r(0, 99) < repeat_p:
            term.append(term[-1])
        else:
            choices = [s for s in range(k) if s != term[-1]]
            term.append(choices[r(0, len(choices) - 1)])
    merges = [{"a": s, "b": s, "c": s, "family": 0} for s in range(k)]
    return {"n": len(term), "term": term, "budget": 2 * len(term) + 10,
            "merges": merges, "splits": []}


def _gen_canyon(seed, pockets, run_lo, run_hi):
    """`pockets` locked pockets, each: color-run, LOCK, color-run (same color both sides).
    Merges only self-collapse each color; the LOCK matches no merge, only its own split
    rule LOCK -> (color, color). Splitting first, then collapsing, fuses both runs plus
    the two new symbols into one long same-color run -> collapses to a single symbol.
    Never splitting leaves the term stuck at "color LOCK color" (length 3) per pocket."""
    r = _rng(seed)
    term = []
    merges = []
    splits = []
    for p in range(pockets):
        color = 2 * p
        lock = 2 * p + 1
        r1 = r(run_lo, run_hi)
        r2 = r(run_lo, run_hi)
        term += [color] * r1 + [lock] + [color] * r2
        merges.append({"a": color, "b": color, "c": color, "family": 0})
        splits.append({"a": lock, "b": color, "c": color})
    return {"n": len(term), "term": term, "budget": 2 * len(term) + 10,
            "merges": merges, "splits": splits}


def _gen_braided(seed, chains, tail_lo, tail_hi):
    """`chains` independent chains "A B B..B E" (tail B's). Two rules match (A,B): a
    family-0 dead-end (A,B)->A (keeps absorbing B's alone, then stuck against E), and a
    family-1 bridge (A,B)->C. Only the bridge symbol C keeps absorbing B's (C,B)->C AND
    can finally react with the trailing E via (C,E)->Z for one extra collapse. Whichever
    of the two same-pattern rules is picked at the FIRST (A,B) in a chain determines
    whether that chain bottoms out at length 2 (dead end) or length 1 (bridge)."""
    r = _rng(seed)
    term = []
    merges = []
    for c in range(chains):
        base = 5 * c
        A, B, E, C, Z = base, base + 1, base + 2, base + 3, base + 4
        tail = r(tail_lo, tail_hi)
        term += [A] + [B] * tail + [E]
        merges.append({"a": A, "b": B, "c": A, "family": 0})   # dead end (listed FIRST)
        merges.append({"a": A, "b": B, "c": C, "family": 1})   # bridge (listed SECOND)
        merges.append({"a": C, "b": B, "c": C, "family": 1})   # bridge keeps absorbing
        merges.append({"a": C, "b": E, "c": Z, "family": 1})   # bridge unlocks the tail
    return {"n": len(term), "term": term, "budget": 2 * len(term) + 10,
            "merges": merges, "splits": []}


_SPECS = [
    ("monotone", _gen_monotone, dict(seed=101, n=24, k=4, repeat_p=55)),
    ("monotone", _gen_monotone, dict(seed=102, n=32, k=5, repeat_p=45)),
    ("monotone", _gen_monotone, dict(seed=103, n=46, k=4, repeat_p=60)),   # held-out, larger
    ("canyon",   _gen_canyon,   dict(seed=201, pockets=3, run_lo=2, run_hi=4)),
    ("canyon",   _gen_canyon,   dict(seed=202, pockets=4, run_lo=2, run_hi=5)),
    ("canyon",   _gen_canyon,   dict(seed=203, pockets=6, run_lo=3, run_hi=6)),  # held-out, larger
    ("braided",  _gen_braided,  dict(seed=301, chains=4, tail_lo=2, tail_hi=4)),
    ("braided",  _gen_braided,  dict(seed=302, chains=4, tail_lo=3, tail_hi=5)),
    ("braided",  _gen_braided,  dict(seed=303, chains=5, tail_lo=2, tail_hi=6)),
    ("braided",  _gen_braided,  dict(seed=304, chains=7, tail_lo=3, tail_hi=6)),  # held-out, larger
]


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}]."""
    out = []
    for _kind, fn, kwargs in _SPECS:
        pub = fn(**kwargs)
        out.append({"public": pub, "hidden": {}})
    return out


# ----------------------------- replay / scoring -----------------------------
def _coerce_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x) or abs(x - round(x)) > 1e-9:
            return None
        return int(round(x))
    return None


def replay(pub, moves):
    """Apply `moves` to the term in order. Returns (ok, final_state_list_or_None)."""
    state = list(pub["term"])
    merges = pub["merges"]
    splits = pub["splits"]
    budget = pub["budget"]
    if not isinstance(moves, list) or len(moves) > budget:
        return False, None
    for mv in moves:
        if not isinstance(mv, dict):
            return False, None
        op = mv.get("op")
        if op not in ("merge", "split"):
            return False, None
        pos = _coerce_int(mv.get("pos"))
        rule = _coerce_int(mv.get("rule"))
        if pos is None or rule is None:
            return False, None
        if op == "merge":
            if not (0 <= rule < len(merges)):
                return False, None
            if not (0 <= pos <= len(state) - 2):
                return False, None
            r = merges[rule]
            if state[pos] != r["a"] or state[pos + 1] != r["b"]:
                return False, None
            state = state[:pos] + [r["c"]] + state[pos + 2:]
        else:
            if not (0 <= rule < len(splits)):
                return False, None
            if not (0 <= pos <= len(state) - 1):
                return False, None
            r = splits[rule]
            if state[pos] != r["a"]:
                return False, None
            state = state[:pos] + [r["b"], r["c"]] + state[pos + 1:]
    return True, state


def baseline(inst):
    """Cost of doing nothing (empty move sequence) -- the original term's length."""
    return len(inst["public"]["term"])


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    if not isinstance(answer, dict):
        return False, None
    moves = answer.get("moves", None)
    ok, state = replay(pub, moves)
    if not ok or state is None:
        return False, None
    obj = len(state)
    if obj < 1 or not math.isfinite(obj):
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
