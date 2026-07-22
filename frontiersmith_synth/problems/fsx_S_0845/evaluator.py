#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0845 -- "The Bloc-Proof Ballot" (family: bloc-manipulation-voting-rule;
format B, quality-metric).

THEME.  A jurisdiction runs a 10-candidate, 100-voter ranked election.  Every voter casts a full
strict preference order over the 10 candidates.  The candidate PROGRAM under test *is the
aggregation rule itself*: it reads the full ballot set and outputs a winner (mechanism
"aggregation-rule-program").  Fairness in the abstract is not the design goal here -- there is a
PUBLISHED, deterministic manipulation generator that, for each of 10 fixed base elections and a
bloc size b in {5,10,20}, rewrites exactly b ballots (a small organized bloc) using one of three
known attack recipes (mechanism "strategic-bloc-sweep"):
  * COMPROMISE  -- b voters who sincerely back the frontrunner defect, en bloc, to bullet-vote a
    second, lower-quality contender up into contention.
  * BURY        -- b voters who sincerely rank the best candidate highly instead rank that
    candidate LAST, everything else kept in truthful relative order.
  * CYCLE-INJECT -- b voters, split into three groups as equal as possible, each unanimously impose one of the
    three cyclic rotations of the current top-3 candidates, trying to manufacture a majority
    cycle a naive tie-break can be steered through.
The submitted rule is run on the untouched election AND on all three rewrites; the score is driven
by the WORST (highest) resulting DISTORTION -- the elected winner's true social cost divided by
the true-optimal candidate's social cost, using HIDDEN per-voter utilities the rule never sees
(mechanism "distortion-ratio-scoring"). Rules that are merely accurate on sincere ballots but whose
winner swings wildly under a b-ballot rewrite of these specific sweep profiles score LOW; rules
whose winner has bounded sensitivity to the sweep score HIGH.  This is deliberately NOT "design a
fair rule in the abstract" -- it is "minimize sensitivity of the winner function to the published
sweep", exactly the innovation hook of this family.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called repeatedly -- once per sweep variant).
  stdin : ONE JSON object (the PUBLIC instance -- one election, either untouched or rewritten):
            {"num_voters": 100, "num_candidates": 10,
             "ballots": [[c_0,...,c_9], ...]}     # 100 permutations of 0..9, most-preferred first
  stdout: ONE JSON object:
            {"winner": w}                          # 0 <= w < 10, integer
  Any malformed / missing / out-of-range / non-integer winner, a crash, a timeout, or non-JSON on
  ANY of the 4 calls made for one instance -> that instance scores 0.0 (a rule that cannot survive
  a bloc rewrite of its own input has failed the robustness requirement by construction).

SCORING (deterministic; no wall-time).  Per instance i, hidden per-voter utilities (never sent to
the candidate) give a social cost per candidate; c* = argmin cost.  For a rule R we evaluate R on
4 ballot sets (untouched, COMPROMISE(b), BURY(b), CYCLE-INJECT(b)) and take
    D_i(R) = max over the 4 calls of  cost(R's winner) / cost(c*)          (>= 1, lower better)
Normalizing against an internal PLURALITY reference computed by THIS process (never a candidate
file) the same way:
    r_i = clamp( 0.1 * D_i(plurality) / max(D_i(R), 1e-9), 0, 1 )
Plain plurality reproduces the reference exactly (r_i = 0.1 on every instance: "textbook, no
defense" anchor).  A rule with materially lower worst-case distortion than plurality on the sweep
scores above 0.1; a rule that is even MORE exploitable scores below 0.1.  3 of the 10 base
elections have a genuine 3-way majority cycle among their top candidates (Smith set size 3) baked
in from the start, so a rule that only ever resolves single Condorcet winners is also exercised.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via `isorun.run_candidate`
for EVERY one of the 4 sweep calls; it only ever sees one ballot set at a time, never the hidden
utilities, cost table, or which sweep variant it is looking at.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys
import json
import random
import isorun

N_VOTERS = 100
N_CAND = 10
DELTA = 1.0     # own-favorite utility boost (own-camp voters)
SECOND = 0.25   # rotation-group second-preference boost (cycle instances only)
NOISE = 0.07    # bounded per-voter idiosyncratic noise
TIMEOUT = 15


# ============================== instance construction =======================================
def _finish(seed, rng, q, voters, target_hint, second_boost_fn=None):
    util = []
    for v in range(N_VOTERS):
        fav = voters[v]
        sb = second_boost_fn(fav) if second_boost_fn else None
        u = [0.0] * N_CAND
        for c in range(N_CAND):
            base = q[c]
            if c == fav:
                base += DELTA
            elif c == sb:
                base += SECOND
            noise = rng.uniform(-NOISE, NOISE)
            u[c] = min(1.0, max(0.0, base + noise))
        util.append(u)
    ballots = [sorted(range(N_CAND), key=lambda c: (-util[v][c], c)) for v in range(N_VOTERS)]
    cost = [sum(1 - util[v][c] for v in range(N_VOTERS)) for c in range(N_CAND)]
    cstar = min(range(N_CAND), key=lambda c: (cost[c], c))
    worst = max(range(N_CAND), key=lambda c: (cost[c], c))
    tally = [0] * N_CAND
    for b in ballots:
        tally[b[0]] += 1
    order = sorted(range(N_CAND), key=lambda c: (-tally[c], c))
    return dict(seed=seed, ballots=ballots, cost=cost, cstar=cstar, worst=worst,
                tally=tally, L=order[0], RU=order[1], target=target_hint)


def _build_compromise_profile(seed, s_L, s_RU, filler_sizes, q_L, q_RU, q_fill):
    """A dominant, decent-quality LEADER camp vs. a smaller, LOW-quality RUNNER-UP camp (a
    narrow-interest candidate with real but limited backing) plus a few modest filler camps."""
    rng = random.Random(seed)
    perm = list(range(N_CAND))
    rng.shuffle(perm)
    L_id, RU_id = perm[0], perm[1]
    filler_ids = perm[2:2 + len(filler_sizes)]
    rest_ids = perm[2 + len(filler_sizes):]
    q = [0.0] * N_CAND
    q[L_id] = q_L
    q[RU_id] = q_RU
    for fid, qf in zip(filler_ids, q_fill):
        q[fid] = qf
    for rid in rest_ids:
        q[rid] = rng.uniform(0.15, 0.45)
    camp_list = [(L_id, s_L), (RU_id, s_RU)] + list(zip(filler_ids, filler_sizes))
    assigned = sum(c for _, c in camp_list)
    remainder = N_VOTERS - assigned
    if remainder > 0:
        camp_list.append((rest_ids[0], remainder))
    voters = []
    for cid, size in camp_list:
        voters.extend([cid] * size)
    return _finish(seed, rng, q, voters, target_hint=RU_id)


def _build_cycle_profile(seed, X, filler_ids, camp_size, filler_sizes, q_top, q_fill):
    """Three comparable camps X[0],X[1],X[2].  Camp k's voters rank X[k] first and X[k+1 mod 3]
    (boosted) second, X[k+2 mod 3] third -- the classic rotation that plants a genuine majority
    cycle among the three, independent of their true quality gap."""
    rng = random.Random(seed)
    used = set(X) | set(filler_ids)
    rest_ids = [i for i in range(N_CAND) if i not in used]
    q = [0.0] * N_CAND
    for xi, qv in zip(X, q_top):
        q[xi] = qv
    for fid, qv in zip(filler_ids, q_fill):
        q[fid] = qv
    for rid in rest_ids:
        q[rid] = rng.uniform(0.05, 0.2)
    camp_list = [(X[0], camp_size), (X[1], camp_size), (X[2], camp_size)]
    for fid, fs in zip(filler_ids, filler_sizes):
        camp_list.append((fid, fs))
    assigned = sum(c for _, c in camp_list)
    remainder = N_VOTERS - assigned
    if remainder > 0:
        camp_list.append((rest_ids[0] if rest_ids else filler_ids[0], remainder))
    voters = []
    for cid, size in camp_list:
        voters.extend([cid] * size)

    def second_boost(fav):
        if fav in X:
            k = X.index(fav)
            return X[(k + 1) % 3]
        return None

    return _finish(seed, rng, q, voters, target_hint=X[1], second_boost_fn=second_boost)


def _trap_spec(b):
    # margin(s_L, s_RU) < 2b -> a b-voter COMPROMISE bloc can flip the leader outright
    if b == 5:
        return 48, 40, [6, 6]
    if b == 10:
        return 45, 30, [13, 12]
    return 55, 25, [10, 10]


def _control_spec(b):
    # margin(s_L, s_RU) >> 2b -> the same bloc size cannot flip it
    if b == 5:
        return 60, 15, [15, 10]
    if b == 10:
        return 65, 10, [15, 10]
    return 70, 5, [15, 10]


def make_instances():
    insts = []
    for seed, b in [(9001, 20), (9101, 20), (9201, 10), (9301, 10), (9401, 5)]:
        s_L, s_RU, fill = _trap_spec(b)
        prof = _build_compromise_profile(seed, s_L, s_RU, fill, 0.68, 0.06, [0.45, 0.30])
        insts.append({"public": {"num_voters": N_VOTERS, "num_candidates": N_CAND,
                                  "ballots": [list(x) for x in prof["ballots"]]},
                      "prof": prof, "b": b, "kind": "compromise-trap"})
    for seed, b in [(6, 20), (13, 10)]:
        prof = _build_cycle_profile(seed, [2, 5, 8], [1, 6], 28, [8, 8],
                                    [0.50, 0.52, 0.55], [0.20, 0.15])
        insts.append({"public": {"num_voters": N_VOTERS, "num_candidates": N_CAND,
                                  "ballots": [list(x) for x in prof["ballots"]]},
                      "prof": prof, "b": b, "kind": "cycle-trap"})
    for seed, b in [(9501, 5), (9601, 10), (9801, 20)]:
        s_L, s_RU, fill = _control_spec(b)
        prof = _build_compromise_profile(seed, s_L, s_RU, fill, 0.68, 0.06, [0.45, 0.30])
        insts.append({"public": {"num_voters": N_VOTERS, "num_candidates": N_CAND,
                                  "ballots": [list(x) for x in prof["ballots"]]},
                      "prof": prof, "b": b, "kind": "control"})
    return insts


# ============================== published sweep (attacks) ===================================
def _bloc_by_first_choice(ballots, prefs, b):
    idxs, seen = [], set()
    for pref in prefs:
        for v, bal in enumerate(ballots):
            if v in seen:
                continue
            if bal[0] == pref:
                idxs.append(v)
                seen.add(v)
                if len(idxs) >= b:
                    return idxs
    for v in range(len(ballots)):
        if v not in seen:
            idxs.append(v)
            seen.add(v)
            if len(idxs) >= b:
                break
    return idxs[:b]


def _attack_compromise(prof, b):
    ballots = [list(x) for x in prof["ballots"]]
    target = prof["target"]
    for v in _bloc_by_first_choice(ballots, [prof["L"]], b):
        rest = [c for c in ballots[v] if c != target]
        ballots[v] = [target] + rest
    return ballots


def _attack_bury(prof, b):
    ballots = [list(x) for x in prof["ballots"]]
    cstar = prof["cstar"]
    idxs = []
    for topk in (2, 4, N_CAND):
        idxs = [v for v in range(len(ballots)) if cstar in ballots[v][:topk]]
        if len(idxs) >= b:
            break
    for v in idxs[:b]:
        rest = [c for c in ballots[v] if c != cstar]
        ballots[v] = rest + [cstar]
    return ballots


def _attack_cycle_inject(prof, b):
    ballots = [list(x) for x in prof["ballots"]]
    order = sorted(range(N_CAND), key=lambda c: (-prof["tally"][c], c))
    X1, X2, X3 = order[0], order[1], order[2]
    n = len(ballots)
    bloc = list(range(n - b, n))
    base, rem = divmod(b, 3)
    sizes = [base + (1 if k < rem else 0) for k in range(3)]  # e.g. b=20 -> [7,7,6]; differ by <=1
    g1, g2 = sizes[0], sizes[0] + sizes[1]
    groups = [bloc[0:g1], bloc[g1:g2], bloc[g2:]]
    rotations = [[X1, X2, X3], [X2, X3, X1], [X3, X1, X2]]
    for grp, rot in zip(groups, rotations):
        for v in grp:
            rest = [c for c in ballots[v] if c not in (X1, X2, X3)]
            ballots[v] = rot + rest
    return ballots


def _sweep_variants(prof, b):
    return [prof["ballots"], _attack_compromise(prof, b),
            _attack_bury(prof, b), _attack_cycle_inject(prof, b)]


# ============================== internal plurality reference =================================
def _plurality_winner(ballots, m):
    tally = [0] * m
    for b in ballots:
        tally[b[0]] += 1
    return max(range(m), key=lambda c: (tally[c], -c))


def _distortion(cost, cstar, winner):
    return cost[winner] / max(cost[cstar], 1e-9)


def baseline(inst):
    prof, b = inst["prof"], inst["b"]
    ds = [_distortion(prof["cost"], prof["cstar"], _plurality_winner(bal, N_CAND))
          for bal in _sweep_variants(prof, b)]
    return max(ds)


# ============================== answer validation =============================================
def _validate_winner(answer):
    if not isinstance(answer, dict):
        return None
    w = answer.get("winner")
    if isinstance(w, bool) or not isinstance(w, int):
        return None
    if w < 0 or w >= N_CAND:
        return None
    return w


def score(inst, cand_path):
    """Run the candidate rule on all 4 sweep variants (isolated). Return (ok, D_i)."""
    prof, b = inst["prof"], inst["b"]
    worst_d = 1.0
    for bal in _sweep_variants(prof, b):
        public = {"num_voters": N_VOTERS, "num_candidates": N_CAND, "ballots": bal}
        ans, st = isorun.run_candidate(cand_path, public, timeout=TIMEOUT)
        if st != "OK":
            return False, None
        w = _validate_winner(ans)
        if w is None:
            return False, None
        d = _distortion(prof["cost"], prof["cstar"], w)
        if not (d == d) or d in (float("inf"), float("-inf")):
            return False, None
        if d > worst_d:
            worst_d = d
    return True, worst_d


# ============================== driver =========================================================
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()

    vec = []
    for inst in insts:
        ok, D = score(inst, cand)
        if not ok:
            vec.append(0.0)
            continue
        b_ref = baseline(inst)
        r = min(1.0, 0.1 * b_ref / max(D, 1e-9))
        if not (r == r) or r in (float("inf"), float("-inf")) or not (0.0 <= r <= 1.0):
            r = 0.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
