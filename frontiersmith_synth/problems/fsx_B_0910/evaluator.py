"""
evaluator.py -- FROZEN scaffold + scorer for fsx_B_0910 (Automaton Combination Lock).

Deterministic, seeded instance generator + an interactive (multi-round) driver that
repeatedly invokes the candidate program, ONE isolated subprocess call per guess round,
via isorun.run_candidate. The candidate never sees the hidden target or its state
trajectory -- only the automaton definition (public) and the growing public history of
(guess, feedback, correct) from prior rounds.

CLI: python3 evaluator.py <candidate.py>
Prints:
    Ratio: <mean normalized score in [0,1]>
    Vector: [r1, ..., r10]
"""
import sys
import json
import random
import isorun


# ---------------------------------------------------------------------------
# Automaton construction (S live states 0..S-1, plus one absorbing REJECT state
# at index S). From each live state only m of the A symbols are "legal"; the
# rest transition to REJECT (which stays REJECT forever -- it is never in the
# accept set, and the secret target -- being a language member -- never visits it).
# ---------------------------------------------------------------------------
def make_local_automaton(A, S, m, seed):
    """History-independent: the legal-symbol set and their target states are the
    SAME at every live state (state == a relabeling of the last symbol read).
    Fully injective on the allowed subset -> no automaton-state merges."""
    rng = random.Random(seed)
    REJ = S
    delta = [[REJ] * A for _ in range(S + 1)]
    allow = sorted(rng.sample(range(A), m))
    targets = rng.sample(range(S), min(m, S))
    while len(targets) < m:
        targets.append(rng.randrange(S))
    tmap = dict(zip(allow, targets))
    for q in range(S):
        for a in range(A):
            delta[q][a] = tmap.get(a, REJ)
    delta[REJ] = [REJ] * A
    return delta, REJ


def make_coupled_automaton(A, S, m, merge_p, seed):
    """History-dependent: legal-symbol set and target states are chosen PER live
    state. With probability merge_p, two of a state's m legal symbols are forced
    to collide on the same next state (a genuine automaton-state merge: two
    different symbols become indistinguishable via state feedback at that step)."""
    rng = random.Random(seed)
    REJ = S
    delta = [[REJ] * A for _ in range(S + 1)]
    for q in range(S):
        allow = sorted(rng.sample(range(A), m))
        targets = rng.sample(range(S), min(m, S))
        while len(targets) < m:
            targets.append(rng.randrange(S))
        if m >= 2 and merge_p > 0 and rng.random() < merge_p:
            targets[1] = targets[0]
        for a, t in zip(allow, targets):
            delta[q][a] = t
    delta[REJ] = [REJ] * A
    return delta, REJ


def dp_counts(delta, accept, S, A, L, REJ):
    """cnt[i][q] = number of length-(L-i) suffixes from state q that reach accept."""
    cnt = [[0] * (S + 1) for _ in range(L + 1)]
    for q in range(S + 1):
        cnt[L][q] = 1 if q in accept else 0
    for i in range(L - 1, -1, -1):
        for q in range(S + 1):
            cnt[i][q] = sum(cnt[i + 1][delta[q][a]] for a in range(A))
    return cnt


def sample_uniform(delta, cnt, S, A, L, start, rng):
    """Sample a length-L accepted string uniformly at random (weighted walk)."""
    q = start
    s = []
    for i in range(L):
        weights = [cnt[i + 1][delta[q][a]] for a in range(A)]
        tot = sum(weights)
        r = rng.randrange(tot)
        acc = 0
        for a in range(A):
            acc += weights[a]
            if r < acc:
                s.append(a)
                q = delta[q][a]
                break
    return s


def enumerate_lang_sorted(delta, cnt, S, A, L, start):
    """All accepted length-L strings, in ascending lexicographic order (DFS with
    DP-pruned branches only -- feasible since |language| is kept small)."""
    out = []

    def dfs(i, q, cur):
        if i == L:
            out.append(tuple(cur))
            return
        for a in range(A):
            nq = delta[q][a]
            if cnt[i + 1][nq] > 0:
                cur.append(a)
                dfs(i + 1, nq, cur)
                cur.pop()

    dfs(0, start, [])
    return out


def trajectory(delta, s, start):
    q = start
    traj = []
    for a in s:
        q = delta[q][a]
        traj.append(q)
    return traj


# ---------------------------------------------------------------------------
# Instance table: (L, A, S, m, mode, merge_p, salt)
# mode 'local'   -> history-independent (no automaton-state merges): the
#                   "obvious" per-position elimination strategy is basically sound.
# mode 'coupled' -> history-dependent, merge_p controls how often two legal
#                   symbols from a state deliberately collide on the same next
#                   state. A candidate that assumes "state feedback == this
#                   symbol is correct" gets fooled here and can lock in the
#                   WRONG symbol early, permanently derailing itself.
# ---------------------------------------------------------------------------
PARAMS = [
    (5, 4, 4, 2, 'local',   0.0, 1),
    (6, 4, 4, 2, 'local',   0.0, 2),
    (6, 5, 4, 2, 'coupled', 0.5, 3),
    (6, 5, 4, 2, 'coupled', 0.6, 4),
    (7, 5, 5, 2, 'coupled', 0.3, 5),
    (7, 6, 5, 2, 'coupled', 0.5, 6),
    (6, 4, 3, 2, 'local',   0.0, 7),
    (8, 6, 4, 2, 'coupled', 0.5, 8),   # held-out, larger
    (9, 5, 4, 2, 'coupled', 0.4, 9),   # held-out, larger
    (9, 6, 5, 2, 'coupled', 0.45, 10),  # held-out, larger
]
BASE_SEED = 20260719
ACCEPT_FRACS = [0.5, 0.4, 0.6, 0.3, 0.7, 0.35, 0.45, 0.55, 0.25, 0.65, 0.2, 0.8, 0.15, 0.1]


def build_one(idx, p):
    L, A, S, m, mode, mp, tw = p
    if mode == 'local':
        delta, REJ = make_local_automaton(A, S, m, BASE_SEED * 7 + tw * 131)
    else:
        delta, REJ = make_coupled_automaton(A, S, m, mp, BASE_SEED * 7 + tw * 131)
    rng = random.Random(BASE_SEED * 13 + tw * 17)
    accept, cnt, lang_size = None, None, 0
    for frac in ACCEPT_FRACS:
        k = max(1, round(S * frac))
        accept = set(rng.sample(range(S), k))
        cnt = dp_counts(delta, accept, S, A, L, REJ)
        lang_size = cnt[0][0]
        if 20 <= lang_size <= 1500:
            break
    start = 0
    lang_sorted = enumerate_lang_sorted(delta, cnt, S, A, L, start)

    target, hi = None, None
    for trial in range(60):
        trng = random.Random(BASE_SEED * 31 + tw * 7 + 999 + trial * 97)
        cand = sample_uniform(delta, cnt, S, A, L, start, trng)
        pos = lang_sorted.index(tuple(cand)) + 1
        if pos >= 0.45 * lang_size:
            target, hi = cand, pos
            break
    if target is None:
        trng = random.Random(BASE_SEED * 31 + tw * 7 + 999)
        target = sample_uniform(delta, cnt, S, A, L, start, trng)
        hi = lang_sorted.index(tuple(target)) + 1

    # max_guesses must comfortably exceed `hi` too, so the non-adaptive baseline
    # sweep (trivial) can always actually reach the target within budget.
    max_guesses = max(L * A + 10, hi + 8)
    target_traj = trajectory(delta, target, start)
    public = {
        "N": L, "A": A, "S": S, "reject_state": REJ,
        "delta": delta, "accept": sorted(accept), "start": start,
        "max_guesses": max_guesses,
    }
    hidden = {"target": target, "target_traj": target_traj, "hi": hi, "lang_size": lang_size}
    return {"public": public, "hidden": hidden}


def make_instances():
    return [build_one(i, p) for i, p in enumerate(PARAMS)]


def baseline(inst):
    """The evaluator's own reference: how many guesses a non-adaptive, feedback-
    blind sweep through the automaton's accepted strings (ascending lexicographic
    order) would need to reach THIS instance's secret target."""
    return inst["hidden"]["hi"]


# ---------------------------------------------------------------------------
# Interactive driver: one isorun.run_candidate call PER GUESS ROUND. The
# candidate process is fresh each round (no persistent state) and must
# recompute everything from the public automaton + the growing "history" list
# handed back to it -- exactly like a real interactive protocol, but each turn
# individually OS-sandboxed and isolated from evaluator internals.
# ---------------------------------------------------------------------------
def play(cand_path, inst):
    pub = inst["public"]
    L, A = pub["N"], pub["A"]
    delta = pub["delta"]
    start = pub["start"]
    max_guesses = pub["max_guesses"]
    target = inst["hidden"]["target"]
    target_traj = inst["hidden"]["target_traj"]

    history = []
    for r in range(max_guesses):
        payload = dict(pub)
        payload["history"] = history
        ans, st = isorun.run_candidate(cand_path, payload, timeout=8)
        if st != "OK" or not isinstance(ans, dict):
            return None
        guess = ans.get("guess")
        if not (isinstance(guess, list) and len(guess) == L
                and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < A for x in guess)):
            return None
        q = start
        fb = []
        for c in guess:
            q = delta[q][c]
            fb.append(1 if q == target_traj[len(fb)] else 0)
        correct = (guess == target)
        history.append({"guess": guess, "feedback": fb, "correct": correct})
        if correct:
            return r + 1
    return None


def score(inst, obj):
    """obj = guesses used (None if never solved / malformed guess). Linear map
    between the blind-sweep baseline `hi` (-> 0.1) and a 1-guess ideal (-> 0.9),
    clamped -- keeps headroom above the strong reference (see statement.md)."""
    if obj is None:
        return False, 0.0
    hi = inst["hidden"]["hi"]
    frac = (hi - obj) / (hi - 1) if hi > 1 else 0.0
    frac = max(0.0, min(1.0, frac))
    r = 0.1 + 0.8 * frac
    return True, r


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        obj = play(cand, inst)
        ok, r = score(inst, obj)
        vec.append(r if (ok and r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
