# TIER: strong
# Entropy-guided, automaton-respecting search. Maintains the exact set of
# automaton-accepted strings still consistent with every past (guess, feedback)
# pair (feedback is the per-prefix STATE-match pattern, not raw symbol match --
# so filtering is done in state-trajectory space, correctly handling merges).
# Each round picks a guess that best splits the remaining candidate set
# (minimizes sum of squared bucket sizes over the feedback partition, a
# Knuth-style Mastermind heuristic) instead of ever assuming a single
# position's symbol is "confirmed" by one round of feedback.
import sys, json, random


def main():
    inst = json.load(sys.stdin)
    L, A, S, REJ = inst["N"], inst["A"], inst["S"], inst["reject_state"]
    delta = inst["delta"]
    accept = set(inst["accept"])
    start = inst["start"]
    history = inst.get("history", [])

    cnt = [[0] * (S + 1) for _ in range(L + 1)]
    for q in range(S + 1):
        cnt[L][q] = 1 if q in accept else 0
    for i in range(L - 1, -1, -1):
        for q in range(S + 1):
            cnt[i][q] = sum(cnt[i + 1][delta[q][a]] for a in range(A))

    lang = []

    def dfs(i, q, cur):
        if i == L:
            lang.append(tuple(cur))
            return
        for a in range(A):
            nq = delta[q][a]
            if cnt[i + 1][nq] > 0:
                cur.append(a)
                dfs(i + 1, nq, cur)
                cur.pop()

    dfs(0, start, [])

    def traj(s):
        q = start
        t = []
        for a in s:
            q = delta[q][a]
            t.append(q)
        return t

    def fb_of(guess, target_traj):
        q = start
        fb = []
        for c in guess:
            q = delta[q][c]
            fb.append(1 if q == target_traj[len(fb)] else 0)
        return fb

    candidates = list(lang)
    for h in history:
        g, fb = h["guess"], h["feedback"]
        newc = []
        for c in candidates:
            if fb_of(g, traj(c)) == fb and list(c) != g:
                newc.append(c)
        candidates = newc

    if not candidates:
        candidates = list(lang)  # defensive fallback, should not trigger

    if len(candidates) == 1:
        guess = list(candidates[0])
    else:
        rng = random.Random(12345 + len(history))
        pool = candidates if len(candidates) <= 150 else rng.sample(candidates, 150)
        trajs = {c: traj(c) for c in candidates}
        best, best_score = None, None
        for g in pool:
            buckets = {}
            for c in candidates:
                fb = tuple(fb_of(list(g), trajs[c]))
                buckets[fb] = buckets.get(fb, 0) + 1
            sc = sum(v * v for v in buckets.values())
            if best_score is None or sc < best_score:
                best_score, best = sc, g
        guess = list(best)

    print(json.dumps({"guess": guess}))


main()
