# TIER: trivial
# Non-adaptive: ignores ALL feedback. Simply enumerates automaton-accepted strings
# of length N in ascending lexicographic order and guesses the k-th one on round k
# (k = number of rounds already played, from the public history).
import sys, json


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
            lang.append(list(cur))
            return
        for a in range(A):
            nq = delta[q][a]
            if cnt[i + 1][nq] > 0:
                cur.append(a)
                dfs(i + 1, nq, cur)
                cur.pop()

    dfs(0, start, [])

    k = len(history)
    guess = lang[k] if k < len(lang) else lang[-1]
    print(json.dumps({"guess": guess}))


main()
