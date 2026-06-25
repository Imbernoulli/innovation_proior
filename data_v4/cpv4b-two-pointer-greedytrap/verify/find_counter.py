import random

# SHARED-BUDGET version:
# left takes prefix of i containers, right takes suffix of j containers,
# i + j <= n (no overlap), pref[i] + suf[j] <= B (single shared budget).
# maximize i + j.

def optimal(n, B, w):
    best = 0
    pref = [0]*(n+1)
    for i in range(n):
        pref[i+1] = pref[i] + w[i]
    suf = [0]*(n+1)
    for j in range(n):
        suf[j+1] = suf[j] + w[n-1-j]
    for i in range(n+1):
        if pref[i] > B: break
        for j in range(n - i + 1):
            if pref[i] + suf[j] <= B:
                best = max(best, i + j)
    return best

def greedy_cheapest_next(n, B, w):
    # two pointers from the ends; at each step take whichever END container is
    # lighter (tie -> left), as long as it fits the shared remaining budget and
    # lo <= hi. stop when neither end fits or pointers cross.
    lo, hi = 0, n - 1
    used = 0
    cnt = 0
    while lo <= hi:
        if lo == hi:
            if used + w[lo] <= B:
                cnt += 1; used += w[lo]
            break
        takeLeft = w[lo] <= w[hi]
        if takeLeft:
            if used + w[lo] <= B:
                used += w[lo]; cnt += 1; lo += 1
            elif used + w[hi] <= B:
                used += w[hi]; cnt += 1; hi -= 1
            else:
                break
        else:
            if used + w[hi] <= B:
                used += w[hi]; cnt += 1; hi -= 1
            elif used + w[lo] <= B:
                used += w[lo]; cnt += 1; lo += 1
            else:
                break
    return cnt

rng = random.Random(7)
found = []
for trial in range(300000):
    n = rng.randint(1, 7)
    w = [rng.randint(1, 8) for _ in range(n)]
    B = rng.randint(0, sum(w))
    o = optimal(n, B, w)
    g = greedy_cheapest_next(n, B, w)
    if g < o:
        found.append((n, B, w, g, o))
        if len(found) >= 10:
            break

for f in found:
    print("n=%d B=%d w=%s greedy=%d optimal=%d" % f)
if not found:
    print("NO COUNTEREXAMPLE FOUND")
