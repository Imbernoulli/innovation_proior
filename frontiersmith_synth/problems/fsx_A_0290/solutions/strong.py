# TIER: strong
# Ratio-greedy + absolute-greedy warm starts, then local search over
# ADD / DROP-and-SWAP moves on a coverage-count grid.  We keep a per-cell count of
# how many built towers watch it, so a tower can be removed in O(area) and swaps
# reevaluated cheaply.  Passes: (1) add any affordable tower with positive marginal
# gain; (2) for each built tower, try replacing it with the best affordable unbuilt
# tower and keep the swap only if total watched weight strictly rises.  Two warm
# starts are tried and the better final plan is returned.  This reclaims value that
# one-shot greedy overlaps away -- but the all-towers ceiling stays unaffordable, so
# scores keep headroom under 1.0.
import sys, json

inst = json.load(sys.stdin)
N, M, B = inst["N"], inst["M"], inst["B"]
W, tx, ty, tr, tc = inst["weight"], inst["tx"], inst["ty"], inst["tr"], inst["tc"]

RECT = []
for j in range(M):
    r = tr[j]
    r0 = max(0, ty[j] - r); r1 = min(N - 1, ty[j] + r)
    c0 = max(0, tx[j] - r); c1 = min(N - 1, tx[j] + r)
    RECT.append((r0, r1, c0, c1))


class Cover:
    __slots__ = ("cnt", "weight", "spent", "built")

    def __init__(self):
        self.cnt = [[0] * N for _ in range(N)]
        self.weight = 0
        self.spent = 0
        self.built = set()

    def add_gain(self, j):
        r0, r1, c0, c1 = RECT[j]
        g = 0
        cnt = self.cnt
        for r in range(r0, r1 + 1):
            cr = cnt[r]; wr = W[r]
            for c in range(c0, c1 + 1):
                if cr[c] == 0:
                    g += wr[c]
        return g

    def add(self, j):
        r0, r1, c0, c1 = RECT[j]
        cnt = self.cnt
        for r in range(r0, r1 + 1):
            cr = cnt[r]; wr = W[r]
            for c in range(c0, c1 + 1):
                if cr[c] == 0:
                    self.weight += wr[c]
                cr[c] += 1
        self.spent += tc[j]
        self.built.add(j)

    def remove(self, j):
        r0, r1, c0, c1 = RECT[j]
        cnt = self.cnt
        for r in range(r0, r1 + 1):
            cr = cnt[r]; wr = W[r]
            for c in range(c0, c1 + 1):
                cr[c] -= 1
                if cr[c] == 0:
                    self.weight -= wr[c]
        self.spent -= tc[j]
        self.built.discard(j)


def warm_start(mode):
    """mode 'ratio' = gain/cost; mode 'abs' = raw gain."""
    cov = Cover()
    avail = set(range(M))
    while True:
        best_j = -1; best_key = 0.0
        for j in avail:
            if cov.spent + tc[j] > B:
                continue
            g = cov.add_gain(j)
            if g <= 0:
                continue
            key = g / tc[j] if mode == "ratio" else float(g)
            if key > best_key + 1e-12:
                best_key = key; best_j = j
        if best_j < 0:
            break
        cov.add(best_j); avail.discard(best_j)
    return cov


def local_search(cov):
    for _ in range(8):
        improved = False
        # (1) pure adds
        for j in range(M):
            if j in cov.built:
                continue
            if cov.spent + tc[j] <= B and cov.add_gain(j) > 0:
                cov.add(j); improved = True
        # (2) drop-and-swap
        for i in sorted(cov.built):
            if i not in cov.built:
                continue
            w_before = cov.weight
            cov.remove(i)
            best_j = -1; best_w = cov.weight
            for j in range(M):
                if j in cov.built:
                    continue
                if cov.spent + tc[j] > B:
                    continue
                w_after = cov.weight + cov.add_gain(j)
                if w_after > best_w + 1e-9:
                    best_w = w_after; best_j = j
            if best_j >= 0 and best_w > w_before + 1e-9:
                cov.add(best_j); improved = True
            else:
                cov.add(i)          # restore
        if not improved:
            break
    return cov


best_build, best_w = [], -1
for mode in ("ratio", "abs"):
    cov = local_search(warm_start(mode))
    if cov.weight > best_w:
        best_w = cov.weight; best_build = sorted(cov.built)

print(json.dumps({"build": best_build}))
