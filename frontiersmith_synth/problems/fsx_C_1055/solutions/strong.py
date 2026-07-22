# TIER: strong
# The insight: don't aim at the nominal deficit -- SIMULATE the true
# shadowing-capture landing rule for every candidate aim column and pick the
# one whose ACTUAL landing minimizes the resulting squared error, right now.
# This is a genuine reformulation from "aim to shape" to "pre-compensate for
# shadowing": it naturally does two things a naive recipe does not.
#   - Under the budget shortfall (T < sum(target)), it correctly favors
#     finishing the columns whose squared-error PENALTY for staying unmet is
#     largest, because that is what actually shrinks the objective the most
#     -- not whichever column merely looks "neediest" in raw target units.
#   - Whenever a shot at the seemingly best column would actually be
#     captured by an already-taller neighbor (shadow capture), the
#     simulated delta for that candidate reflects the real, useless (or
#     harmful) outcome, so the search naturally steers away from it instead
#     of blindly repeating the mistake like a deficit-chasing recipe does.
import sys


def landing_col(h, a, R, L, M):
    lo = a - R
    hi = a + R
    if lo < 0:
        lo = 0
    if hi > L - 1:
        hi = L - 1
    ha = h[a]
    best_c = None
    best_key = None
    for c in range(lo, hi + 1):
        if c == a:
            continue
        if h[c] > ha + M:
            key = (h[c], -abs(c - a), -c)
            if best_key is None or key > best_key:
                best_key = key
                best_c = c
    return best_c if best_c is not None else a


def apply_shot(h, aim, R, L, M):
    c = landing_col(h, aim, R, L, M)
    left = h[c - 1] if c - 1 >= 0 else -1
    right = h[c + 1] if c + 1 < L else -1
    h[c] = max(h[c] + 1, left, right)


def lookahead_best(h, L, R, M, target):
    best_a = 0
    best_delta = None
    for a in range(L):
        c = landing_col(h, a, R, L, M)
        left = h[c - 1] if c - 1 >= 0 else -1
        right = h[c + 1] if c + 1 < L else -1
        newh = max(h[c] + 1, left, right)
        old_e = (h[c] - target[c]) ** 2
        new_e = (newh - target[c]) ** 2
        delta = new_e - old_e
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_a = a
    return best_a


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); M = int(next(it)); T = int(next(it))
    target = [int(next(it)) for _ in range(L)]

    h = [0] * L
    schedule = []
    for _ in range(T):
        aim = lookahead_best(h, L, R, M, target)
        schedule.append(aim)
        apply_shot(h, aim, R, L, M)

    print(" ".join(map(str, schedule)))


if __name__ == "__main__":
    main()
