# TIER: strong
"""Phase detector, not a fixed access rule.  Tracks a sliding window of the
last W visitors and, from it, an online estimate of the CURRENT phase's
repeat-traffic probability `p` (fraction of the window that were re-visits of
something already seen inside the window).  Before each visitor's search it
estimates the expected future payoff of splaying the requested tag to the
root -- `p * HORIZON` plus a direct bonus if this exact tag has already
reappeared inside the window -- and only pays for the rotation when that
expected benefit exceeds THIS VISIT's season_weight (the actual cost of
restructuring right now).  So during a cheap, repeat-heavy phase it splays
liberally; during an expensive or repeat-free phase (a peak-season procession
or echo-walk) it recognizes rotating buys nothing and stays out of the way,
paying pure hop cost instead of ruinous restructuring cost.  This metering,
not blanket splaying, is what beats both 'always rotate' and 'never rotate'."""
import sys, json
from collections import deque, defaultdict

W = 30          # sliding-window length for the phase estimate
HORIZON = 6.0   # amortization horizon: how many future hits a hot phase is
                # assumed to still deliver once detected


def build_tree(inst):
    n = inst["n"]
    leftArr = inst["left"]; rightArr = inst["right"]
    parent = [0] * (n + 1)
    left = [0] * (n + 1); right = [0] * (n + 1)
    for k in range(1, n + 1):
        l = leftArr[k - 1]; r = rightArr[k - 1]
        left[k] = l; right[k] = r
        if l:
            parent[l] = k
        if r:
            parent[r] = k
    return {"left": left, "right": right, "parent": parent, "root": inst["root"]}


def rotate_up(tree, k):
    par = tree["parent"]; left = tree["left"]; right = tree["right"]
    p = par[k]
    if p == 0:
        return False
    g = par[p]
    if left[p] == k:
        b = right[k]
        left[p] = b
        if b:
            par[b] = p
        right[k] = p
    else:
        b = left[k]
        right[p] = b
        if b:
            par[b] = p
        left[k] = p
    par[p] = k
    par[k] = g
    if g == 0:
        tree["root"] = k
    else:
        if left[g] == p:
            left[g] = k
        else:
            right[g] = k
    return True


def main():
    inst = json.load(sys.stdin)
    tree = build_tree(inst)
    accesses = inst["accesses"]
    weights = inst["season_weight"]

    window = deque()
    cnt = defaultdict(int)
    dup_in_window = 0   # count of window slots whose key has occurred >=2x in the window

    ops = []
    for i, key in enumerate(accesses):
        w = weights[i]
        wl = len(window)
        p = (dup_in_window / wl) if wl > 0 else 0.0
        key_recent = cnt[key]   # occurrences of THIS key already inside the window
        expected_benefit = p * HORIZON + (2.0 if key_recent > 0 else 0.0)

        oi = []
        if expected_benefit > w:
            cur = key
            while tree["parent"][cur] != 0:
                oi.append(cur)
                rotate_up(tree, cur)
        ops.append(oi)

        # slide the window forward with the raw access key (independent of
        # any tree bookkeeping -- purely a recent-access statistic)
        window.append(key)
        cnt[key] += 1
        if cnt[key] >= 2:
            dup_in_window += 1
        if len(window) > W:
            old = window.popleft()
            if cnt[old] >= 2:
                dup_in_window -= 1
            cnt[old] -= 1
            if cnt[old] == 0:
                del cnt[old]

    print(json.dumps({"ops": ops}))


if __name__ == "__main__":
    main()
