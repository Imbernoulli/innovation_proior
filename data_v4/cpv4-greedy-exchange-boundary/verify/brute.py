import sys
from collections import deque

def solve(n, L, t):
    # Independent brute force (does NOT assume the greedy structure).
    #
    # A snapshot taken at INTEGER time s captures every pulse with
    #     s <= value < s + L      (half-open window [s, s+L)).
    # We want the minimum number of snapshots so every pulse is captured.
    #
    # We enumerate EVERY plausible integer snapshot time s. A snapshot is only
    # useful if it captures at least one pulse, which forces
    #     s <= p  and  p < s + L   for some pulse p,
    # i.e.  p - L < s <= p. So the only integer s worth considering lie in the
    # union of (p-L, p] over all pulses -- a finite, small set for tiny cases.
    # We compute each candidate's coverage bitmask and do a BFS over the number
    # of snapshots used (minimum set cover by breadth-first layers). This is the
    # obvious, slow, clearly-correct method, with no greedy reasoning baked in.
    if n == 0:
        return 0
    pulses = sorted(t)
    m = len(pulses)
    full = (1 << m) - 1

    cand_s = set()
    for p in pulses:
        # integer s with p - L < s <= p
        for s in range(p - L + 1, p + 1):
            cand_s.add(s)

    masks = set()
    for s in cand_s:
        mask = 0
        for j, p in enumerate(pulses):
            if s <= p < s + L:
                mask |= (1 << j)
        if mask:
            masks.add(mask)
    masks = list(masks)

    seen = {0}
    dq = deque([(0, 0)])  # (covered_mask, count)
    while dq:
        cov, cnt = dq.popleft()
        if cov == full:
            return cnt
        for c in masks:
            nc = cov | c
            if nc not in seen:
                seen.add(nc)
                dq.append((nc, cnt + 1))
    return -1  # unreachable for finite input

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    t = []
    for _ in range(n):
        t.append(int(data[idx])); idx += 1
    print(solve(n, L, t))

if __name__ == "__main__":
    main()
