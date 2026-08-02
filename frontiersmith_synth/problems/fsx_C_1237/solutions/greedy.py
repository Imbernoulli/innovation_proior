# TIER: greedy
"""Textbook query-optimizer approach: pick the single join order that
minimizes cost PLUGGING IN the point-estimate selectivities, trust the
estimate completely (no robustness), and use that SAME order for every
reoptimization bucket (no real adaptivity). This is optimal when the
estimates are right and can be catastrophic when they are wrong."""
import sys


def run_from(n, C, lo, hi, size, moves, sel, mem_cap, spill_mult):
    cost = 0.0
    for mv in moves:
        if mv == 'L':
            if lo == 0:
                return None
            edge = lo - 1
            lo -= 1
            size = size * C[lo] * sel(edge)
        else:
            if hi == n - 1:
                return None
            edge = hi
            hi += 1
            size = size * C[hi] * sel(edge)
        step_cost = size if size <= mem_cap else size * spill_mult
        cost += step_cost
    return lo, hi, size, cost


def enumerate_plans(n):
    for s in range(n):
        stack = [(s, s, [])]
        while stack:
            lo, hi, moves = stack.pop()
            if len(moves) == n - 1:
                yield s, moves
                continue
            if lo > 0:
                stack.append((lo - 1, hi, moves + ['L']))
            if hi < n - 1:
                stack.append((lo, hi + 1, moves + ['R']))


def main():
    itok = sys.stdin.read().split()
    idx = 0
    tid = int(itok[idx]); idx += 1
    n = int(itok[idx]); idx += 1
    C = [int(itok[idx + i]) for i in range(n)]; idx += n
    mem_cap = float(itok[idx]); idx += 1
    spill_mult = float(itok[idx]); idx += 1
    h = int(itok[idx]); idx += 1
    est = []
    for e in range(n - 1):
        s_est = float(itok[idx]); idx += 1
        idx += 1  # bound F, ignored -- greedy trusts the point estimate
        est.append(s_est)

    def sel_est(e):
        return est[e]

    best = None
    for s, moves in enumerate_plans(n):
        res = run_from(n, C, s, s, float(C[s]), moves, sel_est, mem_cap, spill_mult)
        if res is None:
            continue
        cost = res[3]
        if best is None or cost < best[0]:
            best = (cost, s, moves)

    _, s, moves = best
    pre = moves[:h - 1]
    tail = moves[h - 1:]

    out = ["START", str(s), "PRE"] + pre
    out += ["BRANCH", "LOW"] + tail
    out += ["BRANCH", "MID"] + tail
    out += ["BRANCH", "HIGH"] + tail
    print(" ".join(out))


if __name__ == "__main__":
    main()
