# TIER: strong
"""The insight: total interconnect cost is ADDITIVE and INDEPENDENT across
lines (a line's cost never depends on any other line's policy), so the
global optimum over the candidate policy set decomposes into L independent
local optimizations -- an exchange argument that a single global policy
cannot exploit. For each line separately, locally simulate INV, UPD, and a
small grid of reactive ADAPT(window,threshold) candidates and keep whichever
is cheapest FOR THAT LINE. This lets write-storm-shared lines take INV,
read-heavy-shared lines take UPD, and lines whose access regime changes
mid-trace take an adaptive rule that switches with them -- something no
single global choice, and no line-blind heuristic, can express."""
import sys

ADAPT_GRID = [
    (2, 0.3), (2, 0.5), (2, 0.7),
    (4, 0.3), (4, 0.5), (4, 0.7),
    (8, 0.3), (8, 0.5), (8, 0.7),
    (16, 0.5),
]


def decide_mode(mode, W, Tt, idx, events):
    if mode != "ADAPT":
        return mode
    lo = max(0, idx - W)
    window = events[lo:idx]
    if not window:
        return "INV"
    reads = sum(1 for (_c, op) in window if op == "R")
    frac = reads / len(window)
    return "UPD" if frac >= Tt else "INV"


def simulate_line(events, MISS, INVC, UPDC, mode, W=None, Tt=None):
    valid = set()
    cost = 0.0
    for idx, (c, op) in enumerate(events):
        if op == "R":
            if c not in valid:
                cost += MISS
                valid.add(c)
        else:
            m = decide_mode(mode, W, Tt, idx, events)
            others = valid - {c}
            if m == "INV":
                cost += INVC * len(others)
                valid = {c}
            else:
                cost += UPDC * len(others)
                valid.add(c)
    return cost


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    T = int(nxt())
    C = int(nxt())
    L = int(nxt())
    MISS = float(nxt())
    INVC = float(nxt())
    UPDC = float(nxt())
    events_by_line = [[] for _ in range(L)]
    for _ in range(T):
        c = int(nxt())
        l = int(nxt())
        op = nxt()
        events_by_line[l].append((c, op))

    out_lines = []
    for li in range(L):
        ev = events_by_line[li]
        best_cost = simulate_line(ev, MISS, INVC, UPDC, "INV")
        best_desc = "INV 1 0.0"
        c_upd = simulate_line(ev, MISS, INVC, UPDC, "UPD")
        if c_upd < best_cost:
            best_cost, best_desc = c_upd, "UPD 1 0.0"
        for (W, Tt) in ADAPT_GRID:
            c = simulate_line(ev, MISS, INVC, UPDC, "ADAPT", W, Tt)
            if c < best_cost:
                best_cost, best_desc = c, "ADAPT %d %.2f" % (W, Tt)
        out_lines.append(best_desc)

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
