#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for vitality-phase-alignment-bakery.

Reads the instance from <in>, the participant's feed schedule from <out>.
Validates strictly, simulates the vitality dynamics, sums the value of fulfilled
orders (F), computes the checker's own baseline construction (B: the single
cheapest tank fed every day of the whole horizon), and prints
  Ratio: <F/B calibrated to [0,1], capped>
on its own final line.
"""
import math
import sys

SCALE = 200


def transition(v, n, fed, GROW, BASE_KICK, DECAY, SAT, CRASH_DIV):
    if fed:
        n2 = n + 1
        if n2 > SAT:
            v2 = v // CRASH_DIV
            n2 = 0
        else:
            growth = (GROW * v * (SCALE - v)) // (SCALE * SCALE)
            v2 = v + growth + BASE_KICK
            if v2 > SCALE:
                v2 = SCALE
    else:
        n2 = n - 1
        if n2 < 0:
            n2 = 0
        v2 = (v * (1000 - DECAY)) // 1000
    return v2, n2


def fail(reason):
    print(f"INVALID: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(tokens, k):
    if len(tokens) < k:
        return None
    out = []
    for t in tokens[:k]:
        try:
            if not (t.lstrip("-").isdigit()):
                return None
            out.append(int(t))
        except Exception:
            return None
    return out


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    idx = 0

    def nxt(k):
        nonlocal idx
        vals = in_tokens[idx: idx + k]
        idx += k
        return vals

    header = read_ints(nxt(4), 4)
    if header is None:
        fail("bad instance header")
    T, H, M, BUDGET = header

    tanks = []
    for _ in range(T):
        row = read_ints(nxt(6), 6)
        if row is None:
            fail("bad instance (tank row)")
        tanks.append(row)  # GROW,BASE_KICK,DECAY,SAT,CRASH_DIV,COST

    orders = []
    for _ in range(M):
        row = read_ints(nxt(3), 3)
        if row is None:
            fail("bad instance (order row)")
        day, theta, value = row
        if not (1 <= day <= H):
            fail("bad instance (order day out of range)")
        orders.append((day, theta, value))

    # ---- parse participant output ----
    try:
        with open(out_path) as f:
            out_text = f.read()
    except Exception:
        fail("cannot read output")

    out_tokens = out_text.split()
    if not out_tokens:
        fail("empty output")

    for tok in out_tokens:
        low = tok.lower()
        if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            fail("non-finite token")
        try:
            fv = float(tok)
        except ValueError:
            fail("non-numeric token")
        if not math.isfinite(fv):
            fail("non-finite value")

    pos = 0
    if pos >= len(out_tokens):
        fail("missing K")
    try:
        K = int(out_tokens[pos])
    except ValueError:
        fail("K not an integer")
    pos += 1
    if K < 0 or K > T * H + 5:
        fail("K out of range")

    if len(out_tokens) < pos + 2 * K:
        fail("truncated schedule")

    seen = set()
    fed = [[False] * H for _ in range(T)]
    total_cost = 0
    for _ in range(K):
        tt = out_tokens[pos]
        dd = out_tokens[pos + 1]
        pos += 2
        try:
            tank = int(tt)
            day = int(dd)
        except ValueError:
            fail("non-integer schedule entry")
        if not (0 <= tank < T):
            fail("tank index out of range")
        if not (0 <= day < H):
            fail("day out of range")
        if (tank, day) in seen:
            fail("duplicate (tank,day) feed event")
        seen.add((tank, day))
        fed[tank][day] = True
        total_cost += tanks[tank][5]

    if pos != len(out_tokens):
        fail("trailing garbage after schedule")

    if total_cost > BUDGET:
        fail(f"budget exceeded ({total_cost} > {BUDGET})")

    # ---- simulate ----
    def simulate(fed_matrix):
        hist = []
        for i in range(T):
            GROW, BASE_KICK, DECAY, SAT, CRASH_DIV, _COST = tanks[i]
            v, n = 0, 0
            row = [0]
            for d in range(H):
                v, n = transition(v, n, fed_matrix[i][d], GROW, BASE_KICK, DECAY, SAT, CRASH_DIV)
                row.append(v)
            hist.append(row)
        return hist

    hist = simulate(fed)

    F = 0
    for (day, theta, value) in orders:
        best = 0
        for i in range(T):
            if hist[i][day] > best:
                best = hist[i][day]
        if best >= theta:
            F += value

    # ---- checker's own baseline: cheapest tank, fed every day of the whole horizon ----
    cheapest = min(range(T), key=lambda i: (tanks[i][5], i))
    base_fed = [[False] * H for _ in range(T)]
    for d in range(H):
        base_fed[cheapest][d] = True
    base_hist = simulate(base_fed)
    B = 0
    for (day, theta, value) in orders:
        if base_hist[cheapest][day] >= theta:
            B += value

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F} B={B} cost={total_cost}/{BUDGET}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
