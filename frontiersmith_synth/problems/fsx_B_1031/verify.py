#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the apical-dominance / heterochrony branching task.

Instance (from <in>):
    line 1:  H BUDGET K
    line 2:  c_1 ... c_H         (commitment cost of bud i, i=1..H)
    line 3:  t_1 ... t_K         (target bud IDs, in the REQUIRED commit order)

Participant artifact (from <out>): H integers r_1..r_H, each 0 or 1
(whitespace-separated, any line layout).  r_t = 1 means auxin is RELEASED
("low") at developmental tick t; r_t = 0 means auxin is actively PRODUCED
("high", suppressing) at tick t.  Producing auxin costs growth budget:
  sum_t (1 - r_t)  <=  BUDGET.
Any malformed / out-of-range / non-finite / over-budget artifact scores 0.

Simulation (exact, integer, deterministic).
  Bud i differentiates at tick i.  FOUNDING CHECK: if r_i == 0 (auxin high
  right at bud i's own founding instant), bud i is arrested at formation and
  can NEVER become an active branch, no matter what happens afterward.
  If r_i == 1, bud i is still "in play": it commits (becomes an active
  branch) at the first tick t >= i such that the cumulative number of
  released ticks in [i, t] reaches c_i.  If that never happens by tick H,
  bud i never commits.
  Committed buds are ranked by (commit tick ascending, bud id descending)
  to get a single deterministic global commit order ACT_ORDER (ties -- buds
  committing on the same tick -- resolve in favor of the more recently
  founded / apex-proximal bud).

Scoring (maximization).
  sim_active = set of buds that ever commit.  correct = sim_active & target
  set.  precision = |correct|/|sim_active|, recall = |correct|/|K|.
  set_score = F-beta(beta^2 = 0.15) of (precision, recall)  -- a false
  branch (an arrested bud that leaks through) is penalised far more heavily
  than a missed target, because leaking even a few unwanted branches wrecks
  the architecture.
  order_score = fraction of concordant pairs, among target buds that were
  correctly committed, between their required order (from line 3) and their
  actual ACT_ORDER ranks (1.0 when <= 1 such bud).
  F = set_score * (0.5 + 0.5*order_score).
  Internal baseline B = F achieved by the checker's own "release everything,
  always" construction (r_t = 1 for all t) -- every bud passes its founding
  check and the plant grows fully bushy in creation order.
    sc    = min(1000, 100*F / max(1e-9, B))
    Ratio = max(0, sc) / 1000
"""
import sys, math

TOL = 1e-9
BETA2 = 0.15
ALPHA = 0.5


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    it = iter(toks)
    H = int(next(it))
    BUDGET = int(next(it))
    K = int(next(it))
    c = [int(next(it)) for _ in range(H)]
    T = [int(next(it)) for _ in range(K)]
    return H, BUDGET, c, T


def simulate(H, c, r):
    """r: 0-indexed list length H.  Returns act_time dict (1..H -> tick or
    None) and act_order (list of committed bud ids, earliest first)."""
    PS = [0] * (H + 1)
    for t in range(1, H + 1):
        PS[t] = PS[t - 1] + r[t - 1]
    act_time = {}
    for i in range(1, H + 1):
        if r[i - 1] == 0:
            act_time[i] = None
            continue
        need = PS[i - 1] + c[i - 1]
        if need <= PS[H]:
            lo, hi = i, H
            while lo < hi:
                mid = (lo + hi) // 2
                if PS[mid] >= need:
                    hi = mid
                else:
                    lo = mid + 1
            act_time[i] = lo
        else:
            act_time[i] = None
    activated = [(t, -i) for i, t in act_time.items() if t is not None]
    activated.sort()
    act_order = [-negi for (t, negi) in activated]
    return act_time, act_order


def score_topology(H, c, T, r):
    act_time, act_order = simulate(H, c, r)
    sim_active = set(i for i in act_time if act_time[i] is not None)
    Tset = set(T)
    correct = sim_active & Tset
    precision = (len(correct) / len(sim_active)) if sim_active else 0.0
    recall = (len(correct) / len(Tset)) if Tset else 1.0
    denom = BETA2 * precision + recall
    set_score = ((1 + BETA2) * precision * recall / denom) if denom > 0 else 0.0

    target_filtered = [x for x in T if x in correct]
    sim_rank = {node: idx for idx, node in enumerate(act_order)}
    m = len(target_filtered)
    if m <= 1:
        order_score = 1.0
    else:
        concordant = 0
        total = 0
        for a in range(m):
            for b in range(a + 1, m):
                total += 1
                if sim_rank[target_filtered[a]] < sim_rank[target_filtered[b]]:
                    concordant += 1
        order_score = concordant / total if total else 1.0

    F = set_score * (ALPHA + (1 - ALPHA) * order_score)
    return F


def baseline_F(H, c, T):
    r_base = [1] * H
    return score_topology(H, c, T, r_base)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    H, BUDGET, c, T = read_instance(sys.argv[1])

    try:
        with open(sys.argv[2], "rb") as fh:
            raw = fh.read(1 << 20)
    except Exception:
        fail("cannot read output")
    text = raw.decode("utf-8", "replace")
    toks = text.split()
    if len(toks) != H:
        fail("need exactly %d tokens, got %d" % (H, len(toks)))

    r = [0] * H
    for idx, tok in enumerate(toks):
        try:
            v = int(tok)
        except Exception:
            fail("token %d unparseable (must be integer 0/1)" % (idx + 1))
        if v not in (0, 1):
            fail("token %d = %s out of range {0,1}" % (idx + 1, tok))
        r[idx] = v

    suppressed = H - sum(r)
    if suppressed > BUDGET:
        fail("growth budget exceeded: suppressed %d > BUDGET %d" % (suppressed, BUDGET))

    F = score_topology(H, c, T, r)
    if not math.isfinite(F) or F < 0:
        fail("degenerate score")

    B = baseline_F(H, c, T)
    if not math.isfinite(B) or B <= 0.0:
        B = 1e-6

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    if sc < 0.0:
        sc = 0.0
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
