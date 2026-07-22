#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0614 -- "Synergy Bin Packing: Dual-Priced Assignment with
Hidden Co-location Bonuses" (family: synergy-multiknapsack-dual; format B, quality-metric;
theme: pack items into capacity-limited bins where co-locating certain pairs pays a bonus).

THEME.  N items must be assigned to M capacity-limited bins (or left out).  Item i has a
weight w[i] and a standalone value v[i].  A public SYNERGY TABLE lists triples (i, j, s):
placing BOTH members of the pair in the SAME bin earns an extra bonus s ON TOP of the two
standalone values.  Value is therefore NON-SEPARABLE -- what an item is worth depends on
what shares its bin.  The operator outputs one assignment; the evaluator scores it.

OBJECTIVE (maximize, deterministic).  For an assignment a (a[i] in {-1,0..M-1}, -1 = left
out) that respects EVERY bin capacity:
    obj(a) = sum_{i: a[i]!=-1} v[i]  +  sum_{(i,j,s) in syn : a[i]==a[j]!=-1} s
Any capacity violation, out-of-range bin, wrong shape, non-integer / NaN entry -> the
assignment is INFEASIBLE and scores 0.0 on that instance.

WHY THE THREE MECHANISMS ARE FORCED.
  * lagrangian-dual-price -- there are several bins and one shared scarce resource
    (capacity).  Deciding WHICH items to admit at all is an aggregate knapsack whose clean
    handle is a price mu on weight: admit a block iff value - mu*weight > 0, and raise mu
    until admitted weight fits the bins.
  * synergy-aware-rounding -- because value is non-separable, pricing/rounding at the
    single-item granularity throws away every bonus.  The fix is to treat a synergistic
    pair as ONE merged super-item (value v_i+v_j+s, weight w_i+w_j) during selection and
    rounding, so the dual keeps a pair together only when it is JOINTLY worthwhile.
  * ejection-chain-repair -- first-fit packing leaves stranded capacity; a high-value
    super-item that no bin can hold is admitted by EJECTING the lowest-value occupants
    (which then relocate), a short chain that repairs the packing without a full re-solve.

INNOVATION HOOK (what `strong` exploits).  Price bin capacity with a Lagrange multiplier
and treat synergistic pairs as merged super-items during rounding, so the dual makes room
for pairs that are only jointly worthwhile.  TRAP instances hide most of the value inside
synergy pairs whose INDIVIDUAL value/weight ratios are mediocre, while scattering
high-ratio "decoy" singletons that soak up capacity.  A ratio-greedy pass (sort by v/w,
first-fit) grabs the decoys, fragments capacity, and never co-locates a synergy pair -- so
it collects almost none of the bonus mass, landing far below the dual+super-item strategy.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "M": int,
             "C":   [c_0 ... c_{M-1}],          # bin capacities (>0)
             "w":   [w_0 ... w_{N-1}],          # item weights (>0)
             "v":   [v_0 ... v_{N-1}],          # item standalone values (>=0)
             "syn": [[i, j, s], ...]}           # co-location bonuses (i<j, s>0)
  stdout: ONE JSON object:
            {"assign": [b_0 ... b_{N-1}]}        # b_i in {-1, 0 .. M-1}

  VALID iff `assign` is a list of exactly N integers, each -1 or a bin index in [0,M),
  and every bin's total assigned weight <= its capacity.  Any violation, crash, timeout,
  or non-JSON -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance:
    U = sum(v) + sum(s over syn)          # loose, always-unreachable upper bound
    obj = obj(a) for the (feasible) candidate assignment, else 0
    r = clamp( 0.1 + 0.9 * obj / U , 0, 1)
  Leaving every item out scores exactly 0.1 (obj=0).  U assumes every item admitted AND
  every pair co-located, which capacity forbids, so even an optimal plan keeps r well
  below 1.0 -- headroom stays open above the reference solutions.  Final score = mean of r
  over 10 fixed seeded instances (4 traps + 3 mixed + 3 dense-chain held-out).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  U and obj are computed by
THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _trap(seed, n_pairs, n_decoy, M, cap_frac, ratio_hi, ratio_med, syn_mult):
    """TRAP: most value hidden in synergy pairs whose individual v/w is mediocre, plus
    high-ratio decoy singletons that a ratio-greedy pass grabs first (fragmenting
    capacity so the pairs are never co-located)."""
    nx = _rng(seed)
    w, v, syn = [], [], []
    # synergy pairs: two mediocre-ratio items + a large joint bonus
    for _ in range(n_pairs):
        wa = nx(12, 20)
        wb = nx(12, 20)
        va = ratio_med * wa
        vb = ratio_med * wb
        i = len(w); w.append(wa); v.append(va)
        j = len(w); w.append(wb); v.append(vb)
        s = int(syn_mult * (va + vb) * nx(90, 115) / 100.0)
        syn.append([i, j, s])
    # decoy singletons: high v/w ratio, no synergy
    for _ in range(n_decoy):
        wd = nx(10, 18)
        w.append(wd)
        v.append(ratio_hi * wd)
    N = len(w)
    totw = sum(w)
    C = totw * cap_frac // M
    caps = [int(C)] * M
    return {"N": N, "M": M, "C": caps, "w": w, "v": v, "syn": syn}


def _mixed(seed, n_pairs, n_decoy, M, cap_frac, ratio_hi, ratio_med, syn_mult):
    """MIXED: modest synergy (super-item ratio no better than the decoys), so the
    ratio-greedy recipe is close to optimal here -- the insight barely helps.  Keeps the
    ladder honest (not every instance is a trap)."""
    nx = _rng(seed)
    w, v, syn = [], [], []
    for _ in range(n_pairs):
        wa = nx(11, 19); wb = nx(11, 19)
        va = ratio_med * wa; vb = ratio_med * wb
        i = len(w); w.append(wa); v.append(va)
        j = len(w); w.append(wb); v.append(vb)
        s = int(syn_mult * (va + vb) * nx(85, 110) / 100.0)
        syn.append([i, j, s])
    for _ in range(n_decoy):
        wd = nx(10, 18)
        w.append(wd); v.append(ratio_hi * wd)
    N = len(w)
    C = sum(w) * cap_frac // M
    return {"N": N, "M": M, "C": [int(C)] * M, "w": w, "v": v, "syn": syn}


def _dense(seed, n_chain, chain_len, n_decoy, M, cap_frac, ratio_med, syn_mult):
    """DENSE / held-out: synergy CHAINS (item shared by several bonuses) so capturing the
    bonus mass needs whole small clusters co-located, not just isolated pairs.  Tests that
    the super-item idea generalizes past single pairs."""
    nx = _rng(seed)
    w, v, syn = [], [], []
    for _ in range(n_chain):
        ids = []
        for _k in range(chain_len):
            wk = nx(9, 15)
            idx = len(w); w.append(wk); v.append(ratio_med * wk)
            ids.append(idx)
        # chain bonuses along consecutive members
        for a in range(chain_len - 1):
            i, j = ids[a], ids[a + 1]
            s = int(syn_mult * (v[i] + v[j]) * nx(80, 105) / 100.0)
            syn.append([i, j, s])
    for _ in range(n_decoy):
        wd = nx(10, 16)
        w.append(wd); v.append(5 * wd)
    N = len(w)
    C = sum(w) * cap_frac // M
    return {"N": N, "M": M, "C": [int(C)] * M, "w": w, "v": v, "syn": syn}


def _build_instances():
    out = []
    specs = [
        # name, kind, params...
        ("trap1", _trap, dict(seed=61401, n_pairs=14, n_decoy=26, M=5, cap_frac=0.52, ratio_hi=4, ratio_med=2, syn_mult=4.5)),
        ("trap2", _trap, dict(seed=61402, n_pairs=16, n_decoy=30, M=6, cap_frac=0.50, ratio_hi=4, ratio_med=2, syn_mult=5.0)),
        ("trap3", _trap, dict(seed=61403, n_pairs=12, n_decoy=22, M=4, cap_frac=0.55, ratio_hi=5, ratio_med=2, syn_mult=4.5)),
        ("trap4", _trap, dict(seed=61404, n_pairs=18, n_decoy=28, M=6, cap_frac=0.48, ratio_hi=4, ratio_med=2, syn_mult=4.0)),
        ("mix1", _mixed, dict(seed=61411, n_pairs=12, n_decoy=28, M=5, cap_frac=0.55, ratio_hi=5, ratio_med=2, syn_mult=0.6)),
        ("mix2", _mixed, dict(seed=61412, n_pairs=14, n_decoy=30, M=6, cap_frac=0.52, ratio_hi=5, ratio_med=2, syn_mult=0.7)),
        ("mix3", _mixed, dict(seed=61413, n_pairs=10, n_decoy=24, M=4, cap_frac=0.58, ratio_hi=6, ratio_med=2, syn_mult=0.5)),
        ("dense1", _dense, dict(seed=61421, n_chain=8, chain_len=3, n_decoy=18, M=5, cap_frac=0.50, ratio_med=2, syn_mult=3.2)),
        ("dense2", _dense, dict(seed=61422, n_chain=7, chain_len=4, n_decoy=16, M=5, cap_frac=0.48, ratio_med=2, syn_mult=3.0)),
        ("dense3", _dense, dict(seed=61423, n_chain=9, chain_len=3, n_decoy=20, M=6, cap_frac=0.50, ratio_med=2, syn_mult=3.5)),
    ]
    for name, fn, kw in specs:
        inst = fn(**kw)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- scoring ------------------------------------
def _upper_bound(inst):
    return float(sum(inst["v"]) + sum(t[2] for t in inst["syn"]))


def _score(inst, answer):
    """Return (feasible, obj).  Strict validation; any violation -> (False, 0)."""
    N, M = inst["N"], inst["M"]
    C, w, v, syn = inst["C"], inst["w"], inst["v"], inst["syn"]
    if not isinstance(answer, dict):
        return False, 0.0
    a = answer.get("assign")
    if not isinstance(a, list) or len(a) != N:
        return False, 0.0
    binw = [0.0] * M
    for i in range(N):
        b = a[i]
        if isinstance(b, bool) or not isinstance(b, int):
            return False, 0.0
        if b == -1:
            continue
        if b < 0 or b >= M:
            return False, 0.0
        binw[b] += w[i]
    for b in range(M):
        if binw[b] > C[b] + 1e-9:
            return False, 0.0
    obj = 0.0
    for i in range(N):
        if a[i] != -1:
            obj += v[i]
    for (i, j, s) in syn:
        if a[i] != -1 and a[i] == a[j]:
            obj += s
    return True, float(obj)


# ----------------------------- driver -------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        U = _upper_bound(inst)
        if U <= 1e-9:
            U = 1e-9
        public = {"name": inst["name"], "N": inst["N"], "M": inst["M"],
                  "C": list(inst["C"]), "w": list(inst["w"]), "v": list(inst["v"]),
                  "syn": [list(t) for t in inst["syn"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = _score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * obj / U
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
