#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0617 -- "Cold Vault: Place What Can Never Move"
(family: no-compaction-heap-allocator; format B, quality-metric).

THEME.  A write-once cold-storage vault packs data blocks into a single linear
address space.  Blocks arrive over time (birth) and are later reclaimed (death);
while a block is alive it owns a fixed, contiguous byte range.  The vault has NO
compaction and NO relocation: once you place a block you may never move it, ever.
Two blocks whose LIFETIMES overlap in time must own DISJOINT byte ranges.  When a
block dies its bytes are freed and may be reused -- but only if a later block
actually fits in the hole you left behind.  The vault's cost is its HIGH-WATER
MARK: the largest address it ever has to touch.  Minimize it.

This is offline Dynamic Storage Allocation (interval memory allocation): each
block is a fixed time interval [birth, death) with a size; you choose its OFFSET;
time-overlapping blocks get non-overlapping byte ranges; minimize
    peak = max_i (offset_i + size_i).
Because placements can never be undone, a block dropped in the wrong spot strands
a permanent hole -- fragmentation you can never compact away.

THREE MECHANISMS (all shape the score):
  * fragmentation-no-compaction : offsets are frozen; a badly placed long-lived
    block permanently splits the free space above and below it.
  * lifetime-colocation         : the stream is built from COHORTS -- groups of
    blocks that share an exact death time -- interleaved with long-lived "spine"
    blocks that live to the end.  Co-locating a cohort makes it vacate a single
    contiguous hole on death; scattering it strands many small holes forever.
  * staged-capacity-reveal      : demand arrives in STAGES; each stage adds a new
    cohort plus new spine blocks, so the memory a good placement needs is revealed
    gradually.  Reusing an earlier cohort's vacated region for a later stage is
    what keeps the high-water mark flat instead of climbing every stage.

INNOVATION HOOK.  Group allocations by anticipated FREE-TIME, not by size:
segregate the long-lived spine low and pack each equal-death cohort together so
its hole reopens contiguously for the next stage.  A size-only or arrival-order
first-fit ignores death times, scatters the spine through addresses opened by
dying cohorts, and permanently fragments the vault.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": M,
             "blocks": [ {"size": s>=1, "birth": b, "death": d}, ... ]}   # b < d
          blocks[i] occupies memory during time interval [birth, death).
  stdout: ONE JSON object:
            {"offset": [o_0, ..., o_{M-1}]}   # o_i >= 0 integer, block i's base address

  VALID iff `offset` is a list of exactly M non-negative integers AND for every
  pair (i, j) whose time intervals overlap (max(b_i,b_j) < min(d_i,d_j)) the byte
  ranges [o_i, o_i+s_i) and [o_j, o_j+s_j) are disjoint.  Invalid output, wrong
  length, an overlap, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    b_fresh   = sum of all block sizes = the fresh-slab high-water mark (never
                reuse any freed byte; the do-nothing vault).
    lb        = max concurrent live bytes at any instant (a valid lower bound
                on ANY feasible high-water mark).
    lb_anchor = floor(0.65 * lb)   -- shrunk BELOW the true floor so even a
                perfect placement cannot saturate the score.
    q_cand    = candidate high-water mark.
    r = clamp(0.1 + 0.9 * (b_fresh - q_cand) / (b_fresh - lb_anchor), 0, 1)
  A do-nothing fresh-slab layout scores exactly 0.1.  Reusing freed regions well
  drives q_cand toward lb and lifts r substantially, but the shrunk anchor keeps
  even an optimal placement short of 1.0, leaving headroom above any reference.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The baseline
and the validation run in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _gen_staged(ni, R, gap, spine_size, short_base, short_variety, ns_lo, ns_hi,
                big_frac_pct):
    """Staged interleaved-lifetime stream that FORCES first-fit fragmentation.

    Each round r (base = r*gap) issues, IN ARRIVAL ORDER, a 1:1 interleaving
      spine, short, spine, short, ...   (nshort pairs)
    where every 'spine' block (size spine_size) lives to the very end
    (death = T_end) and every 'short' block (size short_size, which VARIES by
    round so same-size reuse across rounds is not a free lunch) dies together
    at sdeath = base + gap//2.  Because a spine block sits between EVERY
    consecutive pair of shorts, an arrival-order / size-blind allocator that
    just drops each block at the lowest free address ends up placing shorts
    into SEPARATE addresses isolated by still-alive spine on both sides. When
    the whole short cohort dies at once, those holes stay non-adjacent
    forever (no compaction) instead of merging into one span.

    Right after (birth = sdeath), a SINGLE big block arrives, sized to
    `big_frac_pct` percent of the cohort's total footprint (nshort *
    short_size) -- too big for any one isolated short-hole, but small enough
    to fit the cohort's FULL footprint if it were vacated as one contiguous
    run.  It dies at base + gap, before the next round starts.

    An allocator that groups by anticipated free-time -- segregating the
    long-lived spine into its own low band and packing each round's
    equal-death short cohort contiguously -- vacates exactly one contiguous
    hole per round that the big block reuses, so the high-water mark stays
    near the concurrent-demand floor.  An arrival-order allocator can never
    offer the big block anywhere to go but a brand-new address above every
    round's isolated debris: the high-water mark grows roughly by one whole
    cohort's worth EVERY round -- fragmentation that compounds and can never
    be compacted away.
    """
    blocks = []
    T_end = gap * (R + 2)
    for r in range(R):
        base = r * gap
        nshort = ni(ns_lo, ns_hi)
        short_size = short_base + (r % short_variety)   # varies by round
        sdeath = base + gap // 2
        for _ in range(nshort):                       # 1:1 INTERLEAVE
            blocks.append({"size": spine_size, "birth": base, "death": T_end})
            blocks.append({"size": short_size, "birth": base, "death": sdeath})
        cohort_total = nshort * short_size
        big_size = max(short_size + 1, (cohort_total * big_frac_pct) // 100)
        blocks.append({"size": big_size, "birth": sdeath, "death": base + gap})
    return blocks


def _gen_uniform(ni, M, T, smin, smax, lmin, lmax):
    """Unstructured random intervals -- no planted cohort/spine structure."""
    blocks = []
    for _ in range(M):
        b = ni(0, T)
        d = b + ni(lmin, lmax)
        blocks.append({"size": ni(smin, smax), "birth": b, "death": d})
    return blocks


def _build_instances():
    specs = [
        # (name, seed, kind, params)   staged = interleaved-lifetime fragmentation traps.
        # spine_size != short_size (varies per round) so type-mismatched holes
        # can't be recycled for free; big_frac_pct sizes the single reuse-cohort
        # block relative to that round's short-cohort footprint.
        ("vault_stage_a", 41, "staged", dict(R=8,  gap=20, spine_size=5, short_base=6,
                                             short_variety=3, ns_lo=6, ns_hi=9, big_frac_pct=80)),
        ("vault_stage_b", 57, "staged", dict(R=10, gap=20, spine_size=5, short_base=7,
                                             short_variety=3, ns_lo=7, ns_hi=10, big_frac_pct=78)),
        ("vault_stage_c", 83, "staged", dict(R=7,  gap=22, spine_size=6, short_base=8,
                                             short_variety=4, ns_lo=7, ns_hi=11, big_frac_pct=82)),
        ("vault_stage_d", 109, "staged", dict(R=11, gap=18, spine_size=4, short_base=6,
                                             short_variety=3, ns_lo=6, ns_hi=9, big_frac_pct=76)),
        ("vault_stage_e", 131, "staged", dict(R=9,  gap=22, spine_size=6, short_base=9,
                                             short_variety=4, ns_lo=8, ns_hi=11, big_frac_pct=80)),
        ("vault_stage_f", 167, "staged", dict(R=12, gap=18, spine_size=5, short_base=6,
                                             short_variety=3, ns_lo=6, ns_hi=8, big_frac_pct=78)),
        ("vault_stage_g", 191, "staged", dict(R=8,  gap=20, spine_size=5, short_base=7,
                                             short_variety=3, ns_lo=7, ns_hi=10, big_frac_pct=82)),
        # harder / larger held-out staged instance
        ("vault_stage_h", 223, "staged", dict(R=16, gap=20, spine_size=5, short_base=7,
                                             short_variety=4, ns_lo=9, ns_hi=13, big_frac_pct=80)),
        # unstructured controls (greedy ~ strong): keeps the family honest
        ("vault_unif_a",  251, "uniform", dict(M=70, T=90, smin=3, smax=16, lmin=6, lmax=40)),
        ("vault_unif_b",  277, "uniform", dict(M=85, T=110, smin=4, smax=18, lmin=8, lmax=48)),
    ]
    out = []
    for name, seed, kind, p in specs:
        ni = _rng(seed)
        if kind == "staged":
            blocks = _gen_staged(ni, **p)
        else:
            blocks = _gen_uniform(ni, **p)
        out.append({"name": name, "n": len(blocks), "blocks": blocks})
    return out


# ----------------------------- validation ----------------------------------
def _peak(inst, answer):
    """Validate; return candidate high-water mark or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    off = answer.get("offset")
    if not isinstance(off, list):
        return None
    blocks = inst["blocks"]
    M = inst["n"]
    if len(off) != M:
        return None
    o = []
    for v in off:
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return None
        o.append(v)
    # pairwise: time-overlapping blocks must have disjoint byte ranges
    # sort by birth to prune; still O(M^2) worst case but M is small
    idx = sorted(range(M), key=lambda i: blocks[i]["birth"])
    for a in range(M):
        i = idx[a]
        bi, di, si, oi = blocks[i]["birth"], blocks[i]["death"], blocks[i]["size"], o[i]
        for b in range(a + 1, M):
            j = idx[b]
            bj = blocks[j]["birth"]
            if bj >= di:                    # no later block (by birth) can overlap i in time
                break
            dj, sj, oj = blocks[j]["death"], blocks[j]["size"], o[j]
            if bi < dj and bj < di:         # time overlap
                if oi < oj + sj and oj < oi + si:   # byte overlap
                    return None
    peak = 0
    for i in range(M):
        e = o[i] + blocks[i]["size"]
        if e > peak:
            peak = e
    return peak


def _concurrent_peak(blocks):
    """Max total live bytes at any instant -- a valid (but generally loose)
    lower bound on ANY feasible high-water mark, since concurrently-alive
    blocks can never share bytes regardless of placement skill."""
    events = []
    for bl in blocks:
        events.append((bl["birth"], 1, bl["size"]))
        events.append((bl["death"], -1, bl["size"]))
    events.sort()
    cur = 0
    peak = 0
    for _, typ, s in events:
        if typ == 1:
            cur += s
            if cur > peak:
                peak = cur
        else:
            cur -= s
    return peak


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        blocks = inst["blocks"]
        b_fresh = sum(bl["size"] for bl in blocks)                # weak do-nothing anchor -> 0.1
        lb = max(1, _concurrent_peak(blocks))
        # the TRUE concurrent-demand floor is occasionally exactly reachable by a
        # near-optimal lifetime-aware placement on this family's nested/staged
        # structure, so anchor the 1.0 point BELOW it (shrunk) -- this keeps
        # real headroom above even a perfect reference, per the innovation
        # addendum's "strong <= 0.92" requirement.
        lb_anchor = max(1, (lb * 65) // 100)
        public = {"name": inst["name"], "n": inst["n"],
                  "blocks": [dict(bl) for bl in blocks]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _peak(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        denom = b_fresh - lb_anchor
        if denom < 1e-9:
            denom = 1e-9
        r = 0.1 + 0.9 * (b_fresh - q_cand) / denom
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
