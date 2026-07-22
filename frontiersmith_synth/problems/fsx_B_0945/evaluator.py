#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0945 -- "Rotational Grazing on a Paddock Grid"
(family: regrowth-timed-rotation; format B, quality-metric).

THEME. A herd grazes a grid of R x C paddocks. Grass in every paddock regrows
logistically AND diffuses to/from its 4-neighbours (lush neighbours reseed a
bare paddock; a paddock that keeps donating grass to a bare neighbour drains
itself too). Each day the herd occupies exactly one paddock and eats up to a
fixed daily requirement from it; if that paddock is short, the shortfall must
be covered with costly supplemental feed. Moving the herd to a different
paddock costs a fixed relocation fee plus a per-Manhattan-distance fee (herd
weight loss + labour). The whole grid evolves (regrowth + diffusion) every
single day, whether or not the herd is there.

MECHANISM COMPOSITION.
  - herd-relocation-cost: moving is not free; jumping far is extra costly.
  - spatial-regrowth-diffusion: grass follows g += r*g*(1-g) + D*sum(neighbour
    gradients), so a paddock's recovery speed depends on BOTH the regrowth
    constant r and how lush its neighbours currently are (diffusion can help
    OR hurt: repeatedly draining one paddock also drains the neighbours that
    keep "lending" it grass).

INNOVATION HOOK. The two mechanisms interact through TIME: a paddock that is
grazed needs roughly tau days of undisturbed regrowth+diffusion to become
worth visiting again. A schedule that revisits paddocks on a period matched to
that recovery time constant tau (and tours them along a short, compact path)
gets full paddocks with few moves. A schedule that reactively chases "whichever
paddock looks greenest today" ignores tau and the travel graph: it typically
either (a) ping-pongs between far-apart patches that alternate in short-term
greenness (huge relocation cost) or (b) keeps re-draining a diffusion-fed
local cluster until the cluster and its donors collapse together, forcing
costly supplemental feed later in the season.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the
          exact schema (R, C, T, r, D, D_req, move_fixed, move_per_dist,
          supp_mult, start, g0). Every field the simulation needs is public;
          the candidate can replay the exact dynamics itself.
  stdout: ONE JSON object {"visits": [c_0, ..., c_{T-1}]}, c_t in [0, R*C).

SCORING (deterministic; no wall-time). The evaluator itself replays the
season with the candidate's visit sequence to get obj_cand, and computes two
references purely internally:
    obj_base = objective of the "never move" (stay at `start` forever) policy
    obj_ub   = T * D_req   -- a loose, generally unreachable ceiling (the herd
               would need its full daily requirement, every day, forever,
               for zero relocation and zero supplement cost)
and normalizes with the same affine-anchor shape used across this corpus:
    r = clamp(0.1 + 0.9 * (obj_cand - obj_base) / max(eps, obj_ub - obj_base), 0, 1)
Matching "stay put" scores ~0.1; reaching the unreachable ceiling scores 1.0;
doing worse than staying put scores below 0.1.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the public instance. obj_base/obj_ub
are computed by THIS parent process, so a frame-walking candidate learns
nothing useful.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 20) % (hi - lo + 1)

    def nxtf():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return nxt, nxtf


# ----------------------------- grid geometry --------------------------------
def _neighbors(i, R, C):
    row, col = divmod(i, C)
    out = []
    if row > 0:
        out.append(i - C)
    if row < R - 1:
        out.append(i + C)
    if col > 0:
        out.append(i - 1)
    if col < C - 1:
        out.append(i + 1)
    return out


def _neighbor_table(R, C):
    P = R * C
    return [_neighbors(i, R, C) for i in range(P)]


def _manhattan(a, b, C):
    ra, ca = divmod(a, C)
    rb, cb = divmod(b, C)
    return abs(ra - rb) + abs(ca - cb)


# ----------------------------- g0 patterns ----------------------------------
def _build_g0(seed, R, C, pattern):
    P = R * C
    _, nxtf = _rng(seed)
    if pattern == "uniform_mid":
        return [0.5] * P
    if pattern == "uniform_high":
        return [0.85] * P
    if pattern == "depleted":
        return [0.05 + 0.02 * nxtf() for _ in range(P)]
    if pattern == "clustered_two":
        g = [0.12 + 0.05 * nxtf() for _ in range(P)]
        # two compact hot clusters near opposite corners
        for (r0, c0) in ((0, 0), (R - 1, C - 1)):
            for dr in range(min(2, R)):
                for dc in range(min(2, C)):
                    rr = r0 + dr if r0 == 0 else r0 - dr
                    cc = c0 + dc if c0 == 0 else c0 - dc
                    if 0 <= rr < R and 0 <= cc < C:
                        g[rr * C + cc] = 0.9 + 0.05 * nxtf()
        return g
    if pattern == "clustered_one":
        g = [0.15 + 0.05 * nxtf() for _ in range(P)]
        r0, c0 = R // 2, C // 2
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < R and 0 <= cc < C:
                    g[rr * C + cc] = 0.9 + 0.05 * nxtf()
        return g
    # "random_patchy"
    return [0.1 + 0.8 * nxtf() for _ in range(P)]


def _build_instances():
    """Deterministic instance family: 10 seeded grid/season configurations.
    Fields: (name, R, C, T, r, D, D_req, move_fixed, move_per_dist, supp_mult,
             start, pattern, seed, trap)
    """
    specs = [
        ("field_A", 4, 4, 70, 0.16, 0.05, 0.20, 0.015, 0.008, 1.4, 0, "uniform_mid", 11, False),
        ("field_B", 4, 4, 70, 0.10, 0.02, 0.18, 0.012, 0.006, 1.3, 5, "random_patchy", 22, False),
        ("field_C", 3, 7, 90, 0.14, 0.03, 0.22, 0.06, 0.04, 1.6, 0, "clustered_two", 33, True),
        ("field_D", 6, 6, 100, 0.12, 0.04, 0.20, 0.05, 0.035, 1.5, 14, "clustered_two", 44, True),
        ("field_E", 5, 5, 90, 0.07, 0.06, 0.18, 0.045, 0.03, 1.4, 12, "clustered_one", 55, True),
        ("field_F", 5, 5, 90, 0.08, 0.07, 0.20, 0.05, 0.032, 1.5, 12, "clustered_one", 66, True),
        ("field_G", 4, 5, 80, 0.18, 0.02, 0.22, 0.02, 0.01, 1.7, 2, "depleted", 77, False),
        ("field_H", 4, 6, 100, 0.13, 0.03, 0.19, 0.015, 0.008, 1.4, 9, "random_patchy", 88, False),
        # held-out / larger, harder instances
        ("field_I", 6, 7, 120, 0.09, 0.05, 0.21, 0.055, 0.036, 1.5, 20, "clustered_two", 99, True),
        ("field_J", 5, 6, 110, 0.11, 0.04, 0.20, 0.015, 0.008, 1.4, 15, "random_patchy", 111, False),
    ]
    out = []
    for name, R, C, T, r, D, D_req, mf, mpd, sm, start, pattern, seed, trap in specs:
        g0 = _build_g0(seed, R, C, pattern)
        out.append({"name": name, "R": R, "C": C, "T": T, "r": r, "D": D,
                    "D_req": D_req, "move_fixed": mf, "move_per_dist": mpd,
                    "supp_mult": sm, "start": start, "g0": g0, "trap": trap})
    return out


# ----------------------------- dynamics -------------------------------------
def _evolve(g, nbrs, r, D):
    P = len(g)
    out = [0.0] * P
    for i in range(P):
        gi = g[i]
        s = 0.0
        for j in nbrs[i]:
            s += g[j] - gi
        val = gi + r * gi * (1.0 - gi) + D * s
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        out[i] = val
    return out


def _simulate(inst, visits):
    """Replay one season under a fixed visit sequence. Returns objective (float)."""
    R, C, T = inst["R"], inst["C"], inst["T"]
    r, D, D_req = inst["r"], inst["D"], inst["D_req"]
    mf, mpd, sm = inst["move_fixed"], inst["move_per_dist"], inst["supp_mult"]
    nbrs = _neighbor_table(R, C)
    g = list(inst["g0"])
    prev = inst["start"]
    total_intake = 0.0
    total_supp = 0.0
    total_move = 0.0
    for t in range(T):
        c = visits[t]
        if c != prev:
            d = _manhattan(c, prev, C)
            total_move += mf + mpd * d
        eaten = g[c] if g[c] < D_req else D_req
        shortfall = D_req - eaten
        total_intake += eaten
        total_supp += sm * shortfall
        g[c] = g[c] - eaten
        g = _evolve(g, nbrs, r, D)
        prev = c
    return total_intake - total_supp - total_move


def _stay_put(inst):
    return [inst["start"]] * inst["T"]


# ----------------------------- validation ------------------------------------
def _validate_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    visits = answer.get("visits")
    if not isinstance(visits, list):
        return None
    T = inst["T"]
    P = inst["R"] * inst["C"]
    if len(visits) != T:
        return None
    out = []
    for c in visits:
        if isinstance(c, bool) or not isinstance(c, int):
            return None
        if c < 0 or c >= P:
            return None
        out.append(c)
    return out


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        obj_base = _simulate(inst, _stay_put(inst))
        obj_ub = inst["T"] * inst["D_req"]
        denom = obj_ub - obj_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "R": inst["R"], "C": inst["C"], "T": inst["T"],
                  "r": inst["r"], "D": inst["D"], "D_req": inst["D_req"],
                  "move_fixed": inst["move_fixed"], "move_per_dist": inst["move_per_dist"],
                  "supp_mult": inst["supp_mult"], "start": inst["start"], "g0": list(inst["g0"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            visits = _validate_answer(inst, ans)
        except Exception:
            visits = None
        if visits is None:
            vec.append(0.0)
            continue
        try:
            obj_cand = _simulate(inst, visits)
        except Exception:
            vec.append(0.0)
            continue
        val = 0.1 + 0.9 * (obj_cand - obj_base) / denom
        if not (val == val) or val in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        vec.append(val)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
