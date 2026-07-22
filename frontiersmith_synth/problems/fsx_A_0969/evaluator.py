#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0969 -- "Inspector Who Must Guarantee, Not Guess"
(family: probe-certified-tolerance; format B, quality-metric).

THEME.  A quality inspector checks a manufactured panel against its nominal design.  The
panel occupies the square [0,S]x[0,S].  Its nominal (as-designed) height is a fully public,
exactly-computable smooth surface z_nom(x,y) (a low-degree polynomial with published
coefficients).  The manufactured panel's ACTUAL height is z(x,y) = z_nom(x,y) + g(x,y), where
g -- the "deviation field" -- is HIDDEN.  g is built from a handful of disjoint, compactly
supported "defect" bumps (some sharp cones with slope EXACTLY the published Lipschitz constant
Lc, some softer parabolic bumps with slope held strictly BELOW Lc), so g itself is globally
Lc-Lipschitz, and |g| never exceeds a published cap M_CAP.

You may spend a fixed budget of PROBES, each an (x,y) you choose; a probe returns the EXACT
z(x,y) there (no noise -- determinism is exact).  Probing is ADAPTIVE and ROUND-based: you
see the running history (every point probed so far and its value, including a small FREE
"pilot" batch already on file) before choosing where to probe next.  After your budget is
spent you must output a single number `bound` D, your CERTIFICATE.

VERIFICATION GRID (removes ANY ambiguity about "the panel"): the certificate is checked, not
against a true continuum, but against a fixed, deterministic N_INSPECT x N_INSPECT grid of
inspection points spanning the panel (cell-centre sampled: x = S*(i+0.5)/N_INSPECT for
i in 0..N_INSPECT-1, same for y) -- published exactly by this formula, so both the grader and
any candidate compute the identical set. For any probe p_i with observed deviation
dev_i = z(p_i)-z_nom(p_i), Lipschitz continuity guarantees |g(q)| <= |dev_i| + Lc*dist(q,p_i)
for every q, so the tightest bound honestly justifiable from a probe set, evaluated at
inspection point q, is min(M_CAP, min_i(|dev_i| + Lc*dist(q,p_i))); the tightest bound over the
WHOLE grid is the max of that over every inspection point q. THE GRADER ITSELF RECOMPUTES this
exact quantity, D_earn, from your ACTUAL probe history after you submit -- your claimed `D`
must satisfy D >= D_earn (a tiny floating-point tolerance only); undercutting D_earn is an
unsound certificate and scores 0 THE INSTANT IT IS CAUGHT, regardless of what D_earn itself
would be. There is therefore no way to "know" the answer without earning it through probes:
D_earn is computed from the REAL history the grader itself produced, never from anything the
candidate merely asserts.

INNOVATION HOOK.  The quantity you must eventually minimise -- D_earn, the worst-case value of
the bound formula above, MAXIMISED over the whole inspection grid -- is a function of where you
probed, not of "how much of the panel you covered" in the abstract. The insight the grader
rewards is playing a minimax game directly against your own bound formula: at EVERY round,
find the inspection point where the CURRENT certified envelope (using every probe read so far)
is largest, and attack exactly that point. With no signal anywhere yet this naturally disperses
probes (the loosest points are simply the farthest from everything known), reproducing good
generic coverage as a side effect -- but the instant a probe lands near a hidden defect, the
envelope right around it becomes the new worst point in the domain, pulling the very next probe
there and the next, closing in over several rounds. No fixed schedule decides this in advance:
the formula itself, recomputed fresh every round, decides whether extending coverage or
refining a discovery is currently worth more. A layout fixed in advance -- even an excellent,
evenly spaced one -- can NEVER revisit a still-loose spot after the fact or spend a single
probe on a reading it never asked for.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called once per round).
  stdin (phase="probe"): {"phase":"probe","domain":{"x_lo","x_hi","y_lo","y_hi"},
      "lipschitz_L","m_cap","n_inspect","z_nom_coeffs":{"a0".."a5"},"budget_total",
      "budget_left","round","rounds_total","max_probes_this_round",
      "history":[{"x","y","z"}, ...]}
    -> return {"probes":[{"x":..,"y":..}, ...]}  (<= max_probes_this_round entries; a point
       outside the domain is SILENTLY SKIPPED -- still charged against budget, no reading; a
       malformed entry anywhere -> 0.0 on the WHOLE instance)
  stdin (phase="certify"): same fields plus the FINAL complete history, no further probing.
    -> return {"bound": D}  (one finite float >= 0)
  Any crash / timeout / non-JSON / wrong shape on ANY call -> 0.0 on that instance.

SCORING (deterministic; no wall-time). After the final `bound` D is submitted, the grader
recomputes D_earn from the SAME history it produced (see above). If D < D_earn - tol, the
instance scores 0 (caught lying). Otherwise quality = clip(1 - D / D0, 0, 1), where D0 is
D_earn computed using ONLY the free pilot batch (zero extra probes spent) -- the grader's own
zero-effort reference. Score r = 0.10 + 0.82*quality, clipped to [0,1] (cap 0.92: headroom is
left above the reference solutions on purpose). Final score = mean of r over 10 fixed seeded
instances. Several instances plant one or more "trap" defects (SOFT, parabolic bumps whose
slope is held strictly BELOW Lc, unlike the sharper generic ones), each seeded near a free
pilot reading but positioned so it stays unreachable by several plausible fixed reference
layouts. Each such pilot point therefore already shows a genuine, nonzero, but INCOMPLETE
reading -- a real clue every candidate sees in round 0 -- yet a probe budget spent purely on a
layout planned without looking at that reading never earns anything more from it, while a
policy that reacts to the clue keeps shrinking the certified value there round after round,
using budget the fixed layout would have spent on generic coverage it no longer needs there.

ISOLATION.  The candidate runs in a FRESH sandboxed subprocess via isorun.run_candidate; it
sees only the public fields above.  Coefficients a0..a5 and Lc/M_CAP/n_inspect ARE public (the
nominal design, the tolerance law, and the verification grid are all known); the hidden
defects (their number, location, height, radius) live only in this parent process.

CLI: python3 evaluator.py <solution.py>
"""
import sys, json, math, random
import isorun

# --------------------------- geometry / protocol constants ------------------------
S = 12.0                       # panel side
PILOT = [(3.0, 3.0), (9.0, 3.0), (3.0, 9.0), (9.0, 9.0)]     # free, on-file QC readings
B_OWN = 32                     # candidate's own adaptive probe budget (a blind 4x4 grid only
                                # spends 16 of these; the rest reward reacting to a reading)
ROUNDS = 32
MAX_PER_ROUND = 1
M_CAP = 11.0                   # published global cap on |deviation|
N_INSPECT = 101                # THE OFFICIAL, EXACT verification grid resolution (both axes)
OFFSET, SPAN = 0.10, 0.82      # r = OFFSET + SPAN*quality  (cap 0.92)
SOUND_TOL_REL, SOUND_TOL_ABS = 1e-4, 1e-4

X_LO, X_HI, Y_LO, Y_HI = 0.0, S, 0.0, S


# ------------------------------- surfaces -------------------------------------------
def z_nom(coeffs, x, y):
    a0, a1, a2, a3, a4, a5 = coeffs
    cx = cy = S / 2.0
    return (a0 + a1 * x + a2 * y + a3 * (x * y) / S
            + a4 * ((x - cx) ** 2) / S + a5 * ((y - cy) ** 2) / S)


def g_value(defects, Lc, x, y):
    # Each defect is (cx, cy, h, sign, kind, r).
    #  'cone': bump(d) = max(0, h - Lc*d)                    -- max slope EXACTLY Lc (tight):
    #          a single probe landing anywhere inside recovers the peak perfectly, so only
    #          WHETHER a probe lands inside matters, not how many.
    #  'soft': bump(d) = h*max(0, 1-(d/r)^2)                 -- max slope 2h/r, built STRICTLY
    #          below Lc: a probe landing inside gives PARTIAL, position-dependent evidence, so
    #          probing progressively closer to the true peak over several rounds keeps
    #          tightening the certified value there -- a single lucky landing is not enough.
    total = 0.0
    for (dx, dy, h, sign, kind, r) in defects:
        d = math.hypot(x - dx, y - dy)
        if kind == "soft":
            if d < r:
                bump = h * (1.0 - (d / r) ** 2)
            else:
                bump = 0.0
        else:
            bump = h - Lc * d
            if bump < 0.0:
                bump = 0.0
        if bump > 0.0:
            total += sign * bump
    return total


def true_z(coeffs, defects, Lc, x, y):
    return z_nom(coeffs, x, y) + g_value(defects, Lc, x, y)


# ------------------------------- envelope / D_earn over the OFFICIAL inspection grid ----
def sup_envelope(history_dev, Lc, n=N_INSPECT):
    best = 0.0
    for i in range(n):
        x = S * (i + 0.5) / n
        for j in range(n):
            y = S * (j + 0.5) / n
            v = M_CAP
            for (px, py, dev) in history_dev:
                d = math.hypot(x - px, y - py)
                val = dev + Lc * d
                if val < v:
                    v = val
            if v > best:
                best = v
    return best


# ------------------------------- deterministic RNG helpers --------------------------
def _rand_defect(rng, existing, margin=0.30, r_lo=0.30, r_hi=0.60, h_lo=0.50, h_hi=1.60,
                  edge_pad=0.30, tries=200):
    for _ in range(tries):
        h = rng.uniform(h_lo, h_hi)
        r = rng.uniform(r_lo, r_hi)
        cx = rng.uniform(edge_pad + r, S - edge_pad - r)
        cy = rng.uniform(edge_pad + r, S - edge_pad - r)
        ok = True
        for (ex, ey, er, _h, _k) in existing:
            if math.hypot(cx - ex, cy - ey) < r + er + margin:
                ok = False
                break
        if ok:
            return (cx, cy, r, h, "cone")
    return None


# Defects tagged as TRAPS must be unreachable by a plausible fixed, blind probe layout a
# "reasonable" naive solver might plan in advance (a regular grid at the candidate's own
# budget resolution) AND unreachable by the free pilot -- so no fixed-in-advance layout can
# ever land inside one. They use the SOFT (parabolic) profile with slope held STRICTLY below
# Lc, so a probe landing inside gives only partial, position-dependent evidence: the
# certified value at the (unknown) peak keeps shrinking as later probes land closer to it,
# rewarding an adaptive policy that follows the evidence over several rounds -- a payoff a
# single, pre-planned landing cannot capture.
def _reference_layout_points():
    pts = list(PILOT)
    for gx in (3, 4):
        lines = [S * i / (gx - 1) for i in range(gx)]
        pts += [(a, b) for a in lines for b in lines]
    return pts


_REFERENCE_POINTS = _reference_layout_points()
_GRID_ONLY_POINTS = [p for p in _REFERENCE_POINTS if p not in PILOT]
TRAP_SOFT_SLOPE_FRAC = 0.55     # actual max slope of a trap bump = TRAP_SOFT_SLOPE_FRAC * Lc


# TRAP defects are seeded NEAR one free pilot reading (so the FREE, round-0 history already
# shows a genuine, nonzero, but INCOMPLETE deviation there -- a real starting clue every
# policy sees) while staying unreachable by any of the regular reference grids. Because the
# soft profile's slope is held strictly below Lc, that pilot reading alone certifies a value
# noticeably looser than the true peak; only a policy that reacts to the clue by probing
# CLOSER keeps shrinking it. A layout planned without looking at the pilot never does that,
# no matter how good its blind spacing is elsewhere.
def _rand_trap_defect(rng, existing, Lc, margin=0.30, r_lo=1.1, r_hi=1.4,
                       edge_pad=0.30, tries=4000):
    for _ in range(tries):
        r = rng.uniform(r_lo, r_hi)
        seed_px, seed_py = PILOT[rng.randrange(len(PILOT))]
        d0 = rng.uniform(0.45 * r, 0.75 * r)          # pilot sits INSIDE the disk, off-centre
        ang = rng.uniform(0.0, 2.0 * math.pi)
        cx = seed_px + d0 * math.cos(ang)
        cy = seed_py + d0 * math.sin(ang)
        if not (edge_pad + r <= cx <= S - edge_pad - r and edge_pad + r <= cy <= S - edge_pad - r):
            continue
        if min(math.hypot(cx - px, cy - py) for (px, py) in _GRID_ONLY_POINTS) < r + 0.30:
            continue
        # the OTHER three pilot points must stay clearly outside the disk (only the seed
        # pilot is meant to see a reading)
        others_ok = all(math.hypot(cx - px, cy - py) > r + 0.30
                         for (px, py) in PILOT if (px, py) != (seed_px, seed_py))
        if not others_ok:
            continue
        h = 0.5 * Lc * r * TRAP_SOFT_SLOPE_FRAC     # max slope 2h/r = Lc*TRAP_SOFT_SLOPE_FRAC < Lc
        ok = True
        for (ex, ey, er, _h, _k) in existing:
            if math.hypot(cx - ex, cy - ey) < r + er + margin:
                ok = False
                break
        if ok:
            return (cx, cy, r, h, "soft")
    return None


def _build_instance(seed, n_traps, n_generic, Lc, h_lo, h_hi, r_lo, r_hi, coeff_scale):
    rng = random.Random(seed)
    coeffs = (rng.uniform(-4.0, 4.0), rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
              rng.uniform(-0.05, 0.05) * coeff_scale, rng.uniform(-0.05, 0.05) * coeff_scale,
              rng.uniform(-0.05, 0.05) * coeff_scale)
    defects = []          # list of (cx, cy, r, h, kind) while building, converted after
    for _ in range(n_traps):
        d = _rand_trap_defect(rng, defects, Lc)
        if d is not None:
            defects.append(d)
    for _ in range(n_generic):
        d = _rand_defect(rng, defects, margin=0.30, r_lo=r_lo, r_hi=r_hi, h_lo=h_lo, h_hi=h_hi)
        if d is not None:
            defects.append(d)
    final_defects = []
    for (cx, cy, r, h, kind) in defects:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        final_defects.append((round(cx, 4), round(cy, 4), round(h, 4), sign, kind, round(r, 4)))
    pilot_dev = []
    for (px, py) in PILOT:
        dev = abs(g_value(final_defects, Lc, px, py))
        pilot_dev.append((px, py, dev))
    D0 = sup_envelope(pilot_dev, Lc)
    return {"seed": seed, "coeffs": [round(c, 5) for c in coeffs], "Lc": round(Lc, 4),
            "defects": final_defects, "D0": D0}


def make_instances():
    # (seed, n_traps, n_generic, Lc, h_lo, h_hi, r_lo, r_hi, coeff_scale, name)
    specs = [
        (96901, 2, 4, 2.0, 0.5, 1.4, 0.30, 0.55, 1.0, "trap1"),
        (96902, 2, 5, 2.6, 0.5, 1.4, 0.30, 0.55, 1.2, "trap2"),
        (96903, 3, 3, 3.2, 0.5, 1.3, 0.25, 0.50, 0.8, "trap3"),
        (96904, 2, 6, 2.2, 0.5, 1.5, 0.30, 0.55, 1.4, "trap4"),
        (96911, 0, 7, 2.4, 0.6, 1.9, 0.30, 0.75, 1.0, "moderate1"),
        (96912, 0, 6, 3.0, 0.6, 1.8, 0.30, 0.70, 1.2, "moderate2"),
        (96913, 1, 5, 2.8, 0.6, 1.7, 0.30, 0.65, 1.0, "moderate3"),
        (96921, 0, 3, 1.6, 0.7, 1.6, 0.45, 0.90, 0.6, "gentle1"),
        (96922, 0, 3, 1.8, 0.7, 1.7, 0.45, 0.95, 0.6, "gentle2"),
        (96923, 0, 4, 1.7, 0.7, 1.6, 0.45, 0.90, 0.7, "gentle3"),
    ]
    out = []
    for (seed, nt, ng, Lc, h_lo, h_hi, r_lo, r_hi, cs, name) in specs:
        inst = _build_instance(seed, nt, ng, Lc, h_lo, h_hi, r_lo, r_hi, cs)
        inst["name"] = name
        out.append(inst)
    return out


# ------------------------------- interactive run -------------------------------------
def _public_payload(inst, phase, history, budget_left, rnd):
    return {"phase": phase,
            "domain": {"x_lo": X_LO, "x_hi": X_HI, "y_lo": Y_LO, "y_hi": Y_HI},
            "lipschitz_L": inst["Lc"], "m_cap": M_CAP, "n_inspect": N_INSPECT,
            "z_nom_coeffs": {"a0": inst["coeffs"][0], "a1": inst["coeffs"][1],
                              "a2": inst["coeffs"][2], "a3": inst["coeffs"][3],
                              "a4": inst["coeffs"][4], "a5": inst["coeffs"][5]},
            "budget_total": B_OWN, "budget_left": budget_left,
            "round": rnd, "rounds_total": ROUNDS, "max_probes_this_round": MAX_PER_ROUND,
            "history": [dict(x=h[0], y=h[1], z=h[2]) for h in history]}


def _run_instance(cand, inst):
    coeffs = inst["coeffs"]; Lc = inst["Lc"]; defects = inst["defects"]
    history = []
    for (px, py) in PILOT:
        history.append((px, py, true_z(coeffs, defects, Lc, px, py)))
    budget = B_OWN
    for rnd in range(ROUNDS):
        pub = _public_payload(inst, "probe", history, budget, rnd)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            return 0.0
        probes = ans.get("probes", [])
        if not isinstance(probes, list):
            return 0.0
        parsed = []
        for e in probes:
            if not isinstance(e, dict):
                return 0.0
            vals = []
            for key in ("x", "y"):
                v = e.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return 0.0
                fv = float(v)
                if fv != fv or fv in (float("inf"), float("-inf")):
                    return 0.0
                vals.append(fv)
            parsed.append(vals)
        allow = min(budget, MAX_PER_ROUND)
        taken = 0
        for x, y in parsed:
            if taken >= allow or budget <= 0:
                break
            budget -= 1
            taken += 1
            if x < X_LO - 1e-9 or x > X_HI + 1e-9 or y < Y_LO - 1e-9 or y > Y_HI + 1e-9:
                continue          # out of range: skip -- charged against budget, no reading
            history.append((x, y, true_z(coeffs, defects, Lc, x, y)))
    # ---- certify phase ----
    pub = _public_payload(inst, "certify", history, budget, ROUNDS)
    ans, st = isorun.run_candidate(cand, pub, timeout=20)
    if st != "OK" or not isinstance(ans, dict):
        return 0.0
    Dval = ans.get("bound")
    if isinstance(Dval, bool) or not isinstance(Dval, (int, float)):
        return 0.0
    Dval = float(Dval)
    if Dval != Dval or Dval in (float("inf"), float("-inf")) or Dval < -1e-9:
        return 0.0
    Dval = max(Dval, 0.0)

    # THE GRADER recomputes the tightest bound actually EARNED by this candidate's own,
    # grader-produced probe history -- never trusts an asserted number against a hidden
    # constant. Hardcoding a plausible-looking D without probing cannot pass this: with few
    # or no probes, D_earn stays close to (or equal to) D0, which every trap instance is
    # built to keep well above the true defect heights.
    hist_dev = [(px, py, abs(z - z_nom(coeffs, px, py))) for (px, py, z) in history]
    D_earn = sup_envelope(hist_dev, Lc)
    tol = SOUND_TOL_REL * max(1.0, D_earn) + SOUND_TOL_ABS
    if Dval < D_earn - tol:
        return 0.0                # unsound certificate: caught lying about what was earned

    D0 = inst["D0"] if inst["D0"] > 1e-9 else 1e-9
    quality = 1.0 - Dval / D0
    if quality < 0.0:
        quality = 0.0
    elif quality > 1.0:
        quality = 1.0
    r = OFFSET + SPAN * quality
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0
    return r


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        try:
            r = _run_instance(cand, inst)
        except Exception:
            r = 0.0
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
