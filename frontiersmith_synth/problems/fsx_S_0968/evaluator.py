#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0968 -- "The Hollow-Span: Ballast Against the Canyon Choir"
(family: gust-detuning-mass-ballast; format B, quality-metric).

THEME.  A single rope bridge is strung across a haunted canyon.  Every free plank
of the deck (node i = 0..N-1) hangs between two fixed stone anchors, connected to
its neighbours by rope segments that act as springs (stiffness k_0..k_N).  Each
plank already carries a base mass (timber + old offerings).  The canyon "sings" --
a fixed, KNOWN spectrum of ghostly gust tones blows through it every night.  You
(the candidate) may bolt down integer sacks of ballast on the planks (total mass
<= a budget) before every crossing, and then choose how heavy a cargo to carry
across the resting stone at the deck's centre.  A crossing is safe only if the
resting stone's peak sway, under a whole night's worth of gust tones at several
storm intensities, stays under a fixed danger threshold; unsafe crossings lose
the cargo entirely (0 for that storm).

PHYSICS (small linear multi-DOF vibration model; run entirely by the evaluator).
  M = diag(base_mass + ballast), with `cargo` added onto M at the resting-stone
  node only (the cargo's own mass detunes the bridge WHILE it is being carried).
  K is the symmetric tridiagonal stiffness matrix built from k_0..k_N (fixed-
  fixed boundary conditions -- the two canyon walls never move).
  Mass-normalized modes (omega_k^2, phi_k) solve K phi = omega_k^2 M phi.  For a
  harmonic gust component (omega_g, spatial shape f, amplitude a) the standard
  modal steady-state amplitude at the resting-stone node uses the resonance
  overlap integral  p_k = phi_k . f  (how much the gust's shape excites mode k):
      contribution_k = |p_k * phi_k(node)| / sqrt((omega_k^2-omega_g^2)^2 + (2*zeta*omega_k*omega_g)^2)
  Multiple gust tones and modes are combined with the ABS (absolute-sum) modal
  combination rule -- a standard, deliberately conservative worst-case rule from
  gust/seismic design codes -- giving one scalar "peak" per (ballast, cargo).
  A storm night scales every gust tone's amplitude by a fixed multiplier s; the
  crossing is safe iff s*peak <= amp_threshold.  The instance's raw objective is
      obj = cargo * (fraction of the storm-scale list that is safe)
  averaged the same way the seed episode framing describes: mean scored cargo
  over a fixed, seeded set of "storm nights" per bridge.

MECHANISM COMPOSITION.
  - ballast-distribution: candidate freely allocates an integer budget across
    planks; total mass is fixed but WHERE it sits is the only lever.
  - forced-oscillation-response: score is a genuine steady-state modal response
    to a KNOWN multi-tone forcing spectrum, not a proxy like "static deflection".
  - mode-detuning: gust tones are fixed, known frequencies; ballast only ever
    acts by shifting the structure's own mode frequencies relative to them.

INNOVATION HOOK.  Amplitude is a resonance overlap integral (the p_k = phi_k.f
terms above), so the real design variable is the structure's SPECTRUM, not any
single node's raw sway.  Piling all the ballast onto whichever plank currently
swings the most ("the obvious move") pulls EVERY mode's frequency down together
-- including modes that were previously safely clear of the gust spectrum.  Each
instance is constructed (at generation time, deterministically) so that this
naive move drags a second, previously-safe mode straight into a second gust
tone, while a placement that SPREADS or even predominantly avoids the loudest
plank keeps every mode clear of both tones and can carry much more cargo.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object, the PUBLIC instance (see below).
  stdout: ONE JSON object {"ballast": [b_0..b_{N-1}], "cargo": c}
          b_i >= 0 integers, sum(b_i) <= ballast_budget; c a single integer,
          0 <= c <= cargo_cap.  Any other shape, non-integers, a negative value,
          over-budget ballast, an out-of-range cargo, a crash, a timeout, or
          non-JSON output makes that instance score 0.0.

PUBLIC INSTANCE SCHEMA.
    {"name": str, "n_nodes": N, "base_mass": [N floats], "stiffness": [N+1 floats],
     "ballast_budget": int, "cargo_cap": int, "cargo_node": int,
     "damping_zeta": float, "storm_scales": [floats], "amp_threshold": float,
     "gust_components": [{"omega": float, "shape": [N floats], "amp": float}, ...]}

SCORING (deterministic; no wall-time).  Per instance the evaluator also computes
two of its OWN references (never the candidate's):
  q_base = best-reachable obj using the "obvious" placement: ALL ballast on the
           single plank most excited by the dominant baseline mode (the "biggest
           swinger"), then the best cargo for that fixed placement.
  q_ub   = the best obj found by a small SHORTLIST of placements (the naive one
           above, a two-way split, a uniform spread, and a bounded single-unit
           greedy coordinate search on the peak itself) -- an informed but not
           exhaustive reference, deliberately leaving headroom above it.
and normalizes with an affine anchor (naive move -> 0.1, shortlist -> 0.92):
    r = clamp( 0.1 + 0.82 * (obj_cand - q_base) / max(1e-9, q_ub - q_base), 0, 1 )
Matching q_ub exactly caps at 0.92, never 1.0 -- q_ub is only a bounded
shortlist, not a proven optimum, so headroom remains even when matched.
Because q_ub is only a bounded shortlist (not a true optimum), a candidate that
does real spectral reasoning (build M,K from the public data, solve the modes
itself, and search placements against the ACTUAL gust spectrum) can match or
exceed it on many instances, while still leaving headroom above 1.0 on the
hardest ones.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  q_base/q_ub and
the physics engine live only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, hashlib
import numpy as np
import isorun

# ----------------------------- fixed constants ------------------------------
CAND_TIMEOUT = 10
STORM_SCALES = [0.55, 0.75, 0.95, 1.15, 1.4, 1.65]
SCALE = 0.82  # matching the q_ub shortlist reference tops out at 0.1+SCALE=0.92, never 1.0:
              # q_ub is only a bounded shortlist search, not a proven optimum, so genuine
              # headroom remains above it even when a candidate matches it exactly.

# (tag, n_nodes, ballast_budget, cargo_cap, zeta, seed_off)
INSTANCE_SPECS = [
    ("span01", 6, 22, 30, 0.035, 0),
    ("span02", 6, 26, 34, 0.030, 1),
    ("span03", 7, 26, 32, 0.035, 2),
    ("span04", 7, 30, 38, 0.032, 3),
    ("span05", 7, 24, 30, 0.040, 4),
    ("span06", 8, 32, 42, 0.030, 5),
    ("span07", 8, 34, 40, 0.035, 6),
    # harder / larger held-out spans
    ("span08", 8, 28, 36, 0.028, 7),
    ("span09", 9, 36, 48, 0.032, 8),
    ("span10", 9, 40, 50, 0.030, 9),
]


def _seed_for(off, n, budget):
    digest = hashlib.sha256(f"fsx_S_0968:{off}:{n}:{budget}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ----------------------------- physics core ----------------------------------
def _modes(mass, stiff):
    """mass: (n,) positive; stiff: (n+1,) positive segment stiffnesses.
    Return (omega2, phi): omega2[k] = omega_k^2 ascending, phi[:,k] mass-normalized
    mode shapes (phi_k^T diag(mass) phi_k = 1)."""
    n = len(mass)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = stiff[i] + stiff[i + 1]
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = -stiff[i + 1]
    linv = 1.0 / np.sqrt(mass)
    Ksym = (linv[:, None] * K) * linv[None, :]
    Ksym = 0.5 * (Ksym + Ksym.T)
    w, y = np.linalg.eigh(Ksym)
    w = np.clip(w, 1e-9, None)
    phi = linv[:, None] * y
    return w, phi


def _peak_amplitude(mass, stiff, gusts, zeta, node):
    w, phi = _modes(mass, stiff)
    omega_k = np.sqrt(w)
    total = 0.0
    for g in gusts:
        shape = np.asarray(g["shape"], dtype=float)
        omega_g = float(g["omega"]); amp = float(g["amp"])
        p = phi.T.dot(shape)
        denom = np.sqrt((w - omega_g * omega_g) ** 2 + (2.0 * zeta * omega_k * omega_g) ** 2)
        denom = np.maximum(denom, 1e-9)
        total += amp * float(np.sum(np.abs(p * phi[node, :]) / denom))
    return total


def _frac_success(peak, storms, threshold):
    cnt = sum(1 for s in storms if s * peak <= threshold + 1e-12)
    return cnt / len(storms)


def _swap_search(mass_base, stiff, node, gusts, zeta, n, start, cargo_ref, sweeps=5):
    """Exchange-move local search on the raw peak (worst-case cargo_ref loading):
    move one ballast unit at a time between planks while it strictly helps."""
    ballast = start.copy()
    mass = mass_base + ballast; mass = mass.copy(); mass[node] += cargo_ref
    cur = _peak_amplitude(mass, stiff, gusts, zeta, node)
    for _ in range(sweeps):
        improved = False
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                if ballast[j] <= 0:
                    break
                ballast[i] += 1; ballast[j] -= 1
                mass = mass_base + ballast; mass = mass.copy(); mass[node] += cargo_ref
                p = _peak_amplitude(mass, stiff, gusts, zeta, node)
                if p < cur - 1e-9:
                    cur = p; improved = True
                else:
                    ballast[i] -= 1; ballast[j] += 1
        if not improved:
            break
    return ballast


def _best_obj_for_ballast(mass_base, stiff, ballast, cap, node, gusts, zeta, storms, threshold):
    best = 0.0
    for cargo in range(0, cap + 1):
        mass = mass_base + ballast
        mass = mass.copy()
        mass[node] += cargo
        peak = _peak_amplitude(mass, stiff, gusts, zeta, node)
        obj = cargo * _frac_success(peak, storms, threshold)
        if obj > best:
            best = obj
    return best


# ----------------------------- instance construction --------------------------
def _build_one(tag, n, budget, cap, zeta, off):
    seed = _seed_for(off, n, budget)
    rs = np.random.RandomState(seed)
    hump = np.sin(np.pi * (np.arange(n) + 1) / (n + 1))
    dhump = np.abs(np.sin(2 * np.pi * (np.arange(n) + 1) / (n + 1)))
    node = n // 2
    best_attempt = None
    for attempt in range(100):
        mass = rs.randint(8, 21, size=n).astype(float)
        stiff = rs.uniform(20.0, 45.0, size=n + 1)
        w0, phi0 = _modes(mass, stiff)
        proj0 = np.abs(phi0.T.dot(hump))
        m1 = int(np.argmax(proj0[:min(3, n)]))
        omega1 = math.sqrt(w0[m1]) * rs.uniform(0.92, 1.03)
        i_star = int(np.argmax(np.abs(phi0[:, m1])))

        mass_c = mass.copy(); mass_c[i_star] += budget
        wc, phic = _modes(mass_c, stiff)
        best_k2, best_gap = None, -1.0
        for k2 in range(min(4, n)):
            if k2 == m1:
                continue
            gap = abs(math.sqrt(w0[k2]) - omega1)
            proj_c = abs(float(phic[:, k2].dot(dhump)))
            if proj_c > 0.05 and gap > best_gap:
                best_gap = gap; best_k2 = k2
        if best_k2 is None:
            continue
        k2 = best_k2
        omega2 = math.sqrt(wc[k2]) * rs.uniform(0.94, 1.04)
        if abs(math.sqrt(w0[k2]) - omega2) < 0.18 * omega2:
            continue  # not safely clear at baseline -> retry

        gusts = [{"omega": float(omega1), "shape": hump.tolist(), "amp": 1.0},
                 {"omega": float(omega2), "shape": dhump.tolist(), "amp": 0.85}]

        # Calibrate against the NAIVE (center-loaded) structure at half cargo, so the
        # "obvious" placement sits near the safety boundary -- not comfortably safe
        # at full cargo, forcing a real ballast-vs-cargo trade-off for it.
        ref_cargo = cap // 2
        mass_ref = mass.copy(); mass_ref[i_star] += budget; mass_ref[node] += ref_cargo
        peak_ref = _peak_amplitude(mass_ref, stiff, gusts, zeta, node)
        if peak_ref <= 1e-6:
            continue
        threshold = peak_ref * rs.uniform(0.90, 1.10)

        public = {"name": tag, "n_nodes": n, "base_mass": mass.tolist(),
                  "stiffness": stiff.tolist(), "ballast_budget": budget,
                  "cargo_cap": cap, "cargo_node": node, "damping_zeta": zeta,
                  "storm_scales": list(STORM_SCALES), "amp_threshold": threshold,
                  "gust_components": gusts}
        q_base, q_ub = _reference(public)
        gap = q_ub - q_base
        if best_attempt is None or gap > best_attempt[0]:
            best_attempt = (gap, public)
        if gap >= 0.13 * cap:
            return public
    return best_attempt[1]


def _reference(pub):
    """Evaluator's own (q_base, q_ub) references, computed from PUBLIC data only."""
    n = pub["n_nodes"]; budget = pub["ballast_budget"]; cap = pub["cargo_cap"]
    node = pub["cargo_node"]; zeta = pub["damping_zeta"]
    gusts = pub["gust_components"]; storms = pub["storm_scales"]; thr = pub["amp_threshold"]
    mass = np.array(pub["base_mass"], dtype=float)
    stiff = np.array(pub["stiffness"], dtype=float)

    hump = np.sin(np.pi * (np.arange(n) + 1) / (n + 1))
    w0, phi0 = _modes(mass, stiff)
    proj0 = np.abs(phi0.T.dot(hump))
    m1 = int(np.argmax(proj0[:min(3, n)]))
    i_star = int(np.argmax(np.abs(phi0[:, m1])))

    b_naive = np.zeros(n); b_naive[i_star] = budget
    q_base = _best_obj_for_ballast(mass, stiff, b_naive, cap, node, gusts, zeta, storms, thr)

    candidates = [b_naive]
    j_far = int(np.argmax(np.abs(np.arange(n) - i_star)))
    b_split = np.zeros(n); half = budget // 2
    b_split[i_star] = budget - half; b_split[j_far] += half
    candidates.append(b_split)

    q_u, r_ = divmod(budget, n)
    b_uni = np.full(n, float(q_u))
    for i in range(r_):
        b_uni[i] += 1
    candidates.append(b_uni)

    b_greedy = np.zeros(n)
    for _ in range(budget):
        best_i, best_peak = 0, None
        for i in range(n):
            b_greedy[i] += 1
            mass_try = mass + b_greedy; mass_try[node] += cap
            peak = _peak_amplitude(mass_try, stiff, gusts, zeta, node)
            b_greedy[i] -= 1
            if best_peak is None or peak < best_peak:
                best_peak = peak; best_i = i
        b_greedy[best_i] += 1
    candidates.append(b_greedy)

    q_ub = q_base
    for b in candidates:
        obj = _best_obj_for_ballast(mass, stiff, b, cap, node, gusts, zeta, storms, thr)
        if obj > q_ub:
            q_ub = obj
    return q_base, q_ub


def _reference_full(pub):
    """Scoring-time (q_base, q_ub): the cheap shortlist above PLUS a bounded
    swap-local-search with several restarts -- more search power than any single
    fixed-point construction, but still not exhaustive (deterministic, seeded
    from the instance's own public data). Only used at actual scoring time (not
    during generation's trap search) since it costs noticeably more to compute."""
    n = pub["n_nodes"]; budget = pub["ballast_budget"]; cap = pub["cargo_cap"]
    node = pub["cargo_node"]; zeta = pub["damping_zeta"]
    gusts = pub["gust_components"]; storms = pub["storm_scales"]; thr = pub["amp_threshold"]
    mass = np.array(pub["base_mass"], dtype=float)
    stiff = np.array(pub["stiffness"], dtype=float)

    hump = np.sin(np.pi * (np.arange(n) + 1) / (n + 1))
    w0, phi0 = _modes(mass, stiff)
    proj0 = np.abs(phi0.T.dot(hump))
    m1 = int(np.argmax(proj0[:min(3, n)]))
    i_star = int(np.argmax(np.abs(phi0[:, m1])))

    q_base, q_ub = _reference(pub)

    b_naive = np.zeros(n); b_naive[i_star] = budget
    j_far = int(np.argmax(np.abs(np.arange(n) - i_star)))
    b_split = np.zeros(n); half = budget // 2
    b_split[i_star] = budget - half; b_split[j_far] += half
    q_u, r_ = divmod(budget, n)
    b_uni = np.full(n, float(q_u))
    for i in range(r_):
        b_uni[i] += 1

    rseed = int(hashlib.sha256(f"ref:{pub['name']}:{n}:{budget}".encode()).hexdigest()[:8], 16)
    rs2 = np.random.RandomState(rseed)
    starts = [b_naive, b_split, b_uni, np.zeros(n)]
    for _ in range(4):
        w = rs2.dirichlet(np.ones(n))
        starts.append(np.floor(w * budget))
    for start in starts:
        b = _swap_search(mass, stiff, node, gusts, zeta, n, start, cap, sweeps=5)
        obj = _best_obj_for_ballast(mass, stiff, b, cap, node, gusts, zeta, storms, thr)
        if obj > q_ub:
            q_ub = obj
    return q_base, q_ub


def make_instances():
    out = []
    for tag, n, budget, cap, zeta, off in INSTANCE_SPECS:
        pub = _build_one(tag, n, budget, cap, zeta, off)
        out.append({"public": pub, "hidden": {}})
    return out


# ----------------------------- answer validation ------------------------------
def _validate_answer(pub, answer):
    if not isinstance(answer, dict):
        return None
    n = pub["n_nodes"]; budget = pub["ballast_budget"]; cap = pub["cargo_cap"]
    ballast = answer.get("ballast"); cargo = answer.get("cargo")
    if not isinstance(ballast, list) or len(ballast) != n:
        return None
    bl = []
    for x in ballast:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0:
            return None
        bl.append(x)
    if sum(bl) > budget:
        return None
    if isinstance(cargo, bool) or not isinstance(cargo, int):
        return None
    if cargo < 0 or cargo > cap:
        return None
    return np.array(bl, dtype=float), cargo


def score(inst, answer):
    pub = inst["public"]
    parsed = _validate_answer(pub, answer)
    if parsed is None:
        return False, 0.0
    ballast, cargo = parsed
    mass = np.array(pub["base_mass"], dtype=float) + ballast
    mass[pub["cargo_node"]] += cargo
    peak = _peak_amplitude(mass, np.array(pub["stiffness"], dtype=float),
                            pub["gust_components"], pub["damping_zeta"], pub["cargo_node"])
    frac = _frac_success(peak, pub["storm_scales"], pub["amp_threshold"])
    obj = cargo * frac
    if not math.isfinite(obj) or obj < 0:
        return False, 0.0
    return True, obj


def baseline(inst):
    return _reference_full(inst["public"])


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        pub = inst["public"]
        ans, st = isorun.run_candidate(cand, pub, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj_cand = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        q_base, q_ub = baseline(inst)
        denom = max(1e-9, q_ub - q_base)
        r = 0.1 + SCALE * (obj_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
