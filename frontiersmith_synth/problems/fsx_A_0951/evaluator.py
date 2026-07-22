#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0951 -- "Waste-Fed Row: Cross-Feeding Biofilm Layout"
(family: crossfeeding-biofilm-layout; format B, quality-metric).

THEME.  A single limiting primary nutrient (metabolite type 0) diffuses in from the
two open edges of a 1-D strip of biofilm cells.  The operator has a small catalogue
of candidate microbial SPECIES, each defined by which metabolite type it consumes,
which metabolite type its waste is (-1 if it produces none), a max uptake rate
(vmax), a half-saturation constant (Km, Monod kinetics), a self-inhibition constant
(Ki: SMALL Ki means the species is strongly poisoned by a build-up of its OWN
waste product; huge Ki means negligible self-inhibition), and two yields
(yield_biomass, yield_byproduct: fraction of consumed mass converted to new
biomass / to waste respectively).  The candidate chooses which species (if any) to
inoculate at every cell of the strip.  A deterministic reaction-diffusion-growth
simulation is then run for T steps and TOTAL biomass is the score, MAXIMIZED.

MECHANISM COMPOSITION.
  - diffusion-consumption: metabolites diffuse (explicit finite-difference, no-flux
    strip boundary, Dirichlet re-supply of metabolite 0 at both edges) and are
    consumed locally via Monod kinetics.
  - metabolic-crossfeeding: a species' own waste product inhibits ITSELF (Ki term)
    but is a perfectly good substrate for a DIFFERENT species placed nearby.
  - spatial-arrangement: because byproduct diffusion is slow, relief only works if
    the waste-consuming species is placed close to the producer -- WHERE you put
    each species, not just WHICH species you pick, determines the score.

TRAP.  The species with the single highest raw uptake rate (vmax) looks like the
obvious pick -- fill the whole strip with it.  But that species is also the most
self-inhibited: with nobody nearby to consume its waste, the waste piles up locally
and throttles its own uptake (Ki term collapses toward 0).  A layout that
interleaves it with its waste-consuming partner relieves the inhibition and lets
BOTH species grow, beating any monoculture -- spatial metabolic division of labor,
not the single most efficient consumer, maximizes total yield.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "L": int, "T": int, "boundary_conc": float,
             "diffusion": [d0, d1],
             "species": [{"consumes":int,"produces":int,"vmax":float,"Km":float,
                          "Ki":float,"yield_biomass":float,"yield_byproduct":float}, ...]}
  stdout: ONE JSON object:
            {"assign": [g_0, ..., g_{L-1}]}
          g_i is -1 (empty cell) or a valid index into "species".

  A layout is VALID iff `assign` is a list of exactly L integers, each -1 or a
  valid species index.  Invalid output, wrong length, a crash, a timeout, or
  non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes, ITSELF:
    obj_base = total biomass from a fixed reference layout: species index 0
               inoculated in EVERY cell (a naive, instance-agnostic monoculture),
    obj_ceil = (2 * diffusion[0] * boundary_conc * T) * best_chain_yield,
               where best_chain_yield is the largest achievable biomass-per-unit-
               of-consumed-primary-nutrient over any simple metabolite hand-off
               chain among the given species (an idealized, transport-unlimited
               upper reference -- computed from the species table alone, no
               simulation), and the leading factor is a generous bound on how much
               primary-nutrient mass the two open edges could ever inject in T
               steps,
    obj_cand = total biomass from the candidate's simulated layout.
  and normalizes with an affine anchor (naive monoculture -> 0.1):
    r = clamp( 0.1 + 0.9 * (obj_cand - obj_base) / max(1e-9, obj_ceil - obj_base), 0, 1 )
  Because obj_ceil assumes unlimited transport (no diffusion bottleneck, no local
  competition for space), no achievable layout reaches it -- real solutions stay
  below 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  obj_base/obj_ceil
are computed by THIS parent process, so a frame-walking / introspecting candidate
learns nothing useful.

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

    def nxt(lo, hi):
        """uniform float in [lo, hi)."""
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        frac = (state >> 11) / float(1 << 53)
        return lo + frac * (hi - lo)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_species(seed, hard):
    """Five roles at fixed slots, numeric parameters randomized within ranges
    that preserve the qualitative structure (trap always present):
      0 TRIV          consumes 0, produces -1, weak, uninhibited generalist
      1 FAST          consumes 0, produces 1, HIGHEST vmax, strongly self-inhibited
      2 RELIEF        consumes 1, produces -1, the genuinely good waste-eating partner
      3 DECOY_SAFE    consumes 0, produces -1, moderate vmax, no self-inhibition
      4 DECOY_MISMATCH consumes 1, produces -1, a weak/mediocre waste-eating partner
    `hard`=True widens Ki (harsher trap) and narrows RELIEF's edge over
    DECOY_MISMATCH (harder to tell the partners apart from the numbers alone).
    """
    ni = _rng(seed)
    triv = {"consumes": 0, "produces": -1, "vmax": round(ni(0.10, 0.22), 4),
            "Km": round(ni(1.6, 2.4), 4), "Ki": 1.0e9,
            "yield_biomass": round(ni(0.24, 0.34), 4), "yield_byproduct": 0.0}
    fast_vmax = round(ni(1.15, 1.55), 4)
    ki_lo, ki_hi = (0.9, 1.4) if hard else (1.3, 2.0)
    fast = {"consumes": 0, "produces": 1, "vmax": fast_vmax,
            "Km": round(ni(0.8, 1.3), 4), "Ki": round(ni(ki_lo, ki_hi), 4),
            "yield_biomass": round(ni(0.38, 0.50), 4),
            "yield_byproduct": round(ni(0.45, 0.58), 4)}
    relief = {"consumes": 1, "produces": -1, "vmax": round(ni(0.65, 0.95), 4),
              "Km": round(ni(0.55, 0.85), 4), "Ki": 1.0e9,
              "yield_biomass": round(ni(0.60, 0.80), 4), "yield_byproduct": 0.0}
    decoy_safe = {"consumes": 0, "produces": -1, "vmax": round(ni(0.55, 0.80), 4),
                  "Km": round(ni(1.0, 1.5), 4), "Ki": 1.0e9,
                  "yield_biomass": round(ni(0.34, 0.44), 4), "yield_byproduct": 0.0}
    mismatch_gap = (0.35, 0.55) if hard else (0.15, 0.35)
    mismatch_vmax = round(max(0.10, relief["vmax"] - ni(*mismatch_gap)), 4)
    decoy_mismatch = {"consumes": 1, "produces": -1, "vmax": mismatch_vmax,
                       "Km": round(ni(1.2, 1.8), 4), "Ki": 1.0e9,
                       "yield_biomass": round(ni(0.28, 0.42), 4), "yield_byproduct": 0.0}
    species = [triv, fast, relief, decoy_safe, decoy_mismatch]
    if hard:
        # a second, decoy "fast" producer: high vmax but produces NOTHING usable
        # (dead-end waste type -1 handled by giving it produces=-1, i.e. it never
        # tempts the chain search) -- purely a distractor with vmax between TRIV
        # and FAST so "just don't pick the very fastest" is not a safe shortcut.
        rogue = {"consumes": 0, "produces": -1, "vmax": round(ni(0.85, 1.05), 4),
                 "Km": round(ni(0.9, 1.4), 4), "Ki": 1.0e9,
                 "yield_biomass": round(ni(0.30, 0.40), 4), "yield_byproduct": 0.0}
        species.append(rogue)
    return species


def _build_instances():
    """Deterministic instance family: (seed, L, T, boundary_conc, d0, d1, hard)."""
    specs = [
        (101, 18, 28, 5.0, 0.40, 0.15, False),
        (102, 20, 30, 4.5, 0.38, 0.16, False),
        (103, 22, 30, 5.5, 0.42, 0.14, True),
        (104, 16, 26, 4.0, 0.36, 0.18, False),
        (105, 24, 32, 5.0, 0.40, 0.13, True),
        (106, 20, 28, 6.0, 0.44, 0.17, False),
        (107, 21, 30, 4.8, 0.39, 0.15, True),
        # harder / larger held-out instances
        (208, 28, 36, 5.2, 0.41, 0.14, True),
        (209, 30, 34, 4.6, 0.37, 0.16, False),
        (210, 26, 38, 5.5, 0.40, 0.12, True),
    ]
    out = []
    for seed, L, T, bc, d0, d1, hard in specs:
        species = _build_species(seed, hard)
        out.append({"name": f"strip{seed}", "L": L, "T": T, "boundary_conc": bc,
                    "diffusion": [d0, d1], "species": species, "hard": hard})
    return out


# ----------------------------- simulation -----------------------------------
def _simulate(inst, assign):
    L = inst["L"]; T = inst["T"]; bc = inst["boundary_conc"]
    d = inst["diffusion"]; species = inst["species"]
    K = 2
    conc = [[0.0] * L for _ in range(K)]
    conc[0][0] = bc
    conc[0][L - 1] = bc
    biomass = [0.0] * L
    for _ in range(T):
        newconc = [row[:] for row in conc]
        for k in range(K):
            dk = d[k]
            ck = conc[k]
            nk = newconc[k]
            for i in range(L):
                left = ck[i - 1] if i > 0 else ck[i]
                right = ck[i + 1] if i < L - 1 else ck[i]
                nk[i] = ck[i] + dk * (left + right - 2 * ck[i])
        conc = newconc
        conc[0][0] = bc
        conc[0][L - 1] = bc
        for i in range(L):
            s = assign[i]
            if s is None or s < 0:
                continue
            sp = species[s]
            c = sp["consumes"]; p = sp["produces"]
            c_up = conc[c][i]
            denom = sp["Km"] + c_up
            monod = c_up / denom if denom > 0 else 0.0
            if p != -1:
                inhib = sp["Ki"] / (sp["Ki"] + conc[p][i])
            else:
                inhib = 1.0
            rate = sp["vmax"] * monod * inhib
            consumed = rate if rate < c_up else c_up
            if consumed < 0:
                consumed = 0.0
            conc[c][i] -= consumed
            biomass[i] += consumed * sp["yield_biomass"]
            if p != -1:
                conc[p][i] += consumed * sp["yield_byproduct"]
    return sum(biomass)


def _best_chain_yield(species):
    """Idealized (transport-unlimited) best biomass-per-unit-consumed-nutrient
    over any simple metabolite hand-off chain starting at metabolite 0."""
    n = len(species)
    best = [0.0]

    def dfs(cur_type, mass, total, visited):
        for j in range(n):
            if j in visited:
                continue
            sp = species[j]
            if sp["consumes"] != cur_type:
                continue
            biomass_out = mass * sp["yield_biomass"]
            leftover = mass * sp["yield_byproduct"]
            newtotal = total + biomass_out
            if newtotal > best[0]:
                best[0] = newtotal
            if sp["produces"] != -1 and leftover > 1e-9:
                dfs(sp["produces"], leftover, newtotal, visited | {j})

    dfs(0, 1.0, 0.0, frozenset())
    return best[0]


# ----------------------------- validation ----------------------------------
def _score(inst, answer):
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    L = inst["L"]
    S = len(inst["species"])
    if len(assign) != L:
        return None
    clean = []
    for g in assign:
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < -1 or g >= S:
            return None
        clean.append(g)
    return _simulate(inst, clean)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        L = inst["L"]
        obj_base = _simulate(inst, [0] * L)
        bcy = _best_chain_yield(inst["species"])
        mass_bound = 2.0 * inst["diffusion"][0] * inst["boundary_conc"] * inst["T"]
        obj_ceil = mass_bound * bcy
        denom = obj_ceil - obj_base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "L": L, "T": inst["T"],
                  "boundary_conc": inst["boundary_conc"],
                  "diffusion": list(inst["diffusion"]),
                  "species": [dict(s) for s in inst["species"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            obj_cand = _score(inst, ans)
        except Exception:
            obj_cand = None
        if obj_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj_cand - obj_base) / denom
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
