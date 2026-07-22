#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0751 -- "Consignment Sort: Hashing the Cartel's Manifest"
(family: lattice-aware-hash-policy; format B, quality-metric).

THEME.  A parcel-forwarding depot has been quietly laundering a cartel's shipments.
Every consignment is stamped with an integer TRACKING CODE.  To keep the manifest
looking innocuous to auditors, the cartel issues codes in tidy internal batches:
a single arithmetic run, a two-axis batch grid (carton-in-pallet x pallet-in-truck
strides), or a "lifted" subgroup of allowed check-digit residues repeated across
runs.  Whatever the batching scheme, EVERY pair of codes in an instance differs by
a multiple of some fixed BATCH STRIDE (the instance's hidden "lattice determinant").
The depot must route each parcel to one of B outbound TRUCKS by computing a
composed affine hash of its tracking code and reducing mod B.  Overloading a truck
(far more parcels than the fair share) blows the depot's schedule and draws the
very scrutiny the cartel is trying to avoid; the depot wants routing parameters
that keep every truck's load as small as possible.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "B": B (int),
             "codes": [x_0, ..., x_{N-1}]}    # N integer tracking codes
  stdout: ONE JSON object:
            {"a": int, "c": int, "M": int}
          describing the composed affine hash
             bucket(x) = ((a*x + c) mod M) mod B
          used to route every code to a truck in [0, B).

  VALID iff M is an int with 2 <= M <= MAX_M, a is an int with 1 <= a < M, and c is
  an int with 0 <= c < M.  Any invalid output, wrong type, a crash, a timeout, or
  non-JSON output makes that instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute:
    q_lb   = ceil(N / B)                       # pigeonhole lower bound (loose)
    q_base = max truck load of the RAW "code mod B" routing (weak baseline)
    q_cand = max truck load of the candidate's composed-hash routing
  and normalize with an affine anchor (weak baseline -> 0.1, pigeonhole ideal -> 0.85):
    r = clamp( 0.1 + 0.75 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching the raw baseline scores ~0.1; reaching the (generally
  unreachable) pigeonhole bound scores ~0.85 -- headroom is left above it on
  purpose.  Doing worse than the raw baseline scores below 0.1.

  Every instance's codes secretly live on an affine lattice x = r0 + (batch
  structure), so the codes' pairwise GCD equals a fixed BATCH STRIDE D.  A hash
  modulus M that shares a prime factor with D collapses the periodic structure
  into far fewer than M residues before the final "mod B" -- overloading a small
  number of trucks.  A modulus coprime to D (found by computing D = gcd of code
  differences, factoring it, and picking M with no shared prime factor) restores
  full spread.  Several instances plant D with large powers of 2 specifically to
  punish the common "power-of-two hash table size" idiom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(pigeonhole bound, raw-mod-B baseline) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MAX_M = 5_000_000_000  # sanity cap on the hash modulus a candidate may pick


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family ------------------------------
def _build_codes(seed, kind, n, params):
    """Return N integer tracking codes, secretly lying on a planted affine
    lattice (or, for 'uniform', on no lattice at all). Deterministic."""
    ni = _rng(seed)
    codes = []
    if kind == "ap":
        # single arithmetic run: x = r0 + D*t
        D = params["D"]
        r0 = ni(0, 1_000_000)
        for t in range(n):
            codes.append(r0 + D * t)
    elif kind == "grid":
        # two-axis batch grid: x = r0 + p*i + q*j  (carton-in-pallet x pallet-in-truck)
        p, q, I, J = params["p"], params["q"], params["I"], params["J"]
        r0 = ni(0, 1_000_000)
        cnt = 0
        for i in range(I):
            for j in range(J):
                if cnt >= n:
                    break
                codes.append(r0 + p * i + q * j)
                cnt += 1
            if cnt >= n:
                break
    elif kind == "coset":
        # lifted subgroup coset: x = r0 + M0*t + (random multiple of d), d | M0
        M0, d = params["M0"], params["d"]
        r0 = ni(0, 1_000_000)
        s = M0 // d
        for t in range(n):
            m = ni(0, s - 1) * d
            codes.append(r0 + M0 * t + m)
    elif kind == "uniform":
        # no planted structure at all (generalization / no-free-lunch cases)
        hi = params["hi"]
        for _ in range(n):
            codes.append(ni(0, hi))
    else:
        raise ValueError("bad kind")
    return codes


def _build_instances():
    """Deterministic instance family: (seed, kind, n, B, params)."""
    specs = [
        (90101, "ap",      3000, 100, {"D": 1 << 20}),
        (90102, "ap",      3000, 128, {"D": (1 << 16) * 3}),
        (90103, "grid",    3000, 100, {"p": (1 << 18) * 3, "q": (1 << 18) * 5, "I": 60, "J": 60}),
        (90104, "coset",   3000, 80,  {"M0": 1 << 20, "d": 1 << 15}),
        (90105, "ap",      3000, 90,  {"D": 3 ** 5 * 7}),
        (90106, "uniform", 6000, 60,  {"hi": 50_000_000}),
        (90107, "grid",    3000, 96,  {"p": 3 * 5 * 7 * 11, "q": 3 * 5 * 7 * 13, "I": 60, "J": 50}),
        # harder / held-out instances
        (90108, "ap",      4000, 250, {"D": 1 << 22}),
        (90109, "coset",   4000, 150, {"M0": (1 << 17) * 3, "d": 1 << 17}),
        (90110, "uniform", 5000, 50,  {"hi": 2_000_000}),
    ]
    out = []
    for seed, kind, n, B, params in specs:
        codes = _build_codes(seed, kind, n, params)
        out.append({"name": f"manifest{seed}", "n": n, "B": B, "codes": codes})
    return out


# ----------------------------- references -----------------------------------
def _pigeonhole_lb(n, B):
    return -(-n // B)   # ceil(n / B)


def _raw_mod_baseline(codes, B):
    """Weak reference: route parcel x to truck (x mod B) directly -- no hash
    parameters at all."""
    loads = [0] * B
    for x in codes:
        loads[x % B] += 1
    return max(loads)


# ----------------------------- validation ------------------------------------
def _routed_max_load(inst, answer):
    """Validate answer against the instance; return the candidate's max truck
    load, or None if infeasible/malformed."""
    if not isinstance(answer, dict):
        return None
    if "a" not in answer or "c" not in answer or "M" not in answer:
        return None
    a, c, M = answer["a"], answer["c"], answer["M"]
    for v in (a, c, M):
        if isinstance(v, bool) or not isinstance(v, int):
            return None
    if M < 2 or M > MAX_M:
        return None
    if not (1 <= a < M):
        return None
    if not (0 <= c < M):
        return None
    B = inst["B"]
    codes = inst["codes"]
    loads = [0] * B
    for x in codes:
        b = ((a * x + c) % M) % B
        loads[b] += 1
    return max(loads)


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        B = inst["B"]
        codes = inst["codes"]
        n = inst["n"]
        q_lb = _pigeonhole_lb(n, B)
        q_base = _raw_mod_baseline(codes, B)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n": n, "B": B, "codes": list(codes)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _routed_max_load(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.75 * (q_base - q_cand) / denom
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
