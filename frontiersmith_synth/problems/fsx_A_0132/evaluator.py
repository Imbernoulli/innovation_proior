#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0132 -- "Selene Station: Streaming Cargo Consolidation"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A lunar habitat ("Selene Station") receives supply canisters that fall out
of the delivery pipeline ONE AT A TIME (a live stream -- there is no look-ahead at
what will arrive next).  Each canister has an integer stowage cost s (mass+volume
units).  The station stows canisters into pressurized storage modules, each with a
fixed usable budget C.  A module can hold any set of canisters whose costs sum to
<= C.  Powering and pressurizing a module is expensive, so the station wants to
consolidate the whole stream into as FEW modules as possible.

This is ONLINE 1-D bin packing (FunSearch's bin-packing testbed, dispatch-rule
flavor of ALE-Bench): canisters = items, module budget = bin capacity, modules
powered = bins used, MINIMIZE.  Crucially the placement is ONLINE and irrevocable:
a FIXED streaming simulator (this evaluator) processes canisters in arrival order
and, for each arrival, stows it in the currently-open module chosen by a PRIORITY
rule; if no open module has room, it powers a fresh module.  A canister is never
moved once stowed and the stream is never reordered.

WHAT THE MODEL SUPPLIES.  The model does NOT emit a full assignment (it cannot --
it never sees the future stream in the simulator).  It supplies the *priority rule*
as a weight vector `w` over a FIXED feature basis.  For the arriving canister of
cost s and each already-open module with remaining budget `res >= s`, the simulator
computes a priority score
        score(module) = w . phi(res, s, C)
    phi[0] = 1.0                         # bias / constant
    phi[1] = res / C                     # how empty the module is BEFORE stowing
    phi[2] = (res - s) / C               # leftover budget fraction AFTER stowing
    phi[3] = ((res - s) / C) ** 2        # squared leftover (nonlinear)
and stows the canister into the HIGHEST-score module (ties -> lowest module index).
If NO open module fits, a fresh module is powered.  Classic rules are special
cases: best-fit = w=[0,0,-1,0] (minimize leftover), worst-fit = w=[0,0,1,0]
(maximize leftover), first-fit = w=[1,0,0,0] (all equal -> lowest index wins).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "n": N (int),
             "items": [s_0, ..., s_{N-1}]}   # integer costs, 1 <= s_i <= C
          (the item list is provided for context/analysis; the simulator, not the
           candidate, decides placement online using the returned weights.)
  stdout: ONE JSON object:
            {"weights": [w0, w1, w2, w3]}    # exactly 4 finite real numbers
  Anything else (wrong type, wrong length, a non-finite weight, a crash, a
  timeout, non-JSON) -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    q_lb   = L1 lower bound = ceil(sum(items) / C)                # unreachable ideal
    q_base = modules used by the WORST-FIT priority rule           # weak reference
    q_cand = modules used by the candidate's priority rule
  normalized with an affine anchor (worst-fit -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  Worst-fit scores ~0.1; reaching the (generally unreachable) L1 bound scores 1.0;
  doing worse than worst-fit scores < 0.1.  Because L1 is a LOOSE lower bound even
  the best online rule stays strictly below 1.0 on most instances -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance and returns numbers.
The streaming simulator and references (L1, worst-fit) are computed by THIS parent
process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
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
def _build_items(seed, n, C, dist):
    """Return N integer canister costs in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "uni":
            s = ni(1, C)
        elif dist == "medium":                       # costs near half a module
            s = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "bimodal":                      # many tiny + some large
            s = ni(1, max(1, C // 5)) if ni(0, 99) < 55 else ni((3 * C) // 5, (9 * C) // 10)
        elif dist == "heavy":                        # mostly large, hard to pair
            s = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        elif dist == "weibull":                      # deterministic Weibull(k=1.5) shape
            u = ni(1, 100000) / 100001.0
            x = (-math.log(1.0 - u)) ** (1.0 / 1.5)
            s = 1 + int((x / 2.5) * (C - 1))
        else:
            s = ni(1, C)
        if s < 1:
            s = 1
        if s > C:
            s = C
        out.append(s)
    return out


def _build_instances():
    """Deterministic large-scale instance family. (seed, n, C, dist)."""
    specs = [
        (101, 90, 40, "uni"),
        (102, 120, 50, "medium"),
        (103, 150, 60, "bimodal"),
        (104, 100, 40, "heavy"),
        (105, 130, 50, "weibull"),
        (106, 110, 45, "medium"),
        (107, 160, 55, "uni"),
        (108, 140, 50, "bimodal"),
        # harder / larger held-out instances
        (109, 180, 60, "medium"),
        (110, 200, 50, "weibull"),
        (111, 170, 45, "heavy"),
        (112, 210, 60, "uni"),
        (113, 240, 55, "medium"),
        (114, 190, 50, "bimodal"),
    ]
    out = []
    for seed, n, C, dist in specs:
        items = _build_items(seed, n, C, dist)
        out.append({"name": f"selene{seed}", "capacity": C, "n": n,
                    "items": items, "dist": dist})
    return out


# ----------------------------- fixed streaming simulator -------------------
def _simulate(items, C, w):
    """Online single-pass simulator. For each arriving item, stow it into the
    highest-priority open module that fits (ties -> lowest index); power a fresh
    module only when nothing fits.  Returns the number of modules used."""
    w0, w1, w2, w3 = w
    residuals = []
    for s in items:
        best_i = -1
        best_score = None
        for i, res in enumerate(residuals):
            if res >= s:
                la = (res - s) / C
                score = w0 + w1 * (res / C) + w2 * la + w3 * (la * la)
                if best_score is None or score > best_score:
                    best_score = score
                    best_i = i
        if best_i < 0:
            residuals.append(C - s)
        else:
            residuals[best_i] -= s
    return len(residuals)


def _l1(items, C):
    return -(-sum(items) // C)          # ceil(sum / C)


_WORST_FIT = (0.0, 0.0, 1.0, 0.0)       # weak reference (0.1 anchor)


# ----------------------------- weight validation ---------------------------
def _weights(answer):
    """Return a 4-tuple of finite floats, or None if the answer is malformed."""
    if not isinstance(answer, dict):
        return None
    w = answer.get("weights")
    if not isinstance(w, list) or len(w) != 4:
        return None
    out = []
    for x in w:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")):
            return None
        out.append(xf)
    return tuple(out)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        items = inst["items"]
        q_lb = _l1(items, C)
        q_base = _simulate(items, C, _WORST_FIT)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C,
                  "n": inst["n"], "items": list(items)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            w = _weights(ans)
        except Exception:
            w = None
        if w is None:
            vec.append(0.0)
            continue
        try:
            q_cand = _simulate(items, C, w)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
