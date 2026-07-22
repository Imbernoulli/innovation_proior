#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0730 -- "Firefly Census: a Tiny Automaton Counts Blinks"
(family: streaming-count-automaton; format B, quality-metric).

THEME.  A meadow biologist wants to know how many times a target flash-pattern
occurred during a night's stream of firefly blink-codes, but her logger chip has
only a handful of memory cells: a finite-state machine with at most `s` internal
states.  The stream can run for hundreds of blinks, so an exact tally (which needs
one state per possible count, 0..L) busts the budget -- the chip must APPROXIMATE.

MECHANISM COMPOSITION (why this is not one textbook algorithm in a costume):
  1. automaton-memory-tradeoff -- the candidate submits a genuine Moore-machine
     description (states, a transition table indexed by (state, raw symbol), a
     start state, an output table) with <= s states; the EVALUATOR (not the
     candidate) simulates it, so position-independent finite-state limits are
     enforced structurally, not by trust.
  2. counting-approximation -- because L (stream length) and the true hit count
     can vastly exceed s, an exact incrementing counter is information-
     theoretically impossible; a good design must trade EXACT small counts for
     bounded RELATIVE error on large ones (Morris-style deterministic geometric
     bucketing), not just clamp/saturate.
  3. modular-hashing -- which raw symbols count as a "hit" is defined by a modular
     filter (symbol mod k in a residue set) baked into each public instance; the
     candidate must derive the hit/non-hit column classification for the WHOLE
     alphabet from that filter, and different instances use different (k, filter)
     pairs, so the classification cannot be hard-coded.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER INSTANCE).
  stdin : ONE JSON object, the PUBLIC instance:
            {"name": str, "m": int, "s": int, "k": int,
             "target_residues": [int, ...], "l_min": int, "l_max": int,
             "n_streams": int}
          m = alphabet size (raw symbols are integers 0..m-1); s = state budget;
          a raw symbol x counts as a HIT iff (x % k) is in target_residues.
  stdout: ONE JSON object describing a Moore machine:
            {"n_states": int, "start": int,
             "trans": [[int,...] * m] * n_states,   # trans[state][symbol] -> next state
             "out":   [number, ...] * n_states}      # out[state] = reported hit-count estimate

  VALID iff: 1 <= n_states <= s; 0 <= start < n_states; trans has exactly n_states
  rows each of exactly m entries, every entry an int in [0, n_states); out has
  exactly n_states finite numeric entries.  Anything else (wrong shape, non-int
  transitions, non-finite output, crash, timeout, non-JSON) -> that instance's
  score is 0.0.

SCORING (deterministic, no wall-time).  For each instance the evaluator itself
(never the candidate) generates `n_streams` seeded hidden blink streams of length
drawn from [l_min, l_max] over alphabet {0..m-1}, and knows the true hit count of
each.  The candidate's machine (obtained ONCE from isorun) is simulated by the
PARENT over every hidden stream: run the transition table over the raw symbols,
read the output table at the final state as the estimate.  Per-stream error:
    e = |estimate - true| / (true + 1)
Instance mean error `q_cand` is compared against a fixed weak reference
`q_base` = mean error of the "always guess 0" estimator (computed directly, no
candidate involved) and the ideal `q_lb = 0`:
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(q_base, 1e-9), 0, 1 )
Guessing 0 everywhere scores ~0.1; the reported Ratio is the mean of r across all
instances (harder / larger-alphabet instances are held out for generalization).

ISOLATION.  The candidate runs OS-sandboxed via isorun.run_candidate and only ever
sees the public instance; the hidden streams and true counts live only in this
parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:  Ratio: <mean r>   /   Vector: [r_1, ..., r_n]
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


# ----------------------------- instance family ------------------------------
def _specs():
    # (name, seed, m, s, k, target_residues, l_min, l_max, n_streams)
    return [
        ("meadow-easy-A",      1101, 16, 14, 4, [0],          20,  40,  40),
        ("meadow-easy-B",      1102, 18, 12, 3, [0],          15,  35,  40),
        ("steady-medium-A",    1201, 30, 14, 5, [0, 1],       60, 100,  40),
        ("steady-medium-B",    1202, 36, 12, 6, [0, 2],       70, 110,  40),
        ("steady-medium-C",    1203, 40, 16, 8, [0, 3, 5],    90, 130,  40),
        ("monsoon-dense-A",    1301, 18,  7, 3, [0, 1],      250, 350,  45),
        ("monsoon-dense-B",    1302, 22,  6, 4, [0, 1, 2],   280, 380,  45),
        ("monsoon-dense-C",    1303, 48,  9, 2, [0],         300, 420,  45),
        ("cloudburst-held-A",  1401, 60,  8, 5, [0, 2],      350, 500,  50),
        ("cloudburst-held-B",  1402, 54, 11, 9, [0, 1, 2, 3],320, 480,  50),
    ]


def _hit_rate(m, k, residues):
    rs = set(residues)
    hits = sum(1 for x in range(m) if (x % k) in rs)
    return hits / m


def _build_instances():
    out = []
    for name, seed, m, s, k, res, lmin, lmax, nstreams in _specs():
        rs = sorted(set(res))
        ni = _rng(seed)
        streams = []
        for _ in range(nstreams):
            L = ni(lmin, lmax)
            syms = [ni(0, m - 1) for _ in range(L)]
            true_cnt = sum(1 for x in syms if (x % k) in rs)
            streams.append((syms, true_cnt))
        public = {"name": name, "m": m, "s": s, "k": k,
                  "target_residues": rs, "l_min": lmin, "l_max": lmax,
                  "n_streams": nstreams}
        out.append({"public": public, "streams": streams})
    return out


# ----------------------------- validation -----------------------------------
def _parse_machine(answer, m, s):
    if not isinstance(answer, dict):
        return None
    n_states = answer.get("n_states")
    start = answer.get("start")
    trans = answer.get("trans")
    out = answer.get("out")
    if isinstance(n_states, bool) or not isinstance(n_states, int):
        return None
    if not (1 <= n_states <= s):
        return None
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if not (0 <= start < n_states):
        return None
    if not isinstance(trans, list) or len(trans) != n_states:
        return None
    for row in trans:
        if not isinstance(row, list) or len(row) != m:
            return None
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            if not (0 <= v < n_states):
                return None
    if not isinstance(out, list) or len(out) != n_states:
        return None
    for v in out:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            return None
    return {"n_states": n_states, "start": start, "trans": trans,
            "out": [float(v) for v in out]}


def _simulate(machine, syms):
    state = machine["start"]
    trans = machine["trans"]
    for x in syms:
        state = trans[state][x]
    return machine["out"][state]


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = inst["public"]
        m, s = public["m"], public["s"]
        streams = inst["streams"]

        # weak reference: "always guess 0" -- computed directly, no candidate
        base_errs = [tc / (tc + 1) for (_syms, tc) in streams]
        q_base = sum(base_errs) / len(base_errs)
        denom = q_base if q_base > 1e-9 else 1e-9

        ans, st = isorun.run_candidate(cand, dict(public), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            machine = _parse_machine(ans, m, s)
        except Exception:
            machine = None
        if machine is None:
            vec.append(0.0)
            continue

        try:
            errs = []
            for syms, tc in streams:
                est = _simulate(machine, syms)
                errs.append(abs(est - tc) / (tc + 1))
            q_cand = sum(errs) / len(errs)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
