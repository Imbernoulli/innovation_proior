# Problem-Generation Brief — Format B (isolated heuristic evaluation)

Format B trains "write a program that solves an instance well across a distribution"
(FunSearch / AlphaEvolve / OpenEvolve shape) — but with the candidate run in an **isolated
subprocess** so untrusted model output cannot reward-hack the evaluator. Self-validate with the
program-mode harness until PASS. Return the requested JSON only.

Hard rules: **deterministic scoring only** (seed every instance; never wall-time/GPU). **The candidate
is untrusted and isolated** — it runs in its own process and only ever sees the PUBLIC view of an
instance; the answer, held-out data, and any oracle state stay in the evaluator process.

## Why isolation (this is mandatory)
An in-process candidate can `sys._getframe().f_back` to read the evaluator's locals/closures/globals,
monkeypatch shared modules, or print a fake `Ratio:` then `sys.exit(0)`. All were demonstrated. So the
candidate MUST be a standalone stdin→stdout program and the evaluator MUST run it via `isorun`.

## The candidate contract (what a solver writes; also your 4 solutions)
A candidate is a standalone program: read ONE JSON "public instance" from stdin, write ONE JSON answer
to stdout.
```python
import sys, json
inst = json.load(sys.stdin)      # public inputs ONLY
# ...compute...
print(json.dumps(answer))        # the ONLY thing the evaluator reads
```
`solutions/{trivial,greedy,strong,invalid}.py` are such programs; first line `# TIER: <name>`.

## Files in `<probdir>`
```
statement.md   task, the PUBLIC instance JSON schema, the answer JSON schema, objective, scoring
evaluator.py   FROZEN scaffold + scorer.  CLI: `python3 evaluator.py <candidate.py>`
config.yaml    checker: evaluator.py  /  memory: 512m  /  n_instances: <k>  /  time: 60s  /  type: program
meta.json      {id,tier,format:"B",family,eval_form,theme,title,strategies:[...]}
solutions/{trivial,greedy,strong,invalid}.py
```

## evaluator.py contract (deterministic + isolated)
```python
import sys, json, isorun          # isorun is provided on PYTHONPATH by the harness
def make_instances():             # deterministic, seeded; return list of dicts
    ... return [ {"public": {...}, "hidden": {...}}, ... ]   # split public vs hidden
def baseline(inst): ...           # trivial-construction objective the evaluator computes ITSELF
def score(inst, answer): ...      # validate answer against inst["public"]+inst["hidden"]; return objective
def main():
    cand = sys.argv[1]; insts = make_instances(); vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)   # ISOLATED subprocess
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)      # strictly validate; infeasible/garbage -> not ok
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj,1e-12))   # minimization; F/B analog for maximization
        vec.append(r if (r==r and 0<=r<=1) else 0.0)
    ratio = sum(vec)/len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x,6) for x in vec]))
main()
```
Rules:
- **Send only `inst["public"]` to the candidate.** Never put the answer / held-out / oracle in the
  public view. `score()` uses the full instance in the PARENT — the candidate never sees it.
- Validate the answer strictly: type/shape/range/finiteness (reject `nan`/`inf`); any violation → 0.
- Normalize so a trivial candidate ≈ 0.1; **leave headroom** — `strong` must NOT hit ~1.0 on the
  hardest instances (target ≤ ~0.9). Include some harder / held-out instances for generalization.
- Deterministic: seed everything; the harness re-runs and requires identical `Ratio`+`Vector`.
- Print `Ratio:` and `Vector:` each on their OWN final line (harness takes the LAST of each and
  requires the evaluator to exit 0).

## Self-validate before returning
```
python3 <SYNTH>/harness/validate_pyproblem.py <probdir>
```
Gates: G1 import · G3 bounds · G3b vector integrity (∈[0,1], len==n_instances, ratio∈[min,max]) ·
G4 determinism (all solutions) · G5 feasibility (invalid→0) · **G5b** bad candidates (raise/None/garbage/
nan)→0 · **G5c ISOLATION** (a frame-walk/introspection adversary must NOT beat trivial — proves the
candidate can't reach evaluator internals) · G6 baseline(trivial∈[0.03,0.35]) · G7 discrimination ·
G8 divergence. Iterate ≤6 rounds to PASS.

## Return (JSON only)
```json
{"id":"<id>","verdict":"PASS|FAIL","title":"...","format":"B","family":"...",
 "metrics":{"trivial":0.1,"greedy":..,"strong":..,"divergence":..,"invalid":0.0},
 "rounds":<int>,"notes":"core idea + isolation confirmed (or why it failed)"}
```
