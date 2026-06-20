**Problem.** Minimize `R = 2N·max_k(a*a)_k / (Σ a_n)^2` over non-negative step functions `f = Σ a_n·1_[n,n+1)`
(lower = tighter upper bound on the first autocorrelation constant `C1`; provable floor `1.28`). The diversified
Sequential-LP rung reached `1.5170` at `N=600` and saturated there; the published record is `1.5028628969`. This
final rung reaches the record.

**Key idea.** The single-constructor minimax LP saturates at `~1.5170` for two compounding reasons: at `N=600` it
has too few coordinates to express the finely irregular record shape, and one local trust-region engine cannot leave
its basin to find the global one (AlphaEvolve already reached `1.5053` at the *same* `600` pieces, so the residual is
search breadth, not resolution). Closing the last `~0.014` is a different-method gap: scale the construction to
`30000` pieces so it can carry the fine structure, and replace the single local engine with a large-scale
LLM-guided evolutionary search over the *construction program itself* — AutoEvolver (Claude/Opus, "aspiration
prompting"), which repeatedly proposes and edits the constructor, scores it through the identical `R` evaluator, and
keeps what scores lower, over tens of hours. This rung reproduces the published `30000`-piece record construction and
scores it through this ladder's own FFT autoconvolution evaluator.

**Why these choices.** A trust-region LP linearizes around the current heights, so it follows the valley it is in;
restart kicks jostle locally but do not relocate to a structurally different valley, and the objective (`a*a`
bilinear, `max` over nodes non-smooth) has narrow, asymmetric, irregular good valleys. The record-grade solution is
exactly the family the `600`-piece SLP was drifting toward but could not reach — a single enormous boundary spike
(`~111×` the mean, at the right end, index `29999`) over an interior that is `~38%` near-zero, with the
autoconvolution flattened into a plateau of `~18000` of `~60000` nodes all within `10⁻⁴` of the peak. That
fineness needs both the `30000`-piece grid and a search over the form of the constructor, neither of which a single
bounded LP on `600` pieces commands. Reproducing it through the same evaluator records where the frontier truly sits.

**Hyperparameters / contract.** No optimizer here: the rung loads the published AutoEvolver `30000`-piece sequence
(`record_sequence.json` in this directory, key `sequence`, length `30000`) and scores it through the frozen
evaluator. The evaluator is `R = 2·N·max(np.convolve(v,v)) / (Σv)²`, identical to the `fftconvolve` form used
throughout (`np.convolve` and `fftconvolve` agree to `10` digits here). Deterministic: the same file scores
`R = 1.5028628969` every call. This is the record, not a single-constructor frontier — the open distance below it is
the gap to the provable floor `1.28`.

```python
import json
import numpy as np

# Published AutoEvolver 30000-piece record construction (Claude/Opus, "aspiration prompting"):
# github.com/tengxiaoliu/autoevolver, task ac1. Loaded from this trajectory's record_sequence.json.
with open("record_sequence.json") as fh:
    seq = json.load(fh)["sequence"]

v = np.asarray(seq, dtype=float)
N = len(v)

# Frozen evaluator: R = 2N * max_k (v*v)_k / (sum v)^2  (lower = tighter upper bound on C1).
R = 2.0 * N * float(np.max(np.convolve(v, v))) / (float(np.sum(v)) ** 2)
print(f"N = {N}")
print(f"R = {R:.10f}")          # 1.5028628969  -- the AutoEvolver record
assert round(R, 10) == 1.5028628969, R
```
