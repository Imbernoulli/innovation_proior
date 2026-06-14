**Problem (from step 2).** Z-loss improved `val_loss` to 2.2926 by holding the *average* log-partition
down, but `λ·(log Z)²` is a soft average force — its gradient `2λ·log Z·p_j` puts no hard ceiling on any
*single* logit, so a particularly excited coordinate at one position can still climb into the bfloat16
unfaithful regime while the batch-averaged level sits near zero. The residual is a per-value bound on the
logits feeding the exp, which neither the target attack (smoothing) nor the average-level attack (z-loss)
provides.

**Key idea.** Attack the logit *values* structurally. Pass logits through a transform `g` before the
softmax that is (1) bounded (so they cannot blow up the low-precision exp), (2) smooth with nonzero
gradient everywhere (so no coordinate goes dead, unlike a hard `clamp` whose derivative is 0 outside
`[-s,s]`), and (3) strictly monotone (so the argmax — the predicted token — is preserved). Those three
force an S-curve. The canonical symmetric form is `s·tanh(z/s)` (Gemma-2, `s=30`): identity slope at the
origin, smooth asymptotic cap at `±s`. The exact identity `tanh(u)=2σ(2u)-1` rewrites it in sigmoid
terms — `15·tanh(z/15) = 30·σ(z/7.5) - 15`, and the `-15` is invisible to softmax — which fuses cleanly
into one cap-plus-cross-entropy pass.

**Why it works.** `g(z) ∈ (0, A)` for every real `z` (a *per-value* bound, not an average one), so the
exp never sees a runaway; the gradient `(1/C)·z̃·(1 - z̃/A)` is a bell — full in the middle, tapering but
never identically zero — so every finite logit stays connected to the task gradient and there is no kink.
It is a *nonlinear* monotone map, not a temperature rescale, so the cross-entropy is the honest negative
log-likelihood of the model whose output head is "linear + soft cap" — part of the forward map, applied
at train and eval alike, no train/eval split needed.

**Same-named ≠ paper.** The task description cites the Gemma-2 *tanh* cap at 30, but the actual edit is
the modded-nanogpt *sigmoid* form with retuned, asymmetric constants `A, B, C = 23.0, 5.0, 7.5` (range
`(0,23)`, inflection at `z=-5`, max slope `23/30`). It is the same bounded-monotone family — strictly
increasing, bounded, smooth — but *not* pointwise identical to `15·tanh(z/15)` nor that curve plus a
softmax-invisible constant; the constants are a knob for this model size, vocabulary, and bfloat16 regime.

**Hyperparameters.** `A, B, C = 23.0, 5.0, 7.5`; `capped = A·sigmoid((logits.float() + B)/C)` (cast to
float32 so the saturating exp runs in high precision); plain `F.cross_entropy` on the capped logits with
`ignore_index = -1`. `CONFIG_OVERRIDES` left empty — only the loss changes.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 3: cross-entropy with sigmoid logit soft-cap
def compute_loss(logits, targets):
    """Cross-entropy with logit softcapping via sigmoid."""
    # Softcap: maps logits through A * sigmoid((logits + B) / C)
    # Prevents extreme logit magnitudes while preserving ranking
    # Constants from modded-nanogpt PR #199
    A, B, C = 23.0, 5.0, 7.5
    capped_logits = A * torch.sigmoid((logits.float() + B) / C)
    return F.cross_entropy(
        capped_logits.view(-1, capped_logits.size(-1)), targets.view(-1),
        ignore_index=-1
    )


# training-setup hook left at the default — only the loss changed:
CONFIG_OVERRIDES = {}
```
