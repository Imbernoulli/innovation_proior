**Problem.** Rung one acted on the spatial *shape* of conv weights — two steps removed from the failure,
on a pipeline that already shrinks those weights (L2) and normalizes their activations (BN) — and left
the place where over-fitting is actually visible unregularized. When these nets over-fit, the symptom is
in the output: an over-confident classifier piles almost all softmax mass on one class (low entropy).
That is what DropBlock's conv-only penalty could not reach (most plainly on VGG's dense head). Act on the
output distribution directly.

**Key idea.** Penalize low-entropy (over-confident) outputs by *subtracting* the softmax entropy from the
loss: `L = CE − beta · H(p)`, with `H(p) = − sum_i p_i log p_i`. Minimizing `L` maximizes `H`, i.e.
penalizes confidence. This ports the entropy bonus that keeps an RL policy stochastic into supervised
classification. It is, up to a constant, the *reverse* KL to uniform, `D_KL(p‖u) = −H(p) + log K` — label
smoothing with the KL direction flipped, and (unlike label smoothing, which rewrites the target/loss this
edit surface freezes) implementable as a pure *additive* term.

**Why it should help.** The entropy gradient on the logits is `∂H/∂z_i = p_i(−log p_i − H)` — the
`p_i`-weighted deviation from the mean surprisal — so it pulls the dominant (confident) logit down and,
because of the `p_i` weight, leaves near-zero classes essentially untouched, flattening toward uniform
*without* destroying the wrong-class ratios ("dark knowledge"). The output is parameterization-invariant
(one strength travels across architectures, unlike a weight-scale coefficient), and the penalty
self-attenuates early — outputs are naturally high-entropy when the net is unsure, so no explicit warm-up
is needed (the opposite of rung one's too-late schedule); it bites only once outputs start to spike.

**Hyperparameters.** `beta = 0.1` — one task-dependent knob, conservative end of the range, held fixed
across all three pairs; no schedule, no threshold (the single-`beta` form is the simplest default). One
extra reduction over logits already computed; no auxiliary passes. A `1e-8` floor inside the log keeps it
finite when a peaked output drives some `p_i` toward underflow.

```python
# EDITABLE region of pytorch-vision/custom_reg.py (lines 246-273) — step 2: confidence penalty.
# torch / nn / F already imported at module scope.
def compute_regularization(model, inputs, outputs, targets, config):
    """Confidence penalty: penalize low-entropy predictions.

    Computes negative entropy of the softmax distribution and adds it
    as a penalty, encouraging the model to be less over-confident.
    Beta=0.1.
    """
    probs = F.softmax(outputs, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return -0.1 * entropy  # penalize confident (low-entropy) predictions
```
