**Problem.** Poly-1 lifted the easy pool (MobileNetV2/FashionMNIST → 94.74) but left the two CIFAR-100 pairs flat-to-down (71.56, 74.06), suggesting that on a deep 100-class net, pushing *harder* on confidence is the wrong direction — the ceiling is overconfident memorization. Focal and Poly-1 both kept the one-hot target and only rescaled the per-example push; neither touches the target itself.

**Key idea.** Cross-entropy against a one-hot target is minimized only at an *infinite* true-vs-rest logit gap, which drives the network to memorize the training labels overconfidently. Move the target off infinity: bleed mass `ε` onto a uniform prior, `q'(k) = (1-ε) δ_{k,y} + ε/C`. Now every class has a floor `ε/C > 0`, so an infinite gap becomes infinitely costly. Equivalently `H(q',p) = (1-ε) H(one_hot,p) + ε H(uniform,p)` — cross-entropy plus an anti-overconfidence pull toward uniform.

**Why.** This is the opposite force from Poly-1's pro-confidence `ε₁(1-P_t)`: the hard 100-class pairs need *less* confidence pressure, not more. Geometrically (logit = `-½‖x-w_k‖²`), the equal `ε/C` floor on every wrong class drives activations equidistant from all wrong templates at a bounded gap, tightening classes into compact, bounded-magnitude clusters — the structural antidote to overconfident memorization.

**Hyperparameters.** One knob, `ε = 0.1` (standard); applied via the decomposition, so no smoothed target vector is built. `ε = 0` recovers cross-entropy. Nothing per-example or epoch-dependent.

**What to watch.** Expect the two CIFAR-100 pairs to rise above their Poly-1 marks (71.56, 74.06), since bounding the logit runaway is aimed at their failure mode. FashionMNIST (10 classes, easy) may give a touch back from Poly-1's 94.74 under the mild anti-confidence penalty, but should stay high.

```python
# EDITABLE region of pytorch-vision/custom_loss.py — step 3: label smoothing (eps=0.1)
def compute_loss(logits, targets, config):
    """Label-smoothing cross-entropy: soften the one-hot target to
    (1 - eps)*one_hot + eps/C with eps = 0.1, preventing the true-vs-rest logit
    gap from running to infinity and curbing overconfidence."""
    return F.cross_entropy(logits, targets, label_smoothing=0.1)
```
