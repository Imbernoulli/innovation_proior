**Problem.** The confidence penalty fixed the *output* over-confidence but said nothing about the
*internal* conditioning of the network — and its smallest gain (+0.21) was on the deepest model,
ResNet-56. A deep conv net's behavior is dominated by what happens inside: forward activations and
backward gradients are vectors hit by a long product of weight matrices, so the *spectra* of those
matrices decide whether signal and gradient vanish, stay balanced, or explode. Act on the weight
spectrum, not (as DropBlock did) the weights' spatial shape.

**Key idea.** Keep each conv layer's singular values near one — the norm-preserving / dynamical-isometry
condition — by softly pushing the output-channel filters toward mutually orthonormal. For each conv
weight reshaped to `W = [out_channels, in·k·k]` (rows = filters), penalize the squared Frobenius norm of
the Gram residual from identity, summed over layers: `lambda · sum_W ||W W^T − I||_F^2`. The off-diagonals
of `W W^T − I` are pairwise filter correlations (drive to zero → decorrelated filters, no wasted capacity);
the diagonal is `||filter||^2 − 1` (drive to zero → unit-norm filters, no magnitude collapse). For wide
conv weights only `W W^T = I` (output-channel Gram) is feasible, which is also the one with this clean
interpretation.

**Why it should help / why a penalty not a constraint.** Orthogonal *init* gives a flat spectrum at step
zero but the data gradient drags `W` off the manifold immediately; a hard Stiefel constraint would hold
it but costs an SVD/QR every step and is awkward for rectangular conv weights. A soft penalty is a
standing force applied every step (no schedule — unlike rung one, it must act from step zero to oppose the
drift), fully differentiable, gradient `4(W W^T − I)W` — two matmuls, no factorization. It complements,
not duplicates, the optimizer's L2: L2 controls *scale* (shrinks all singular values), this controls
*shape* (flattens the feasible spectrum). This is the internal lever the output penalty lacked; expect the
gain to concentrate on the deepest model, ResNet-56.

**Hyperparameters.** `lambda = 1e-4` — small, conservative, the lightest touch that keeps the spectrum
from drifting; same order as the earlier rungs' coefficients. Applied to 4D conv weights only
(`'conv' in name and 'weight' in name and p.dim() == 4`), skipping biases, BN scales, and the linear
classifier. No schedule. Costs essentially nothing per step.

```python
# EDITABLE region of pytorch-vision/custom_reg.py (lines 246-273) — step 3: orthogonal regularization.
# torch / nn / F already imported at module scope.
def compute_regularization(model, inputs, outputs, targets, config):
    """Orthogonal regularization on convolutional weights.

    Penalizes deviation from orthogonality: ||W^T W - I||_F^2 for each
    4D conv weight reshaped to [out_channels, in*k*k]. Coefficient=1e-4.
    """
    reg = torch.tensor(0.0, device=outputs.device)
    for name, p in model.named_parameters():
        if 'conv' in name and 'weight' in name and p.dim() == 4:
            W = p.view(p.size(0), -1)  # [out, in*k*k]
            WtW = W @ W.t()
            I = torch.eye(W.size(0), device=W.device)
            reg = reg + ((WtW - I) ** 2).sum()
    return 1e-4 * reg
```
