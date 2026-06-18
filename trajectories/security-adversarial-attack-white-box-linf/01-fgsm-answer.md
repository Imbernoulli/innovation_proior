**Problem.** White-box `L_inf` evasion at a tight budget `eps = 2/255`. The scaffold default returns
clean images and flips nothing (0 ASR). The only known generator — an L-BFGS-with-line-search inner
optimizer per image — is far too slow for 1000 samples across six scenarios. I need a closed-form
attack costing about one backward pass that beats the zero floor.

**Key idea.** Adversarial examples follow from *linearity in high dimension*, not curvature: a linear
readout's activation moves by `w^T η`, and under `||η||_inf ≤ eps` the worst case is `η = eps·sign(w)`,
giving an output swing `eps·||w||_1` that grows with dimension while each pixel stays invisibly small.
Lift this to the classifier's loss `J(θ,x,y)`: Taylor-expand in the input,
`J(x+η) ≈ J(x) + η^T ∇_x J`, and maximize over the `L_inf` box. By Hölder (`∞`/`1`), the maximizer is
the cube corner `η = eps·sign(∇_x J)` — the fast gradient sign step. One forward pass, one backward
pass, an elementwise sign.

**Why it works / why it stops here.** `sign(g)` is the steepest-ascent direction under the `L_inf`
geometry (an `L_2` budget would give `eps·g/||g||_2` instead); the sign, not the magnitude, is what
matters when each coordinate has its own `eps` allowance. The honest "imperceptible" notion is
per-coordinate (8-bit precision), which is exactly the max-norm the harness scores. But the step
maximizes the *linearization*, taken across the whole budget at once, so it underfits the curved loss
— most visibly on the wider VGG architecture. That underfit is the floor the iterative rungs improve.

**Hyperparameters.** Single step; full budget `eps = 2/255` (set by the harness). Cross-entropy loss.
`n_classes` / `device` unused. Perturbation projected to `[-eps,eps]` (a no-op for one step, but the
explicit budget projection shared with later rungs) then the image clamped to `[0,1]`.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    _ = (device, n_classes)
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x)[0]

    with torch.no_grad():
        x_adv = x + eps * grad.sign()
        delta = torch.clamp(x_adv - images, min=-eps, max=eps)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)

    return x_adv.detach()
```
