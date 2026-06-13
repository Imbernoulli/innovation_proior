**Problem.** Unconditional CIFAR-10 diffusion with everything frozen except the regression target. The
most literal target is the clean image itself: train the denoiser to output $x_0$, and at sampling time
use that output directly. It is the natural floor, and the one fill with zero risk of a
training/sampling consistency bug, since the recovery is the identity.

**Key idea (the floor target).** `compute_training_target` returns $x_0$ unchanged; `predict_x0`
returns the model output unchanged (the `prediction_type="sample"` DDIM scheduler already expects a
predicted $x_0$). No schedule algebra appears in either function — the $x_0$ parameterization is the one
point in the target family where both scalars $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$ drop out.

**Why it should be weakest.** The three targets are equal at the optimum but not under a finite budget.
The flat, unweighted MSE (the loop, fixed) weights every $t$ equally in pixel space, but the $x_0$
target's difficulty swings by orders of magnitude across $t$: at small $t$ it is nearly the identity
(trivial, tiny loss), at large $t$ it asks the network to hallucinate a full clean image from near-pure
noise (near-impossible, huge irreducible error). The gradient is dominated by the hopeless high-noise
term, capacity is mis-allocated, and the coarse-structure DDIM steps are taken on a shaky $\hat x_0$ —
exactly where $x_0$-prediction is least conditioned. None of the usual remedies (loss reweighting,
training-time clipping) are available here, so this is $x_0$-prediction in its rawest form.

**Hyperparameters.** None introduced — the entire fill is two one-line function bodies. All training
settings come from the fixed substrate (35k steps/scale, AdamW lr $2\times10^{-4}$, EMA 0.9995, linear
$\beta$ schedule, 50-step DDIM, FID at three channel scales).

**What to watch.** Real samples, but the worst FID of any sensible target, at all three scales, by a
margin that should not close as the network grows. That confirms the defect is the target's *uneven
conditioning across noise levels* — pointing the next step at a target that stays evenly hard at every
$t$.

```python
def compute_training_target(x_0, noise, timesteps, schedule):
    # X0-prediction: model directly predicts the clean image
    return x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Model output IS x_0, no conversion needed
    return model_output
```
