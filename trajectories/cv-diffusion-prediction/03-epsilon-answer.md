**Problem (from step 2).** $v$-prediction fixed the $x_0$ target's scale imbalance and improved at every
scale (FID 21.70 / 11.59 / 8.80), but did not reach the well-tuned-CIFAR-DDPM regime. Its residual sits
at high noise, where $v$ interpolates toward a high-variance $x_0$-like request. The question is whether
the *noise* corner of the target family does better at this low-resolution, full-50-step-DDIM setting.

**Key idea (epsilon prediction).** Predict the noise: `compute_training_target` returns $\epsilon$
unchanged (no schedule algebra). Recovery inverts $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$:
$\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, with the denominator clamped to a
small positive floor so it stays finite at the highest noise levels.

**Why it should win here.** (1) The fixed flat, uniform-$t$ MSE *is* the DDPM simplified objective —
with the target set to $\epsilon$, the frozen loop becomes the exact recipe this parameterization was
tuned for; the flat weight deliberately removes the over-emphasis the true variational weight places on
the low-value near-identity small-$t$ terms. (2) Conditioning tilt: at high noise $x_t\approx\epsilon$,
so predicting $\epsilon$ is nearly the identity — best-conditioned exactly at the coarse DDIM steps that
set global structure; at low noise the recovery divides the noise error by the small
$\sqrt{1-\bar\alpha_t}$, damping it where the target is hardest. This is the reverse tilt of $x_0$ and a
better FID match than $v$'s balanced-but-$x_0$-tilted high-noise behavior.

**Regime caveat.** $v$'s known advantage over $\epsilon$ appears at very few steps, high resolution, or
zero-terminal-SNR — all suppressed here ($32\times32$, 50 steps, frozen linear schedule with nonzero
terminal SNR). So the recovery's division by $\sqrt{\bar\alpha_t}$ stays finite and the clamp almost
never engages; the ordering reverses to $\epsilon < v < x_0$ in FID.

**Hyperparameters.** None introduced beyond the recovery clamp (`min=1e-8`). All training settings come
from the fixed substrate. This is $\epsilon$-prediction as the pure DDPM simplified objective — no
learned variance, importance-sampled $t$, or cosine schedule (the harness freezes those).

**What to watch.** $\epsilon$ should beat $v$ at every scale (small $<$ 21.70, medium $<$ 11.59, large
$<$ 8.80), large the cleanest. Failing to beat $v$ would falsify the regime argument.

```python
def compute_training_target(x_0, noise, timesteps, schedule):
    # Epsilon prediction: model learns to predict the added noise
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from epsilon prediction:
    # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    # => x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)
```
