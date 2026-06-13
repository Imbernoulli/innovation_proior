**Problem (from step 1).** The $x_0$ target trailed the field at every scale (FID 25.25 / 13.56 / 11.99)
because its difficulty swings by orders of magnitude across $t$ under the fixed flat MSE — trivial at
low noise, near-impossible at high noise. The fix must be a member of the equivalent target family whose
regression problem is *evenly conditioned across all $t$*.

**Key idea (velocity prediction).** Write $a=\sqrt{\bar\alpha_t}$, $b=\sqrt{1-\bar\alpha_t}$, so
$x_t=a\,x_0+b\,\epsilon$ is a point at angle $\phi_t$ on a circle with axes $x_0,\epsilon$ (since
$a^2+b^2=1$). Its velocity is
$v=\tfrac{dx_t}{d\phi_t}=a\,\epsilon-b\,x_0=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$.
Predict $v$. Recovery, by eliminating $\epsilon$ from the two equations:
$\hat x_0=a\,x_t-b\,\hat v=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,\hat v$ (the $\epsilon$ terms
cancel via $a^2+b^2=1$).

**Why it should beat $x_0$.** $v$ is an orthonormal *rotation* of $(x_0,\epsilon)$, so it has the **same
scale as $x_t$ at every $t$** — a balanced mixture whose blend angle rotates but whose norm is
stationary. It asks for $\epsilon$ at low noise (well-conditioned there) and for $-x_0$ at high noise
(but with bounded scale, unlike pure $x_0$-prediction). The recovery is the exact inverse and needs no
clamp on $a$, so it is numerically clean at high noise where the $\epsilon$ recovery is most fragile.

**Hyperparameters.** None introduced — the fill is the two functions over the schedule scalars. All
training settings come from the fixed substrate. The harness does *not* expose the moves $v$ is usually
paired with (SNR loss weighting, zero-terminal-SNR schedule) — flat MSE and the linear schedule are
frozen — so this is $v$-prediction stripped to its target-and-recovery essence.

**What to watch.** $v$ should improve on $x_0$ at every scale (the conditioning fix is geometric, not
capacity-dependent). Open question: whether it reaches the field's best or sits between $x_0$ and the
noise corner — at $32\times32$ with full 50-step DDIM, the $\epsilon$ parameterization's known strength
may already capture most of the benefit.

```python
def compute_training_target(x_0, noise, timesteps, schedule):
    # V-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * x_0
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from v-prediction:
    # v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0
    # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    # => x_0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
```
