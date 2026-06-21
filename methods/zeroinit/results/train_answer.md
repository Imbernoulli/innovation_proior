Flow-matching image and video generators sample by integrating an ordinary differential equation $dx/dt = v_t^\theta(x \mid y)$ from noise at $t=0$ to a clean sample at $t=1$, where the condition $y$ is a class label, a text prompt, or some other control signal. To make the condition actually bite at inference time, the standard tool is classifier-free guidance: query the same frozen network once with the condition and once with it dropped, obtaining $v_{\text{cond}}$ and $v_{\text{uncond}}$, and integrate the linear mix $v_{\text{guided}} = (1-w)\,v_{\text{uncond}} + w\,v_{\text{cond}}$ with a guidance scale $w>1$. The trouble is that this mix trusts the two learned velocities as if they were accurate enough to amplify. Writing $v_{\text{uncond}} = v_{\text{uncond}}^* + e_{\text{uncond}}$ and $v_{\text{cond}} = v_{\text{cond}}^* + e_{\text{cond}}$, the guided estimate carries the ideal guided part plus $(1-w)\,e_{\text{uncond}} + w\,e_{\text{cond}}$: the very scale that strengthens the conditional signal strengthens the conditional error in lockstep. When the model is underfitted, a large $w$ cannot tell semantic control apart from a wrong velocity. The other available options do not close this gap. Classifier guidance separates the baseline direction from a condition-improving direction, but needs an extra classifier trained on noisy inputs. Adaptive, characteristic, rectified, and CFG++-style variants target known high-scale pathologies and the geometry of the reverse step, yet leave the fixed velocity mix brittle when the flow field itself is inaccurate. Guidance-interval heuristics simply switch guidance off over a hand-chosen slice of the schedule, a coarse binary that can both discard useful guidance and admit harmful guidance. Predictor–corrector analyses explain why the CFG direction is suboptimal but offer no concrete drop-in replacement for the per-step mix.

What I want is a guidance rule that keeps the drop-in nature of CFG — no retraining, no extra network calls, the same two predictions per step — and behaves sensibly precisely when those two predictions are imperfect. I propose CFG-Zero*, which has two pieces: an optimized per-sample scale on the unconditional prediction, and a zero-init inert prefix at the start of the solver. The first piece needs a way to measure where the learned field is least trustworthy, and a Gaussian path supplies an exact yardstick. With $x_0 \sim \mathcal{N}(0,I)$, $x_1 \sim \mathcal{N}(\mu,I)$, and the linear path $x_t = (1-t)x_0 + t x_1$, the optimal flow-matching velocity is the conditional mean $v_t^*(x) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$. The pair $(x_t, x_1 - x_0)$ is jointly Gaussian with means $\mathbb{E}[x_t] = t\mu$ and $\mathbb{E}[x_1 - x_0] = \mu$, with $\operatorname{Var}(x_t) = ((1-t)^2 + t^2)\,I$, and with $\operatorname{Cov}(x_t, x_1 - x_0) = (1-t)(0-I) + t(I-0) = (2t-1)\,I$. The Gaussian conditional-mean identity then gives

$$v_t^*(x) = \mu + \frac{2t-1}{(1-t)^2 + t^2}\,(x - t\mu).$$

This is a diagnostic, not the deployed rule, but it lets me compare a learned velocity to the exact one across timesteps and checkpoints, and it shows the error is not one global training number — it varies with $t$ and is worst at the source end, where the sample is still close to noise yet guidance is already allowed to push hard.

The fix for the mix starts from the observation that CFG rigidly ties the unconditional coefficient to $(1-w)$, even when the unconditional prediction is too large, too small, or poorly aligned with the conditional one at this particular $(x,t)$. Borrowing the separation that classifier guidance suggests, I attach a scalar $s$ to the unconditional prediction, $v_s = (1-w)\,s\,v_{\text{uncond}} + w\,v_{\text{cond}}$, which reduces to ordinary CFG at $s=1$. I do not want $s$ to become another hand-tuned knob, so I choose it from the two vectors already in hand. Ideally I would minimize $\lVert v_s - v^* \rVert^2$, but $v^*$ is invisible on real data — that is the wall. Setting $\delta = w-1$ rewrites the same guided velocity as $v_s = v_{\text{cond}} + \delta\,(v_{\text{cond}} - s\,v_{\text{uncond}})$, so the unavailable loss is $\lVert (v_{\text{cond}} - v^*) + \delta\,(v_{\text{cond}} - s\,v_{\text{uncond}}) \rVert^2$. For any $\lambda > 0$, Young's inequality bounds $\lVert a + \delta b \rVert^2 \le (1+\lambda)\lVert a \rVert^2 + (1 + 1/\lambda)\,\delta^2 \lVert b \rVert^2$ with $a = v_{\text{cond}} - v^*$ and $b = v_{\text{cond}} - s\,v_{\text{uncond}}$. The first term still hides the unknown truth, but it does not depend on $s$; the only $s$-dependent part of the bound is a positive constant times $\lVert v_{\text{cond}} - s\,v_{\text{uncond}} \rVert^2$. So minimizing the bound in $s$ is just a projection using the two model predictions. Differentiating $g(s) = \lVert v_{\text{cond}} - s\,v_{\text{uncond}} \rVert^2$ gives $g'(s) = -2\,v_{\text{uncond}}^\top (v_{\text{cond}} - s\,v_{\text{uncond}})$, and setting it to zero yields

$$s^* = \frac{v_{\text{cond}}^\top v_{\text{uncond}}}{\lVert v_{\text{uncond}} \rVert^2},$$

with second derivative $2\lVert v_{\text{uncond}} \rVert^2 > 0$ confirming the least-squares minimizer whenever the unconditional vector is nonzero. Geometrically $s^* v_{\text{uncond}}$ is the orthogonal projection of $v_{\text{cond}}$ onto the line spanned by $v_{\text{uncond}}$, and the residual $v_{\text{cond}} - s^* v_{\text{uncond}}$ is the conditional part the unconditional prediction cannot explain. That residual — not the whole conditional vector — is what guidance should amplify, which is why the projection beats the fixed $(1-w)$ coefficient: it cancels exactly the component of the unconditional prediction that is redundant with the conditional one, per sample. The guided velocity can be written either as $(1-w)\,s^*\,v_{\text{uncond}} + w\,v_{\text{cond}}$ or, reading off the geometry, as $s^*\,v_{\text{uncond}} + w\,(v_{\text{cond}} - s^*\,v_{\text{uncond}})$: rescale the unconditional baseline, then push along the conditional residual. Numerically I floor the denominator with a small $\epsilon$, flatten all non-batch dimensions, take one dot product and one squared norm per sample, and broadcast the scalar back.

A better mix still does not rescue a step whose velocity is simply bad, and that is exactly the situation at the source end. The closed-form diagnostic sharpens the question at $t=0$: is the guided first-step velocity any closer to $v_0^*$ than the zero vector is? In the underfitted regime the diagnostic can satisfy $\lVert v_{\text{guided}}(t{=}0) - v_0^* \rVert_2^2 \ge \lVert 0 - v_0^* \rVert_2^2$, which reads as a decision rule: a guided move at the source end can be a worse velocity estimate than no move at all, so taking it injects the largest wrong direction precisely when the trajectory carries the least semantic information. Setting the velocity to zero leaves the ODE update inert and $x$ unchanged, avoiding that bad step. So the solver gets an inert prefix — zero velocity for the first $K$ steps, the optimized guided velocity afterward. $K$ stays small (the default is $K=1$) because the inequality is a claim about the unreliable source end, not the whole trajectory; once the field becomes informative, continuing to do nothing would only waste solver steps. In the implementation convention with the branch written as $i \le \texttt{zero\_steps}$, setting $\texttt{zero\_steps}=0$ zeros exactly the first step. The algebra closes cleanly: $v_{\text{uncond}}\,\alpha + w\,(v_{\text{cond}} - v_{\text{uncond}}\,\alpha)$ collects to $(1-w)\,\alpha\,v_{\text{uncond}} + w\,v_{\text{cond}}$, and at $\alpha = 1$ with the zero branch disabled this is standard CFG. The whole rule remains drop-in: the same two predictions per step, one dot product and one norm for the scale, and an inert prefix at the source end.

```python
import torch


def optimized_scale(positive_flat, negative_flat, eps=1e-8):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + eps
    return dot_product / squared_norm


@torch.no_grad()
def sample(pipeline, cond, uncond, guidance_scale, num_steps, zero_steps=0, use_zero_init=True):
    x = pipeline.initialize_sample()

    for i, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)

        batch_size = v_cond.shape[0]
        positive_flat = v_cond.view(batch_size, -1)
        negative_flat = v_uncond.view(batch_size, -1)
        alpha = optimized_scale(positive_flat, negative_flat)
        alpha = alpha.view(batch_size, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)

        if (i <= zero_steps) and use_zero_init:
            v = v_cond * 0.0
        else:
            v = v_uncond * alpha + guidance_scale * (v_cond - v_uncond * alpha)

        x = pipeline.ode_step(x, t, v)

    return x
```

For an $\epsilon$-prediction DDIM or CFG++ sampler, the same zero-init operation lives in the noise-prediction frame and is implemented by skipping the initial denoise-and-renoise updates with `if step < K: continue`:

```python
import torch


@torch.no_grad()
def sample_ddim_zeroinit(pipeline, prompt, cfg_guidance=7.5, K=2):
    uc, c = pipeline.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
    zt = pipeline.initialize_latent()

    for step, t in enumerate(pipeline.scheduler.timesteps):
        if step < K:
            continue

        at = pipeline.alpha(t)
        at_prev = pipeline.alpha(t - pipeline.skip)

        eps_uc, eps_c = pipeline.predict_noise(zt, t, uc, c)
        eps_pred = eps_uc + cfg_guidance * (eps_c - eps_uc)

        z0t = (zt - (1 - at).sqrt() * eps_pred) / at.sqrt()
        zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * eps_uc

    return pipeline.decode(zt)
```
