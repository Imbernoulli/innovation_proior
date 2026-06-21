The whole game here lives inside a single expression. I am pretraining a GPT-2-medium next-token model with everything frozen except the function that turns final logits into a scalar loss, and the default there is plain cross-entropy. For one position with logits $z$ over the vocabulary and target index $y$, the per-token loss is $-\log \mathrm{softmax}(z)_y = -z_y + \log\sum_j e^{z_j}$. Sit with that and the problem is plain: to make it small I push $z_y$ up and the log-sum-exp down, and since the log-sum-exp is dominated by the largest logit, the loss decreases monotonically as the gap $z_y - \max_{j\neq y} z_j$ widens, reaching its infimum only as that gap $\to +\infty$. There is no finite minimizer. The objective is, quite literally, an instruction to make the logits diverge. On data that is even close to separable — which token-level LM data effectively is at the margin — gradient descent does exactly that: logit magnitudes drift upward across the run, the model grows more confident, and its predicted probabilities outrun its actual accuracy.

The reason this matters beyond calibration is the arithmetic underneath. Training runs in mixed precision — float32 master weights, but matmuls, activations, and the logits flow through bfloat16, which keeps 7 mantissa bits against float32's 23. That is about $2^{16}\approx 65{,}536\times$ the roundoff error within a binade, and because a fixed mantissa width spans each interval $[2^k, 2^{k+1})$, the absolute roundoff grows with the magnitude of the number. The logits go straight into an exponential, and an exponential turns a small additive perturbation $\delta$ of the input into a multiplicative factor $e^{\delta}$ on the unnormalized probability. So a roundoff error that is negligible on a small logit becomes a real distortion once the logit is large. Concretely: ten logits at $128$ and one at $128.5$, in bfloat16, after softmax subtracts the max; the $0.5$ gap is below bfloat16 resolution at that magnitude, so it rounds away, and the target probability lurches from $e^{0}/(e^{0}+10e^{-0.5})\approx 0.142$ to $e^{0}/(e^{0}+10e^{0})\approx 0.091$ — a 36% swing manufactured purely by roundoff, because the logits were large. The unbounded growth the loss encourages is therefore a direct route to noisy, unstable gradients through the softmax, worsening with vocabulary size. My loss is busy manufacturing the very large numbers that break the precision I run in.

The first instinct is to attack the target rather than the logits, and that is a principled move worth taking seriously. If the trouble is that the one-hot target demands probability 1 on $y$ — a demand for an infinite logit gap — then soften it: mass $1-\varepsilon$ on $y$, $\varepsilon/(V-1)$ spread over the rest, cross-entropy against that. The optimum then has $\mathrm{softmax}(z)_y = 1-\varepsilon$ and $\mathrm{softmax}(z)_{j\neq y}=\varepsilon/(V-1)$, so the optimal gap is the finite $\log\!\big((1-\varepsilon)(V-1)/\varepsilon\big)$, and the divergent incentive is gone. This is label smoothing, and it genuinely helps calibration. But it removes the incentive for one specific gap to run away while putting no ceiling on the absolute magnitude of the logits — a model can sit at $(1000, 990, 990, \dots)$ with exactly the smoothed-optimal gaps and be numerically enormous — and the roundoff problem is about magnitude, not gaps, so smoothing the target never touches it. It also optimizes a different distribution than the honest cross-entropy I am graded on. It is the wrong end of $-z_y+\mathrm{logsumexp}(z)$ for the failure I care about. The soft global penalty route — z-loss, adding $c_z\cdot(\mathrm{logsumexp}\,z)^2$ — does target the precision story, since $\mathrm{logsumexp}(z)\approx\max_j z_j$ for peaked logits, but it introduces a coefficient to tune, mixes a force into every step everywhere, only makes logits small on average without bounding any single one, and perturbs the reported loss value away from the clean modeling cross-entropy. And the crudest structural bound, the hard clamp $z\leftarrow\mathrm{clamp}(z,-s,s)$, caps magnitude but has derivative exactly $0$ outside $(-s,s)$: an over-large logit gets no learning signal through the loss-layer map, a dead coordinate behind a flat derivative, plus a non-differentiable kink at $\pm s$. The stabilization literature already shows the danger — tightening update clipping hard enough to stabilize a large model wrecked its quality, while a smooth z-loss stabilized the same model without that loss. Hard is the wrong shape.

So I want a transformation $g$ applied to the logits before the softmax that (1) bounds their range, (2) is smooth with a nonzero gradient everywhere so no coordinate goes dead, and (3) is strictly monotone so it never reorders the logits and the model's preferred token is preserved. Those three requirements nearly force the shape: monotone, bounded, smooth on the whole real line is a squashing S-curve, and the canonical one is the hyperbolic tangent — $\tanh:\mathbb{R}\to(-1,1)$, smooth, odd, strictly increasing, with derivative $1-\tanh^2$ that is $1$ at the origin and decays smoothly to (but never equals) $0$. I propose **logit soft-capping**: pass each logit through a scaled tanh,
$$\tilde z = s\,\tanh\!\left(\frac{z}{s}\right),$$
and then run plain cross-entropy on $\tilde z$. The two design choices that make this work are both in that formula. The outer $s$ replaces the useless $(-1,1)$ range with a usable cap of $\tilde z\in(-s,+s)$, so the exp inputs can never exceed $s$ in magnitude and the low-precision roundoff catastrophe is killed structurally. The inner $/s$ is the subtle, load-bearing part. The naive $s\,\tanh(z)$ has slope $s$ at the origin — it would amplify small logits by a factor of $s$, which is backwards; that form (slope $C$ at $0$) is a temperature/entropy knob, useful when you want to reshape the whole distribution, but the opposite of what I want. Dividing by $s$ inside fixes the slope: $\frac{d}{dz}\,s\tanh(z/s)=1-\tanh^2(z/s)$, which is exactly $1$ at $z=0$. For small $z$, $\tanh(z/s)\approx z/s$ so $g(z)\approx z$ — the map is the identity on healthy logits — and for large $z$, $\tanh(z/s)\to\pm 1$ so $g(z)\to\pm s$, a smooth asymptotic ceiling. The $/s$-inside is the whole difference between "an entropy knob that reshapes everything" and "identity-plus-soft-ceiling that only touches the runaways."

The gradient is the reason this beats the clamp, so it is worth pinning down. With $\tilde z = s\tanh(z/s)$,
$$\frac{d\tilde z}{dz} = 1 - \tanh^2\!\left(\frac{z}{s}\right) = 1 - \left(\frac{\tilde z}{s}\right)^2,$$
a smooth bell that is $1$ in the middle and tapers in the tails but is strictly positive everywhere on the real line — it only vanishes asymptotically, never on any interval. So an over-large finite logit still has a small but live gradient path; if cross-entropy wants it lower it can still move, with no dead zone and no kink. That is precisely "fully train the normal ones, softly saturate the outliers." For the cap value, the final output logits want $s$ large enough that the healthy range of gaps (a handful of nats) sits entirely in the identity regime, but small enough to bound the exp; a few tens does it, so the final-layer cap is around $30$ (attention pre-softmax scores, on a larger scale, would use a looser $50$, but the loss only sees the final logits). The exact number is not sacred; what is sacred is that $s$ sits above the normal logit range and below where bfloat16 exp starts to hurt.

Two refinements make this practical, both resting on an exact identity, not an approximation: $\tanh(u)=2\sigma(2u)-1$ with $\sigma$ the logistic sigmoid. So the tanh cap is an affine reparameterization of a sigmoid, and for the $s=15$ version, $15\tanh(z/15)=30\,\sigma(z/7.5)-15$. Two things follow. First, the additive $-15$ is invisible to the softmax, since $\mathrm{softmax}(z+c\mathbf 1)=\mathrm{softmax}(z)$ — adding the same constant to every logit cancels in the normalization — so as a softmax input $30\,\sigma(z/7.5)$ is cross-entropy-equivalent to the symmetric tanh cap, and the whole thing collapses to one sigmoid times a scale. Second, the sigmoid form is one cheap op that folds cleanly into a fused cap-plus-cross-entropy kernel, computing the capped value, the log-sum-exp, and the per-target term in one pass over a vocabulary of tens of thousands without ever materializing a separate capped-logit tensor. Generalizing to $\tilde z = A\,\sigma\!\big((z+B)/C\big)$, the parameters are exactly scale, shift, and steepness: $A$ sets the cap height (range $(0,A)$), $C$ the saturation scale, $B$ places the midpoint at $z=-B$. Its gradient stays the right shape,
$$\frac{d\tilde z}{dz} = \frac{A}{C}\,\sigma(u)\big(1-\sigma(u)\big) = \frac{1}{C}\,\tilde z\left(1-\sigma(u)\right),\qquad u=\frac{z+B}{C},$$
a bell peaked at the midpoint with height $A/(4C)$ and vanishing in both tails — everywhere positive, no dead derivative, no kink — and composed with the usual cross-entropy gradient $p_j-\mathbf 1[j=y]$ it gives the cap-aware logit gradient directly, which is what the fused backward needs. Once the cap is accepted as a bounded, smooth, strictly-monotone squash and the softmax-invisible output constant is understood, the constants become knobs for this regime; the retuned setting for short nanoGPT-scale pretraining is $A,B,C = 23.0,\,5.0,\,7.5$, i.e. $23\,\sigma\!\big((z+5)/7.5\big)$, strictly increasing into $(0,23)$ with maximum slope $23/30$. This is not pointwise identical to $15\tanh(z/15)$, nor that curve plus a softmax-invisible constant; it is a tuned member of the same bounded-monotone family with a shifted midpoint and smaller height.

One guardrail: this must be a faithful modeling loss, not a sneaky distortion. A temperature scaling $z\mapsto z/\tau$ is a linear squeeze that uniformly flattens every gap and can mechanically move the loss without the model improving. The soft cap is not that — it is a nonlinear monotone map whose local slope depends on where a logit sits and whose tails saturate, refusing to let any single logit run to infinity rather than uniformly rescaling gaps. Cross-entropy is then computed honestly on the capped logits: it is the true negative log-likelihood of the model whose output head is "linear layer followed by the soft cap." The cap is part of the model's forward map, so the loss is the genuine modeling loss of a better-behaved model, not a cosmetic rescaling of an unchanged one. I cast the logits to float32 before the sigmoid so the saturating exp runs in high precision, and keep $\texttt{ignore\_index}=-1$ for padded positions. That fills the one open slot, the body of $\texttt{compute\_loss}$:

```python
import torch
import torch.nn.functional as F


def compute_loss(logits, targets):
    """Cross-entropy with a smooth soft cap on the final logits.

    logits : (B, T, V); targets : (B, T) with ignore_index = -1.
    Each logit is mapped through a strictly-increasing bounded squash before the
    softmax, so logits cannot run away (bounding the low-precision exp) and the
    token ranking is preserved, while the gradient stays smooth and nonzero.
    """
    # Tuned sigmoid soft cap. Exact identity for the symmetric 15-cap:
    # 15*tanh(z/15) = 30*sigmoid(z/7.5) - 15, and the -15 shift is
    # invisible to softmax. A, B, C below are a retune, not a pointwise-
    # identical tanh curve.
    A, B, C = 23.0, 5.0, 7.5
    # cast to float32 before the sigmoid so the saturating exp runs in high precision
    capped_logits = A * torch.sigmoid((logits.float() + B) / C)
    return F.cross_entropy(
        capped_logits.view(-1, capped_logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )
```

And the equivalent symmetric tanh form, the version to reach for when capping attention scores as well as the final logits:

```python
import torch
import torch.nn.functional as F


def softcap(logits, cap):
    # logits <- cap * tanh(logits / cap): slope 1 at 0, smooth ceiling at +/-cap
    return cap * torch.tanh(logits / cap)


def compute_loss_tanh(logits, targets, final_cap=30.0):
    z = softcap(logits.float(), final_cap)
    return F.cross_entropy(z.view(-1, z.size(-1)), targets.view(-1), ignore_index=-1)
```
