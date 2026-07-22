Let me start from what actually hurts. I have a fixed pile of trajectories — states, actions, rewards, each tagged with the return it earned — collected by some policy I'll never get to query again, and most of these trajectories are mediocre. I want a return-maximizing policy out of this static log, and I'm not allowed to touch the environment. The painful, non-negotiable requirement is stitching: the trajectory I actually want may not exist in the data at all, but it can be assembled out of fragments — a good A→B piece from one logged run, a good B→C piece from another. Pure imitation can never beat the data, because it copies whole trajectories; I need something that recombines pieces across trajectories. That recombination is exactly what dynamic programming buys you in RL, so the obvious move is to reach for value learning.

So what does today's offline RL do, and where does it leave me. I'd estimate the optimal action-value Q*(s,a) = E_{tau~pi*}[R(tau)|s_0=s,a_0=a] by driving down the Bellman residual E_{(s,a,r,s')~D}[(r + gamma·max_{a'} Q_theta(s',a') - Q_theta(s,a))^2], and for continuous actions train a policy pi_phi to maximize E_{s~D,a~pi_phi}[Q(s,a)]. That max over a' is the engine — it's what propagates value backward and lets me stitch sub-optimal segments into something better than any single run. But the estimator is sitting on the deadly triad: function approximation, bootstrapping (the regression target contains Q_theta itself), and off-policy data, all at once, and that combination is famously prone to divergence and over-estimation. And offline it gets worse in a specific, mechanical way. The moment pi_phi drifts even slightly off the data distribution, it starts querying Q_theta at actions that never appear in D, and a neural net asked for a value at an unsupported input will happily extrapolate a large number — and there is no environment to roll out and discover the action was actually bad, so the error never gets corrected. The standard patch is to bolt on an in-distribution constraint, some divergence penalty D(d^{pi_phi} || d^{mu}) folded into the TD loss to keep the policy near the data. Now I have a constrained optimization whose two terms have to be re-balanced per task, with a thicket of heuristics, and the whole thing is brittle.

Let me stop and ask the heretical question: do I actually need a value function to stitch? The reason I reached for Q was "DP gives stitching." But stitching is really a statement about *trajectories* — it's the claim that the dataset contains the raw material for a good trajectory and I just need to find/assemble it. What if I model the trajectory distribution directly and condition on "high return," and let a powerful generative model do the recombination? If I can sample trajectories that (a) look like the data — so I never leave the support, by construction — and (b) are conditioned to be high-return, I'd get the stitching benefit without ever estimating a Q. The thing that makes this even thinkable now is that diffusion models have become absurdly good at modeling rich, multimodal distributions and, crucially, at *composing* — generating novel samples by recombining pieces of training data. That compositionality is exactly the trajectory-stitching primitive, dressed in generative-modeling clothes.

So let me reframe the whole problem. Instead of value estimation, I'll write it as conditional generative modeling: max_theta E_{tau~D}[log p_theta(x_0(tau) | y(tau))], where x_0(tau) is (a piece of) a trajectory and y(tau) is some information about it — for now, the return. Two things fall out immediately and they're worth savoring because they're precisely the pains I just listed. First, there's no Q, no Bellman backup, no bootstrapping — the deadly triad is simply absent because I'm doing maximum-likelihood density estimation, not fixed-point iteration. Second, there's no distribution-shift problem and therefore no need for an explicit in-distribution penalty: a generative model trained by maximum likelihood on D *is* concentrated on D's support by construction; sampling from it stays near the data automatically. The in-distribution constraint that offline RL has to engineer is free here. That's a strong enough reason to commit to this framing and see how far it goes.

Now the concrete diffusion setup. Following DDPM, I have a forward noising process q(x_{k+1}|x_k) = N(sqrt(alpha_k)x_k, (1-alpha_k)I) that turns a clean x_0 into Gaussian noise over K steps, and the closed form at any step is x_k = sqrt(bar_alpha_k)x_0 + sqrt(1-bar_alpha_k)eps with eps ~ N(0,I). The learned reverse process is p_theta(x_{k-1}|x_k) = N(mu_theta(x_k,k), Sigma_k). I won't optimize the full variational bound; the simplified DDPM surrogate L = E_{k,x_0,eps}||eps - eps_theta(x_k,k)||^2 is what works in practice, so eps_theta predicts the noise that was added. From the forward identity I can recover the clean sample estimate as x0_hat = (x_k - sqrt(1-bar_alpha_k) eps_theta(x_k,k)) / sqrt(bar_alpha_k), which is the same as sqrt(1/bar_alpha_k)x_k - sqrt(1/bar_alpha_k - 1)eps_theta(x_k,k). Let me make sure I've inverted the forward map correctly rather than just transcribing it — if I substitute the true eps into that inversion, I should get exactly x_0 back. Take ac = bar_alpha_k at some middle step (say k=50 on a cosine schedule, ac ≈ 0.36), pick an arbitrary x_0 and eps, form x_k = sqrt(ac)x_0 + sqrt(1-ac)eps, and feed it through: sqrt(1/ac)x_k - sqrt(1/ac-1)eps = sqrt(1/ac)(sqrt(ac)x_0 + sqrt(1-ac)eps) - sqrt(1/ac-1)eps = x_0 + (sqrt((1-ac)/ac) - sqrt(1/ac-1))eps, and sqrt((1-ac)/ac) = sqrt(1/ac-1) so the eps term vanishes and I recover x_0. Numerically the residual is ~4e-8, i.e. zero up to float error. Good — the parameterization is self-consistent. Then the analytic posterior q(x_{k-1}|x_k,x0_hat) has variance beta_k(1-bar_alpha_{k-1})/(1-bar_alpha_k) and mean [beta_k sqrt(bar_alpha_{k-1})/(1-bar_alpha_k)]x0_hat + [sqrt(alpha_k)(1-bar_alpha_{k-1})/(1-bar_alpha_k)]x_k. Predicting the reverse mean is therefore equivalent to predicting eps_theta and plugging it through these coefficients. One sanity check on the coefficients while I have them: at the very last reverse step (k=1), bar_alpha_{k-1} = bar_alpha_0 = 1, so the variance coefficient beta_k(1-1)/(1-bar_alpha_k) = 0 and the x_k-coefficient sqrt(alpha_k)(1-1)/(1-bar_alpha_k) = 0, leaving mean = [beta_k·1/(1-bar_alpha_k)]x0_hat with the bracket equal to 1 (plugging the schedule in, the k=1 coefficients come out exactly c1=1, c2=0, var=0). So the final step is deterministic and just emits x0_hat — which is what a denoiser at the cleanest level should do. The arithmetic behaves at the boundary; fine. The question is: what is x_0(tau)? What do I diffuse over?

The naive answer, copying the image setting where you diffuse over all pixels, is to diffuse over the whole trajectory — states *and* actions stacked together, x_k(tau) = (s_t, a_t, s_{t+1}, a_{t+1}, ...)_k, treated as a 2D array, channels × horizon, like a 1D image. That's what the trajectory-diffusion route did before. But let me think about whether the action channel is actually well-suited to a denoiser. States in these control problems are continuous and, along a trajectory, fairly smooth — positions and velocities evolve continuously. Actions are a different animal: they're often joint torques, which are high-frequency and jerky, sometimes near-discrete, and in general much less smooth than the state sequence. A denoiser has to regress the clean signal out of noise, and a high-frequency, non-smooth target is genuinely harder to fit than a smooth one — the model has to spend capacity chasing rapid fluctuations that carry little structure. So jointly diffusing states and actions makes me model the hardest, least-structured part of the trajectory through the same machinery that's doing the easy, smooth part, and the action channel drags it down. Wall. I want the recombination power of diffusion applied to the part of the trajectory it's actually good at.

What if I just don't diffuse the actions at all? Diffuse only over the state sequence, x_k(tau) = (s_t, s_{t+1}, ..., s_{t+H-1})_k — a clean, smooth, continuous signal, a 2D array with one column per timestep, state-dimension tall, horizon wide. But then a state sequence isn't a controller; I can't execute a sequence of states. I need actions. Here's the recovery: if I have two consecutive states s_t and s_{t+1}, the action that produced that transition is recoverable by an inverse dynamics model a_t = f_phi(s_t, s_{t+1}). And I can train f_phi on exactly the same offline data — every transition (s, a, s') in D is a supervised example — with a plain regression, no diffusion needed, no high-frequency signal to denoise. So the plan becomes: diffuse the smooth state plan, then read off each action with a small inverse-dynamics net. The hard, jerky action signal never enters the diffusion model; it's handled by a separate supervised regressor that's well-suited to it. That feels right. (I'd want to verify later that this actually helps — my prediction is that inverse dynamics should beat joint state-action diffusion specifically when actions are high-frequency, like torque control, and the gap should shrink when actions are smooth, like position control. That's the falsifiable consequence of the smoothness argument.)

Now the conditioning, which is the whole point — I need to make sampling prefer high-return trajectories, not just typical ones, because the data is dominated by mediocre behavior. The diffusion-score view gives me the handle: at a fixed noise level, eps_theta(x_k,k) is proportional to the negative score, eps_theta(x_k,k) = -c_k ∇_{x_k} log p(x_k) for a positive scale c_k. Guidance steers generation by changing that score, and in epsilon-parameterization the same change appears with the corresponding sign already baked into the noise formula. So how do I condition on high return?

The first thing I'd try is the classifier-guidance recipe: train a classifier p_phi(y | x_k) on noised trajectories and steer with its gradient, eps_hat = eps_theta(x_k,k) - omega·sqrt(1-bar_alpha_k)·∇_{x_k} log p(y | x_k). But stare at what that classifier *is* when y is the return. A predictor of "the return achieved by this noisy trajectory" is a value function. I'd be training a Q-function again — and with it I drag back in exactly the dynamic-programming machinery and the deadly-triad instability I just spent all this effort escaping. Worse, it's an offline value function, so it suffers the same out-of-distribution over-estimation: the learned return-predictor will erroneously assign high value to trajectories that wander off the data support, and its gradient will then *guide the sampler toward* those out-of-distribution, actually-bad regions, with no environment feedback to ever correct it. And there's a subtler structural ugliness: the diffusion model would be trained unconditionally, never asked to care about the return at training time, and only steered toward high return at test time by a separately-learned object — so the thing I sample is not the thing I modeled. The training objective and the sampling objective disagree. Three strikes. Classifier guidance reintroduces the value function I was trying to kill. Wall.

So I want to condition *without* a separate classifier — to get the classifier gradient out of objects the generative model already has, rather than from a network I have to train on noisy inputs. The gradient I wanted was ∇_x log p(y|x). By Bayes, log p(y|x) = log p(x|y) - log p(x) + log p(y), and the label prior log p(y) has no x-gradient, so ∇_x log p(y|x) = ∇_x log p(x|y) - ∇_x log p(x). That's the move: the classifier gradient is just the conditional score minus the unconditional score, and *both of those are scores of a generative model* — no separate classifier appears anywhere. Since epsilon is the negative score times a positive scale, the corresponding guided noise is the unconditional noise plus the same conditional-minus-unconditional epsilon difference. So if I train one network to be both a conditional denoiser eps_theta(x_k, y, k) and an unconditional one eps_theta(x_k, ∅, k), where ∅ is a null token standing in for "no condition," then the guided sample uses

  eps_hat = eps_theta(x_k(tau), ∅, k) + omega·( eps_theta(x_k(tau), y(tau), k) - eps_theta(x_k(tau), ∅, k) ).

omega = 1 is plain conditional sampling; omega > 1 extrapolates past the conditional, pushing harder toward the attribute y, away from the generic unconditional behavior. And how do I get one network to serve as both? Train it conditioned on y, but with probability p replace y with the null token ∅ — condition-dropout — so the same weights learn the conditional and the unconditional model simultaneously. This is gorgeous for my situation: there's no value function, so no deadly triad and no offline-Q over-estimation; and the network is trained to do conditional modeling at training time, so it's actually good at conditional generation at test time — training objective and sampling objective finally agree. (The folklore that classifier-free beats classifier guidance in practice now has a reason I can see: the classifier-guided route trains an unconditional model that never has to think about the condition until test time, whereas here the conditional model is shaped by the condition throughout training.)

Let me nail down what y is for the offline-RL case and how to make "high return" the target. y(tau) = R(tau), the return, and I'll normalize returns so R(tau) ∈ [0,1] across the dataset. Then sampling a maximal-return trajectory is just conditioning on R = 1 — the top of the normalized range — and letting omega > 1 extrapolate toward it. That's the whole "ask for the best behavior" mechanism: condition on R=1, crank omega.

But wait — there's a failure mode I should check before I trust this. The dataset is mostly sub-optimal. If I just train a conditional model and condition on R=1 at test time, is that enough? The conditional model is fit to the *whole* dataset, so its conditional distribution at any given return is still polluted by the surrounding sub-optimal behavior — the model's idea of "a trajectory" is a noisy, multimodal cloud, and even conditioned on high return I'll sample mediocre members of that cloud as often as good ones. Conditioning gets me the right region; it doesn't guarantee I land on the *high-likelihood* part of that region. I want to suppress the variance and pick out the trajectories the model is most confident in, which — after return-conditioning — are the genuinely high-return ones. The lever for that is the sampling temperature: in the reverse step I draw x_{k-1} ~ N(mu_{k-1}, Sigma_{k-1}), and shrinking that per-step covariance concentrates samples on the high-probability mode.

There's a notation trap here I want to get right before fixing a number. There are two ways to write "shrink the noise" and they are not the same scale. If I think of it as a *variance* multiplier alpha, the draw is x_{k-1} ~ N(mu_{k-1}, alpha·Sigma_{k-1}). But the way it's natural to implement — and the way the code below does it — is to multiply the *standard deviation*: x_{k-1} = mu_{k-1} + eta·Sigma_{k-1}^{1/2}z, which gives covariance eta^2·Sigma_{k-1}. So a std multiplier eta corresponds to a variance multiplier alpha = eta^2; the two knobs differ by a square, and if I conflate them I'll be off by a factor of two-ish in the effective temperature. Let me just tabulate the endpoints under the std convention to be sure the knob does what I expect. eta = 0 gives covariance 0 — the reverse chain becomes deterministic (mean only), collapses to a single mode, and loses the diversity that lets the model recombine fragments differently for different start states. eta = 1 gives covariance Sigma_{k-1} exactly — full DDPM, high variance, keeps drawing mediocre members of the conditional cloud. eta = 0.5 gives covariance 0.25·Sigma_{k-1} — a quarter of the DDPM variance, well inside (0,1), so it damps the noise substantially without zeroing it. That intermediate point is what I want: low enough to favor high-likelihood (post-conditioning: high-return) sequences, not so low that I kill the stitching diversity. So I'll use a standard-deviation multiplier of 0.5 (i.e. an effective variance scale of 0.25), and I'll keep the std-vs-variance distinction explicit in the code so the number means what I think it means.

The pieces now fit. Return-conditioning with classifier-free guidance points the sampler at high-return behavior; low-temperature sampling concentrates it on the high-likelihood trajectories the data actually supports; and the diffusion model's compositional sampling recombines fragments across trajectories. Put together, on a dataset of A→B and B→C segments, conditioning on "reach C with high return" and sampling at low temperature should put probability mass on an A→C trajectory that was never logged as a whole — the model has *implicitly* done the dynamic-programming stitch, with no Bellman backup, no value function, no max over actions anywhere. The stitching I needed came from generative composition plus guidance, not from DP.

Now the training loss, putting the pieces together. The diffusion piece is the DDPM denoising loss but conditional and with the dropout that creates the unconditional branch: for each trajectory sample noise eps ~ N(0,I) and a step k ~ U{1,...,K}, form the noised state plan x_k(tau), and with probability p replace the condition by ∅. The inverse-dynamics piece is a plain regression of the action from consecutive states. Jointly,

  L(theta, phi) = E_{k, tau~D, beta~Bern(p)} || eps - eps_theta( x_k(tau), (1-beta)·y(tau) + beta·∅, k ) ||^2
                + E_{(s,a,s')~D} || a - f_phi(s, s') ||^2.

The (1-beta)y + beta·∅ is just "with probability p, drop the label," written so it's one expression. I'll set p = 0.25 — and I should think about why not the smaller dropout (~0.1) that image classifier-free guidance typically uses. The conditioning signal here is a single scalar return, a much weaker, lower-dimensional handle than a rich text/class embedding, and I'm going to lean hard on the unconditional branch via the guidance extrapolation, so I want that unconditional model trained on a healthy fraction of the batches — a quarter — to be a reliable baseline for the eps(y) - eps(∅) difference. Too small a p and the unconditional model is undertrained and the guidance direction gets noisy.

Architecture for eps_theta. The state plan is a 2D array, state-dim tall and horizon wide — a 1D temporal "image" where the width is time. The natural denoiser is a temporal U-Net of 1D convolutions over the time axis: repeated residual blocks, each a couple of temporal convolutions with a moderate kernel (5) so each output position mixes a local window of timesteps — this is what imposes temporal smoothness and lets the model recombine local segments. I'll use group norm and a smooth nonlinearity (Mish) inside the blocks. The diffusion timestep k and the condition y each get embedded — a sinusoidal/positional embedding for k, an MLP for y, each into a vector of the same width (128) — and I combine them and inject into every residual block by adding to the activations of its first conv. The crucial detail for the null token: when y = ∅, I zero out the condition embedding. That's the cleanest realization of "no condition" — an all-zero contribution where the return embedding would go — and it's exactly what condition-dropout produces during training, so train and test are consistent. The inverse-dynamics f_phi is just a small MLP on the concatenation [s_t, s_{t+1}] → a_t.

Now planning, which is receding-horizon. I observe the current state s; I start the reverse chain from a scaled Gaussian prior, x_K = eta z with z ~ N(0,I) and eta = 0.5; I run the reverse denoising chain k = K...1 using the classifier-free guided eps_hat at each step and the same low-temperature reverse draw. But I have to make the plan *start where I actually am*, so at every reverse step I overwrite the first column of x_k with the observed state(s) — fix the first state(s) to the history, so the diffusion only inpaints the future consistent with the present. (Concretely I keep a short history queue of length C and clamp the leading columns of the plan to it, so the plan is conditioned on the recent past.) After the chain finishes I have a clean state plan x_0; I take the first two states (s_t, s_{t+1}) = x_0(tau), recover the action a_t = f_phi(s_t, s_{t+1}), execute it, observe the next state, and repeat. Standard receding-horizon control, but the planner is the conditional diffusion sampler.

The conditioning variable y doesn't have to be a return. The exact same machinery works if y encodes a *constraint* a trajectory satisfies (a one-hot indicator 1(tau ∈ C_i)) or a *skill* it demonstrates (a one-hot 1(tau ∈ B_i)). I train return- or constraint- or skill-conditioned models identically. And now the compositional structure of score models earns its keep. Suppose I've learned the conditional models for several attributes y^1, ..., y^n, each captured by the same eps_theta conditioned on the respective variable, and I keep the sign straight: ∇_{x_k} log q(x_k | y^i) = -c_k eps_theta(x_k, y^i, k) for the same positive c_k at this noise level. To sample a trajectory satisfying *all* of them jointly — a combination I may never have seen in training — I want q(x_k | {y^i}). Assume the attributes are conditionally independent given the trajectory. Then by Bayes,

  q(x_k | {y^i}_{i=1}^n) ∝ q(x_k) · prod_{i=1}^n  q(x_k | y^i) / q(x_k),

take logs,

  log q(x_k | {y^i}) ∝ log q(x_k) + sum_{i=1}^n ( log q(x_k | y^i) - log q(x_k) ),

and differentiate in x_k,

  ∇ log q(x_k | {y^i}) = ∇ log q(x_k) + sum_{i=1}^n ( ∇ log q(x_k | y^i) - ∇ log q(x_k) ).

Now translate each score back to a denoiser output. Every term in that sum is a score ∇log q(·) and each equals -c_k times the corresponding eps_theta at this same noise level, with the *same* positive c_k. Let me actually push the substitution through rather than wave at it. Write the additive score relation as s_hat = s_∅ + sum_i (s_{y^i} - s_∅), with s_∅ = -c_k·eps_theta(x_k,∅,k) and s_{y^i} = -c_k·eps_theta(x_k,y^i,k). Factor: s_hat = -c_k·[ eps_theta(∅) + sum_i (eps_theta(y^i) - eps_theta(∅)) ]. The thing I want to sample with is the eps that *is* this score, i.e. eps_hat = s_hat / (-c_k), and dividing by -c_k cancels it from every bracketed term at once, leaving

  eps_theta(x_k, {y^i}, k) = eps_theta(x_k, ∅, k) + sum_{i=1}^n ( eps_theta(x_k, y^i, k) - eps_theta(x_k, ∅, k) ).

I checked this numerically with a random c_k and random eps vectors — building s_hat from the scores and dividing back by -c_k reproduces the direct eps formula to ~1e-6 (float error). The cancellation isn't an approximation; it's exact, and it's exact *because* the same c_k multiplies every term (it would not cancel if different conditions sat at different noise levels — but they don't, they all share step k). Folding in the same guidance scale omega as before, the perturbed noise I actually sample with is

  eps_hat = eps_theta(x_k, ∅, k) + omega · sum_{i=1}^n ( eps_theta(x_k, y^i, k) - eps_theta(x_k, ∅, k) ).

Before trusting the n-term formula, I should make sure it doesn't contradict the single-condition case I already derived — set n = 1 and the sum has one term, eps_hat = eps_theta(∅) + omega·(eps_theta(y^1) - eps_theta(∅)), which is character-for-character the classifier-free guidance expression from above. (Substituting n=1 numerically gives a difference of exactly 0.0, as it must — it's literally the same expression with the sum unrolled.) So the general formula degrades to the special case, which is the minimum I'd demand of it. For n > 1 it composes attributes additively in score space — combine two constraints, or sequence two skills the agent only ever saw separately — and that's the same additive-score primitive that does the implicit stitching, now applied across *attributes* instead of across *trajectory fragments*. (Strictly the derivation assumed conditional independence; in practice the composition just needs to be *feasible* — some trajectory satisfying all the y^i must exist — and if it's infeasible the sampler produces incoherent behavior, which is the expected failure. There's also a NOT composition: to avoid an attribute y^j, flip the sign of its score difference, eps_hat = eps(∅) + omega·( sum_{i≠j}(eps(y^i)-eps(∅)) - (eps(y^j)-eps(∅)) ). And note the naive alternative of just feeding the *summed* one-hot condition one-hot(i)+one-hot(j) into a single forward pass fails — the network never saw that input during training — which is why the composition has to live in score space, not in the condition input.)

I also need to keep the DDPM parameterization honest. The clean route is to predict the noise eps and regress against the sampled eps; since the first state of the plan is clamped to the observed history, it carries no denoising target, so I zero its loss weight. An equivalent route predicts x_0 directly and regresses against the clean states, but I will write the noise-prediction form because it exposes the x0_hat inversion and posterior coefficients cleanly. I will weight trajectory timesteps by a normalized discount, zero the conditioned first row, compute the reverse mean from x0_hat with the DDPM posterior coefficients, and implement low temperature with an explicit standard-deviation multiplier eta = 0.5.

So let me write the whole thing as the code I'd actually ship: states-only diffusion, inverse-dynamics action recovery, classifier-free guidance with condition dropout, exact DDPM posterior arithmetic, and a low-temperature receding-horizon sampler.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import Bernoulli


def cosine_beta_schedule(K, s=0.008):
    steps = K + 1
    x = torch.linspace(0, K, steps)
    ac = torch.cos(((x / K) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)
    return betas


def extract(a, t, shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(shape) - 1)))


def apply_conditioning(x, cond, action_dim=0):
    # clamp the leading state(s) of the plan to the observed history (inpainting)
    for tstep, val in cond.items():
        x[:, tstep, action_dim:] = val.clone()
    return x


class ResidualTemporalBlock(nn.Module):
    """Two 1D temporal convolutions (kernel 5) over the horizon axis, with the
    diffusion-step + condition embedding injected additively after the first conv."""
    def __init__(self, c_in, c_out, embed_dim, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv1d(c_in, c_out, kernel, padding=pad),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(c_out, c_out, kernel, padding=pad),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, c_out),
                                      nn.Unflatten(-1, (c_out, 1)))
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.time_mlp(emb)       # inject k+y embedding
        out = self.conv2(out)
        return out + self.res(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, k):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=k.device) * -f)
        e = k[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class TemporalUnet(nn.Module):
    """Canonical temporal U-Net denoiser; cond is carried in the signature because
    the diffusion wrapper clamps observed state columns with the same dictionary."""
    def __init__(self, horizon, transition_dim, dim=128, dim_mults=(1, 2, 4, 8),
                 returns_condition=True, condition_dropout=0.25):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),
                                      nn.Linear(dim, dim * 4), nn.Mish(),
                                      nn.Linear(dim * 4, dim))
        self.returns_condition = returns_condition
        if returns_condition:
            self.returns_mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(),
                                             nn.Linear(dim, dim * 4), nn.Mish(),
                                             nn.Linear(dim * 4, dim))
            self.mask_dist = Bernoulli(probs=1 - condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, (ci, co) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(ci, co, embed_dim),
                ResidualTemporalBlock(co, co, embed_dim),
                nn.Conv1d(co, co, 3, 2, 1) if not last else nn.Identity()]))
        mid = dims[-1]
        self.mid1 = ResidualTemporalBlock(mid, mid, embed_dim)
        self.mid2 = ResidualTemporalBlock(mid, mid, embed_dim)
        for i, (ci, co) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(co * 2, ci, embed_dim),
                ResidualTemporalBlock(ci, ci, embed_dim),
                nn.ConvTranspose1d(ci, ci, 4, 2, 1) if not last else nn.Identity()]))
        self.final = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2),
                                   nn.GroupNorm(8, dim), nn.Mish(),
                                   nn.Conv1d(dim, transition_dim, 1))

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        # x: (b, horizon, obs_dim) -> conv expects (b, obs_dim, horizon)
        x = einops.rearrange(x, 'b h c -> b c h')
        t = self.time_mlp(time)
        if self.returns_condition:
            assert returns is not None
            z = self.returns_mlp(returns)
            if use_dropout:                               # training: drop y -> ∅
                mask = self.mask_dist.sample((z.size(0), 1)).to(z.device)
                z = mask * z
            if force_dropout:                             # sampling: the ∅ branch
                z = 0 * z
            t = torch.cat([t, z], dim=-1)                 # ∅ contributes a zero block
        h = []
        for a, b, down in self.downs:
            x = b(a(x, t), t); h.append(x); x = down(x)
        x = self.mid2(self.mid1(x, t), t)
        for a, b, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = b(a(x, t), t); x = up(x)
        x = self.final(x)
        return einops.rearrange(x, 'b c h -> b h c')


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=100,
                 hidden_dim=512, returns_condition=True, condition_guidance_w=1.2,
                 noise_std=0.5, loss_discount=1.0, predict_epsilon=True):
        super().__init__()
        self.model, self.horizon = model, horizon
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.n_timesteps = int(n_timesteps)
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.noise_std = noise_std                         # repo low-temp std scale
        self.predict_epsilon = predict_epsilon
        self.inv_model = nn.Sequential(
            nn.Linear(2 * observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
        betas = cosine_beta_schedule(n_timesteps)
        ac = torch.cumprod(1 - betas, 0)
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_ac', ac.sqrt())
        self.register_buffer('sqrt_1m_ac', (1 - ac).sqrt())
        self.register_buffer('sqrt_recip_ac', (1 / ac).sqrt())
        self.register_buffer('sqrt_recipm1_ac', (1 / ac - 1).sqrt())
        self.register_buffer('post_mean_c1', betas * ac_prev.sqrt() / (1 - ac))
        self.register_buffer('post_mean_c2',
                             (1 - ac_prev) * (1 - betas).sqrt() / (1 - ac))
        post_var = betas * (1 - ac_prev) / (1 - ac)
        self.register_buffer('post_log_var', torch.log(post_var.clamp(min=1e-20)))
        discounts = loss_discount ** torch.arange(horizon, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        lw = discounts[:, None] * torch.ones(horizon, observation_dim)
        lw[0] = 0.
        self.register_buffer('loss_weight', lw)

    def q_sample(self, x_start, k, noise):
        return extract(self.sqrt_ac, k, x_start.shape) * x_start \
            + extract(self.sqrt_1m_ac, k, x_start.shape) * noise

    def predict_start_from_noise(self, x_k, k, noise):
        if self.predict_epsilon:
            return extract(self.sqrt_recip_ac, k, x_k.shape) * x_k \
                - extract(self.sqrt_recipm1_ac, k, x_k.shape) * noise
        return noise

    def q_posterior(self, x_start, x_k, k):
        mean = extract(self.post_mean_c1, k, x_k.shape) * x_start \
             + extract(self.post_mean_c2, k, x_k.shape) * x_k
        return mean, extract(self.post_log_var, k, x_k.shape)

    def p_losses(self, x_start, cond, k, returns=None):
        eps = torch.randn_like(x_start)
        x_k = apply_conditioning(self.q_sample(x_start, k, eps), cond, action_dim=0)
        pred = self.model(x_k, cond, k, returns)                # dropout y -> ∅ inside
        target = eps if self.predict_epsilon else x_start
        if not self.predict_epsilon:
            pred = apply_conditioning(pred, cond, action_dim=0)
        return (self.loss_weight * (pred - target) ** 2).mean()

    def loss(self, trajectory, cond, returns=None):
        obs = trajectory[:, :, self.action_dim:]                # diffuse over states only
        act = trajectory[:, :, :self.action_dim]
        b = obs.shape[0]
        k = torch.randint(0, self.n_timesteps, (b,), device=obs.device).long()
        diffuse_loss = self.p_losses(obs, cond, k, returns)
        s, s_next = obs[:, :-1], obs[:, 1:]
        a = act[:, :-1]
        pred_a = self.inv_model(
            torch.cat([s, s_next], -1).reshape(-1, 2 * self.observation_dim))
        inv_loss = F.mse_loss(pred_a, a.reshape(-1, self.action_dim))
        return 0.5 * (diffuse_loss + inv_loss)

    def p_mean_variance(self, x, cond, k, returns=None):
        if self.returns_condition:
            eps_cond = self.model(x, cond, k, returns, use_dropout=False)
            eps_uncond = self.model(x, cond, k, returns, force_dropout=True)
            eps = eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)
        else:
            eps = self.model(x, cond, k)
        x0 = self.predict_start_from_noise(x, k, eps)
        return self.q_posterior(x0, x, k)

    @torch.no_grad()
    def p_sample(self, x, cond, k, returns=None):
        mean, log_var = self.p_mean_variance(x, cond, k, returns)
        noise = self.noise_std * torch.randn_like(x)                  # low-temp std scale
        nonzero = (1 - (k == 0).float()).reshape(-1, *((1,) * (x.dim() - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, returns=None, horizon=None):
        b = len(cond[0])
        horizon = horizon or self.horizon
        x = self.noise_std * torch.randn((b, horizon, self.observation_dim),
                                         device=self.betas.device)
        x = apply_conditioning(x, cond, action_dim=0)
        for i in reversed(range(self.n_timesteps)):
            k = torch.full((b,), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, cond, k, returns)
            x = apply_conditioning(x, cond, action_dim=0)
        return x

    @torch.no_grad()
    def plan_action(self, obs, target_return):
        cond = {0: obs}
        x = self.p_sample_loop(cond, returns=target_return)
        s_t, s_t1 = x[:, 0], x[:, 1]
        return self.inv_model(torch.cat([s_t, s_t1], -1))            # a_t = f_phi(s,s')
```
