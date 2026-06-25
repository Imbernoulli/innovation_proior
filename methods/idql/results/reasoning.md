Let me start from what actually hurts in offline RL, because everything I might build has to answer it. I have a fixed dataset of transitions `(s, a, r, s')` drawn by some behavior policy `mu`, and no way to interact with the environment again. The moment I try to learn a value function by bootstrapping — fit `Q(s,a)` toward `r + gamma * Q(s', a')` — I have to decide which `a'` to plug into the next-state value. If I take `a'` from the policy I am trying to learn, that policy will, by construction, want to deviate from `mu`, so `a'` slides off the data support, the function approximator returns a value for an action it has never seen, and because nothing pins those values down they come out too high far more often than too low. The backup then carries that phantom optimism backward through the chain, the policy chases it, and the whole thing diverges. So the real enemy is querying the value of out-of-distribution actions. Every offline method I know of is, underneath, a different way of refusing to trust those queries — constrain the policy to stay near `mu`, or penalize OOD `Q`-values, or avoid the OOD query outright.

The cleanest of those is the in-sample idea: never write down `max_{a'} Q(s', a')` at all, because that max is exactly the thing that reaches outside the data. Instead estimate "the value of the best in-support action" from dataset actions alone. The trick that makes this work is expectile regression. The `tau`-expectile of a random variable `X` is `argmin_m E[L2^tau(x - m)]` with `L2^tau(u) = |tau - 1(u<0)| u^2`: an asymmetric squared loss that, for `tau > 0.5`, punishes the case `x < m` less than the case `x > m`, so the minimizer is pulled above the mean, and as `tau -> 1` it climbs toward the supremum of `X`. Apply that conditionally to the distribution of `Q(s,a)` as `a` ranges over `mu(.|s)`: a high expectile of that distribution is, in effect, the value of the best action the data supports, recovered purely from in-sample actions, with the "max" done implicitly by the loss asymmetry rather than by an explicit `argmax`. I never have to name the maximizing action, so I never query an OOD one. And there's a clean reason to put this expectile on a *separate* value network `V(s)` rather than directly on the TD target: if I expectile-regressed `r + gamma Q(s', a')` itself, the asymmetry would also reward lucky *transitions* — a single `s'` that happened to be good — and confuse "this action is reliably good" with "this sample got lucky in the dynamics." Splitting it — `V` takes the action-expectile, then `Q` does an ordinary SARSA TD backup against `V(s')` — isolates the max over actions from the noise in the environment. So I'd train

  `L_V(psi) = E_{(s,a)~D}[ L2^tau(Q_targ(s,a) - V_psi(s)) ]`,
  `L_Q(theta) = E_{(s,a,s')~D}[ (r + gamma (1-done) V_psi(s') - Q_theta(s,a))^2 ]`,

and the claim is that this interpolates from SARSA at `tau = 0.5` to support-constrained Q-learning as `tau -> 1`. Before I lean on that, let me actually watch it happen on a toy state rather than take it on faith. Put three in-support actions at one state with behavior weights `mu = (0.5, 0.3, 0.2)` and action-values `Q = (1.0, 2.0, 0.5)`. Minimizing `E_{a~mu}[L2^tau(Q-V)]` in `V` numerically: at `tau=0.5` I get `V*=1.200`, which is exactly `sum_a mu_a Q_a = 0.5*1 + 0.3*2 + 0.2*0.5 = 1.2`, the behavior-mean — so `tau=0.5` is plain SARSA. Sweeping up, `tau=0.7 -> 1.429`, `tau=0.9 -> 1.765`, `tau=0.99 -> 1.974`, `tau=0.999 -> 1.997`, monotonically climbing toward `max_a Q_a = 2.0` as promised. So the value side behaves: stable, in-sample, multi-step, with `tau` a real dial from behavior-mean to support-max.

But now I'm stuck on the other half, and this is where I want to slow down. I have a value function trained with no explicit policy anywhere in sight. To actually *act*, I have to extract a policy from it. The standard extraction is advantage-weighted regression: fit a policy by `E[exp(beta (Q-V)) log pi(a|s)]`, which is the maximum-likelihood fit to the KL-constrained reward-maximizing policy `pi ∝ mu * exp(beta A)`. Fine. But here's the thing that keeps bothering me: what policy does my value function actually correspond to? I trained `V` to be the `tau`-expectile of `Q` over `mu`. For `tau = 0.5` that's the mean, so `V` is the value of `mu` itself — the policy is `mu`. For `tau -> 1` it's the max, so the policy is greedy. But for the `tau` I actually use, somewhere in between, what is it? I genuinely don't know. The method computes a value, and I have no description of the policy that achieves that value. And if I don't know that policy, how can I be sure that fitting a unimodal Gaussian by AWR reproduces it? Maybe it does, maybe it doesn't. That gap — "which policy is my critic implicitly evaluating?" — is the thing I have to close before I can trust any extraction.

So let me try to *derive* that policy instead of guessing. Generalize first, because the expectile is just one asymmetric loss and I suspect the structure is more general. Replace `L2^tau` with an arbitrary convex loss `f` on the residual `Q(s,a) - V(s)`, and define the value as

  `V*(s) = argmin_{V(s)} E_{a~mu(a|s)}[ f(Q(s,a) - V(s)) ]`.

Expectile is `f = L2^tau`; but let me keep `f` general and see what the optimum tells me. At the minimum the derivative in `V` vanishes. Write `f'` for the derivative of `f` with respect to its argument; differentiating through the expectation, with `d/dV (Q - V) = -1`,

  `0 = d/dV E_{a~mu}[f(Q(s,a) - V(s))]|_{V*} = - E_{a~mu}[ f'(Q(s,a) - V*(s)) ]`.

So `E_{a~mu}[f'(Q - V*)] = 0`. Now I want to massage that condition into something that looks like "the residual averages to zero under *some* distribution," because if I can do that, that distribution *is* the implicit policy. The obstacle is the `f'` — it's not the residual itself, it's some nonlinear function of it. Let me assume `f` is convex and `f'(0) = 0` (true for expectile: `f'(u) = 2|tau-1(u<0)| u`, which is zero at `u=0`; true for any sensible asymmetric loss whose minimum is at zero residual). Convexity means `f'` is nondecreasing and, with `f'(0)=0`, it has the same sign as its argument. So I can write `f'(x) = |f'(x)| * sign(x) = |f'(x)| * x/|x|`. Substitute:

  `0 = E_{a~mu}[ |f'(Q - V*)| * (Q - V*) / |Q - V*| ]`.

Stare at that. Inside the expectation is `(Q - V*)` — the residual I want — multiplied by `|f'(Q-V*)| / |Q-V*|`, which is a *nonnegative scalar weight* depending on `(s,a)`. So the equation says: the residual, weighted by `mu(a|s) * |f'|/|Q-V*|`, integrates to zero. If I fold that weight into the sampling distribution, define

  `w(s,a) = |f'(Q(s,a) - V*(s))| / |Q(s,a) - V*(s)|`,   `pi_imp(a|s) = mu(a|s) w(s,a) / Z(s)`

with `Z(s)` the normalizer, then `E_{a~mu}[w(s,a)(Q-V*)] = Z(s) * E_{a~pi_imp}[(Q-V*)]`, and the condition becomes simply

  `E_{a~pi_imp}[ Q(s,a) - V*(s) ] = 0`,   i.e.   `V*(s) = E_{a~pi_imp}[Q(s,a)]`.

Read that as a statement about `pi_imp`: it says `V*` is the *expected* `Q` under `pi_imp`, i.e. `V*` is the value function of the policy `pi_imp`. And to see it's not an accident of rearrangement, the first-order condition `E_{pi_imp}[(Q - V*)] = 0` is exactly the stationarity of `argmin_V (1/2) E_{a~pi_imp}[(Q(s,a) - V(s))^2]`, a plain mean-squared problem whose minimizer is the mean. So the very same `V*` that minimizes the generalized expectile-style loss under `mu` is also the solution of an ordinary squared loss under `pi_imp`.

I don't want to trust that purely from the algebra, so let me check it on the same three-action state. For the expectile loss `f = L2^tau`, the derivative-ratio weight `|f'|/|u|` works out to `2|tau - 1(u<0)|`, and the constant `2` cancels in the normalizer, so `w = |tau - 1(Q<V*)|` — `tau` above `V*`, `1-tau` below. At `tau=0.7` my solved `V*=1.42857`; comparing each `Q` to that, action 2 (`Q=2.0`) is above, actions 1 and 3 (`1.0, 0.5`) are below. So unnormalized `pi_imp = (0.5*0.3, 0.3*0.7, 0.2*0.3) = (0.15, 0.21, 0.06)`, which normalizes to `(0.357, 0.500, 0.143)`. Now take its `Q`-mean: `0.357*1.0 + 0.500*2.0 + 0.143*0.5 = 0.357 + 1.000 + 0.0715 = 1.4286`. That lands on `V*=1.42857` to five digits, and `E_{pi_imp}[Q - V*]` comes out at `-8e-17`, numerical zero. So the implicitly trained critic really is the critic of a concrete, named policy:

  `pi_imp(a|s) ∝ mu(a|s) * |f'(Q(s,a) - V*(s))| / |Q(s,a) - V*(s)|`.

That reframes what I built: the "actor" was there all along — it's a reweighting of the behavior policy by `w`, and the choice of `f` controls how hard `w` skews `pi_imp` away from `mu`. The gap I was stuck on — which policy does my critic evaluate? — now has an answer I can write down.

Let me make that concrete by reading off `w` for a few choices of `f`, because the shape of `w` is going to decide everything about extraction. Expectile: `f(u) = L2^tau(u) = |tau - 1(u<0)| u^2`, so `f'(u) = 2|tau - 1(u<0)| u`, and `|f'|/|u| = 2|tau - 1(u<0)|`. The constant 2 cancels in the normalizer, so

  `w_2^tau(s,a) = |tau - 1(Q(s,a) < V*(s))|`.

That's startlingly simple: a two-valued weight, `tau` when the action is above the value and `1-tau` when below. It does not depend on the *magnitude* of the advantage at all, only its sign relative to `V`. So the expectile-implicit actor reweights `mu` by a constant factor on the good half and a smaller constant on the bad half — it broadens or sharpens, but it doesn't single out the single best action; it spreads weight over a whole range of above-`V` actions. Cranking `tau` toward 1 sends `1-tau -> 0`, which kills the below-`V` actions and concentrates on the above-`V` set — more deviation from `mu`. Now the quantile loss `f(u) = |tau - 1(u<0)| |u|`: `f'(u) = |tau-1(u<0)| sign(u)`, so `|f'|/|u| = |tau-1(u<0)|/|u|`, giving

  `w_1^tau(s,a) = |tau - 1(Q < V*)| / |Q - V*|`,

which now *does* depend on magnitude — it blows up for actions near `V`, clustering the implicit policy tightly around the quantile. And the exponential, or linex, loss is the most interesting because I suspect it connects back to the KL-constrained extraction. Take `f(u) = exp(alpha u) - alpha u`. Let me solve for `V_exp` directly rather than just write the weight, because the solution is itself revealing. The objective is `argmin_V E_{a~mu}[exp(alpha(Q-V)) - alpha(Q-V)]`. Drop the term `alpha Q` that doesn't depend on `V`, and `-alpha(-V) = +alpha V`, so it's `argmin_V E_{a~mu}[exp(alpha(Q-V)) + alpha V]`. Write the expectation as a sum over actions, `sum_a mu(a|s)(exp(alpha(Q-V)) + alpha V)`, and set the `V`-derivative to zero:

  `0 = sum_a mu(a|s) ( -alpha exp(alpha(Q - V*)) + alpha )`
    `= alpha [ -exp(-alpha V*) sum_a exp(alpha Q + log mu(a|s)) + sum_a mu(a|s) ]`,

and since `sum_a mu = 1`, this gives `exp(-alpha V*) sum_a exp(alpha Q + log mu) = 1`, hence

  `V_exp(s) = (1/alpha) log sum_a exp(alpha Q(s,a) + log mu(a|s))`.

That is a log-sum-exp — a *soft* maximum — and the derivative-ratio theorem still gives its own nonnegative implicit-actor weight,

  `w_exp(s,a) = alpha | exp(alpha(Q(s,a) - V_exp(s))) - 1 | / |Q(s,a) - V_exp(s)|`.

So the exponential case has two linked faces. The theorem says the value is the mean of `Q` under the derivative-ratio actor above. The closed-form value should also be the log partition of the exponential AWR distribution `pi_exp(a|s) = mu(a|s) exp(alpha(Q(s,a) - V_exp(s)))`, because `exp(alpha V_exp)` is exactly `sum_a mu(a|s) exp(alpha Q(s,a))`. Let me put a number on this before I build anything on it, again on the three-action state, now with `alpha=1.5`. Minimizing `E_{a~mu}[exp(alpha(Q-V)) - alpha(Q-V)]` numerically gives `V_exp=1.44144`; the closed form `(1/1.5) log(0.5 e^{1.5} + 0.3 e^{3.0} + 0.2 e^{0.75})` also gives `1.44144` — they agree, so the log-sum-exp solution is right. Forming `pi_exp = mu * exp(alpha(Q - V_exp))` and summing gives `1.0000` to floating precision, confirming `V_exp` is genuinely the normalizer (log partition), not just close. And the derivative-ratio weight for this loss, `w_exp = alpha|exp(alpha(Q-V_exp))-1|/|Q-V_exp|`, fed back through the theorem yields `E_{pi_imp}[Q] = 1.44144` as well — same value from the implicit-actor side. Good, the two faces coincide numerically.

Now I can quantify the constraint AWR imposes: `KL(mu || pi_exp) = sum_a mu log(mu/pi_exp) = sum_a mu * (-(alpha(Q-V_exp))) = E_{(s,a)}[alpha(V_exp - Q)]`, proportional to the negative advantage scaled by the temperature. Checking the algebra against the numbers: directly evaluating `sum_a mu_a log(mu_a/pi_exp_a)` on the three actions gives `0.36216`, and the identity `E_{a~mu}[alpha(V_exp - Q)]` gives `0.36216` too. So `alpha` literally sets how far that exponential extraction drifts from `mu` in KL. If I train an expectile critic and then extract with AWR, I have mixed two different choices: the critic's theorem actor uses expectile sign weights, while the extraction is using the exponential/KL family.

Now I have a real diagnosis, derived not asserted. The implicit actor `pi_imp ∝ mu * w` is, for any reasonable `f`, a *reweighting of the behavior policy* — and for the expectile loss the weight is a sign-dependent constant that spreads weight over the whole above-value set. The behavior policy `mu` on these continuous control tasks is generically multimodal (a `medium` dataset is a half-trained agent, several action clusters). Reweighting a multimodal `mu` by `w` gives a multimodal `pi_imp`. So the implicit actor is complicated and multimodal — and the standard extraction fits it with a *unimodal Gaussian* via AWR. A unimodal Gaussian cannot represent a multimodal target; it will smear across the modes, putting mass in the low-density valley between them, which on a `Q`-function is exactly where OOD, erroneously-high-valued actions live. The careful in-sample value learning is being thrown away at the last step by an inadequate actor. And here is the subtle part of why this matters so much specifically for *this* family: in a normal actor-critic, the critic chases the actor, so a weak actor at least gets a critic adapted to its weaknesses; here the critic is trained completely decoupled from the actor (that decoupling is the whole source of the method's stability and hyperparameter-robustness), so the critic never compensates for a bad actor. The decoupling only pays off if the explicit actor is expressive enough to reproduce `pi_imp`. So I cannot keep the Gaussian.

What do I replace it with? My first instinct is the obvious one: keep the AWR-style importance-weighted objective but swap the Gaussian for an expressive conditional generative model — a diffusion model or a normalizing flow trained by `E[exp(beta A) log pi]` (or the diffusion analogue, a per-sample-weighted denoising loss). That would directly fit the reweighted target `pi_imp`. The reason I don't reach for it is a failure mode that's been reported for exactly this combination: highly expressive models trained with importance-weighted maximum likelihood tend to raise the likelihood of *all* training points regardless of their weights — the capacity is enough to fit the whole dataset, so the `exp(beta A)` reweighting that was supposed to skew the model toward high-advantage actions gets washed out. I can sanity-check the mechanism at the extreme: a model with capacity to represent `mu` exactly drives every per-point likelihood term to its ceiling, at which point the weights multiply terms that no longer move, and the gradient sees no skew — the weighted and unweighted optima coincide in that limit. I haven't measured how badly this bites at finite capacity here, but it points the wrong way: I'd pour in a powerful model precisely to gain expressiveness, and the expressiveness is what defeats the weighting. So training the actor *with* weights is the route I want to avoid.

Back up and reread my own derivation, because the answer is sitting in the form of `pi_imp` and I almost walked past it. The implicit actor is `pi_imp(a|s) ∝ mu(a|s) w(s,a)` — a reweighting of `mu`. I don't need to *bake* the weighting into the model's training. I can separate the two factors. Train a model to represent `mu` alone — pure behavior cloning, no weights, no critic, nothing for the capacity to wash out, because there's nothing to skew; just learn the data distribution as well as possible. Then, at evaluation time, *sample* from that behavior model and apply the desired critic weights by importance resampling. Concretely: at state `s`, draw `N` candidate actions `a_1, ..., a_N ~ mu_phi(.|s)`, compute weights for those candidates, normalize `p_i = w_i / sum_j w_j`, and resample one action from the categorical `p`. If I use the derivative-ratio `w` for the chosen loss, the resampled action is distributed as `mu_phi * w / Z = pi_imp` in the large-`N` limit. If I want the exponential/KL extraction, I use the finite-sample softmax `p_i ∝ exp(beta (Q(s,a_i)-V(s)))`. The model only ever has to do one easy job (represent `mu`); the critic does the skewing at eval, where there's no training pathology to wash anything out. This also keeps the behavior model fully decoupled from critic learning — it never touches `Q` or `V` during training, only at sampling time — preserving the stability and the cheap, tuning-light training that decoupling buys, while the critic learning stays the in-sample value method I trust.

A refinement falls out of the same picture. I want the policy I run at eval to be *good*, and the implicit actor for the expectile loss is a broad reweighting that still has nonzero probability of picking mediocre actions (its weight is just `tau` vs `1-tau`, not sharply peaked). When the goal at deployment is to act well rather than to faithfully reproduce a stochastic distribution, I'd rather take the *best* candidate. That's the limit where the weight becomes a one-hot on the highest-`Q` sample — `pi(s) = argmax_{a_i ~ mu_phi(.|s)} Q(s, a_i)` — a greedy selection among `N` in-support candidates. This mirrors the standard practice of using a stochastic actor for *learning* and a deterministic one for *evaluation*: the stochasticity helped the critic see varied actions, but at deployment determinism (pick the argmax) avoids ever rolling the dice on a bad action. The derivative-ratio resample, the exponential softmax over advantage, and the greedy argmax are three ways to collapse the same `N` behavior-model candidates using the critic; the training machinery is identical and only the inference weighting differs.

Now the action model. I've been saying "expressive generative model"; let me commit to diffusion and then be honest about its failure modes, because there's a real one I have to fix. A DDPM models `mu(a|s)` through a fixed forward chain `q(a_t|a_{t-1}) = N(sqrt(1-beta_t) a_{t-1}, beta_t I)` and a learned reverse chain, trained by epsilon-prediction: corrupt a real action to `sqrt(abar_t) a + sqrt(1-abar_t) eps` and regress the network's output to the noise `eps`,

  `L_mu(phi) = E_{t, eps, (s,a)~D}[ || eps - mu_phi(sqrt(abar_t) a + sqrt(1-abar_t) eps, s, t) ||^2 ]`,

and sample by running the reverse step `a_{t-1} = (1/sqrt(alpha_t))(a_t - (beta_t/sqrt(1-abar_t)) mu_phi(a_t,s,t)) + sqrt(beta_t) eps` from `a_T ~ N(0,I)`. The catch is that on these low-dimensional continuous action spaces, a plain MLP score network fits the modes loosely and emits *outlier* samples that land far outside the data — and an outlier action is exactly an OOD action that my `Q`-function, never having seen it, may score erroneously high. If my candidate set contains such outliers, the `argmax`/softmax over `Q` will gleefully pick them, and I'm right back to the OOD-exploitation problem I built the whole in-sample machinery to avoid. So the quality of the behavior model's samples is not a cosmetic detail; it directly gates whether the critic-based selection is safe. Increasing batch size and capacity helps fit the distribution and reduces outliers, but doesn't eliminate them. What does help decisively is making the score network high-capacity *and* well regularized at once — the same recipe that stabilizes large sequence models. So I use a residual MLP with LayerNorm inside each block (and a touch of dropout), `n` blocks of `Dropout -> LayerNorm -> Linear(h, 4h) -> activation -> Linear(4h, h)` with a residual skip — the LayerNorm and the residual structure keep the high-capacity network from producing the wild outliers a bare MLP does, and the wider `4h` inner dimension gives it the capacity to fit the modes sharply. This regularized residual network is what makes the sample-and-reweight scheme robust to `N`: with a bare MLP, more candidates means more chances to surface an outlier the critic over-scores, so increasing `N` can *hurt*; with the regularized residual net the candidates stay in-support, so more candidates only sharpens the selection.

A couple of smaller choices follow from the setting. The number of diffusion steps `T`: for these control datasets, which are relatively deterministic compared to noisy 2D toy distributions, a small `T` (around 5) is attractive because at eval I have to resample the entire reverse chain *every environment step*, so a large `T` immediately makes inference slow. The diffusion wrapper is a discrete VP-SDE/DDPM-style sampler with a cosine noise schedule unless another schedule is passed. For the value networks I keep the simple, proven choices: `Q` as a twin to reduce overestimation (take the min for the value target), a target `Q`, Adam at `3e-4` for `Q` and `V`, and cosine decays on the optimizers in the training loop. The expectile `tau` is the one knob that genuinely controls how far the expectile implicit actor deviates from `mu` (higher `tau` -> more deviation, needed for stitching-heavy tasks), so it's the natural per-domain hyperparameter, and because the critic is decoupled the method is forgiving about everything else.

Let me also settle the training schedule between the critic and the behavior model, since they're independent objectives. They share nothing during training, so I can interleave them freely. In practice I update the critic on roughly every other gradient step and the behavior model every step — the behavior model is the harder fit (it has to nail the whole multimodal action distribution well enough that its samples are clean), so it deserves at least as many updates, while the in-sample critic converges quickly and is cheap. Concretely the critic update is: take the expectile `V` step `v_loss = mean(|tau - 1(Q_targ(s,a) - V(s) < 0)| * (Q_targ(s,a) - V(s))^2)`, then the `Q` step with target `r + gamma (1 - done) V(s')` against both twin heads, then Polyak the `Q` target; and every step run one denoising-BC step on the diffusion actor.

So let me write it as the actual code, filling the three slots — the in-sample critic rule, the behavior-model objective, and the inference-time selection — in the harness I already have. The critic learning is pure IQL; the actor is a plain BC diffusion model; the contribution lives in keeping them decoupled and combining samples-from-`mu_phi` with critic weights at eval.

```python
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion import BaseNNDiffusion


# --- behavior model: residual-MLP score network (high capacity, well regularized,
#     so its samples stay in-support and the critic-based selection can't be tricked
#     by OOD outliers) --------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4), nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x):
        return x + self.net(x)          # residual skip + LayerNorm tame the outliers


class ScoreMlp(BaseNNDiffusion):         # epsilon-prediction net for mu_phi(a|s)
    def __init__(self, obs_dim, act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1,
                 timestep_emb_type="positional", timestep_emb_params: Optional[dict] = None):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.obs_dim = obs_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.Mish(),
                                      nn.Linear(emb_dim * 2, emb_dim))
        self.affine_in = nn.Linear(obs_dim + act_dim + emb_dim, hidden_dim)
        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.affine_out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, noise, condition):
        if condition is None:
            condition = torch.zeros(x.shape[0], self.obs_dim, device=x.device)
        t = self.time_mlp(self.map_noise(noise))           # positional timestep embedding
        x = torch.cat([x, t, condition], -1)
        return self.affine_out(self.ln_resnet(self.affine_in(x)))


# --- in-sample critic: twin-Q + a value net (IQL value/critic rule) ----------------
class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, 1))
        self.Q1, self.Q2 = head(), head()

    def both(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.Q1(x), self.Q2(x)

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class V(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        return self.V(obs)


# actor is an epsilon-prediction DDPM wrapping ScoreMlp, exposing:
#   actor.update(act, obs) -> {"loss": ...}   # one weight-free denoising-BC step
#   actor.sample(prior, condition_cfg=obs, n_samples=..., ...) -> (actions, log)
q_net   = TwinQ(obs_dim, act_dim)
q_targ  = deepcopy(q_net).requires_grad_(False).eval()
v_net   = V(obs_dim)
q_optim = torch.optim.Adam(q_net.parameters(), lr=3e-4)
v_optim = torch.optim.Adam(v_net.parameters(), lr=3e-4)


def train_step(batch, n_step, tau, discount, ema=0.995):
    obs, act = batch["obs"], batch["act"]
    next_obs, rew, tml = batch["next_obs"], batch["rew"], batch["tml"]

    # ---- in-sample critic update (dataset actions only) ----
    if n_step % 2 == 0:
        # expectile V: V(s) -> tau-expectile of Q over dataset actions (the action-max,
        # done implicitly by the asymmetric loss; this V is the value of pi_imp)
        q = q_targ(obs, act)
        v = v_net(obs)
        u = q - v
        v_loss = (torch.abs(tau - (u < 0).float()) * u ** 2).mean()
        v_optim.zero_grad(); v_loss.backward(); v_optim.step()

        # SARSA-style TD on Q against V(s') -- V already absorbed the action-expectile,
        # so the target carries no OOD action and no per-sample dynamics luck
        with torch.no_grad():
            td_target = rew + discount * (1 - tml) * v_net(next_obs)
        q1, q2 = q_net.both(obs, act)
        q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
        q_optim.zero_grad(); q_loss.backward(); q_optim.step()

        for p, pt in zip(q_net.parameters(), q_targ.parameters()):
            pt.data.copy_(ema * p.data + (1 - ema) * pt.data)   # target update

    # ---- behavior model: pure (weight-free) diffusion BC of mu(a|s) ----
    bc_loss = actor.update(act, obs)["loss"]
    return bc_loss


@torch.no_grad()
def select_action(obs, num_candidates, act_dim, weight_temperature, args):
    # draw N in-support candidates from the behavior model, then let the critic
    # apply the practical exponential-style advantage reweighting
    n = obs.shape[0]
    obs_rep = obs.unsqueeze(1).repeat(1, num_candidates, 1).view(-1, obs.shape[-1])
    prior = torch.zeros((n * num_candidates, act_dim), device=obs.device)
    act, _ = actor.sample(prior, solver=args.solver, sample_steps=args.sampling_steps,
                          condition_cfg=obs_rep, w_cfg=1.0,
                          n_samples=n * num_candidates, use_ema=args.use_ema,
                          temperature=args.temperature)

    q = q_targ(obs_rep, act)
    v = v_net(obs_rep)
    adv = (q - v).view(-1, num_candidates, 1)            # advantage of each candidate
    w = torch.softmax(adv * weight_temperature, dim=1)   # soft reweight toward high-adv
    act = act.view(-1, num_candidates, act_dim)
    p = (w / w.sum(1, keepdim=True)).squeeze(-1)
    idx = torch.multinomial(p, 1).squeeze(-1)            # resample one reweighted candidate
    return act[torch.arange(act.shape[0]), idx]
    # greedy variant: idx = q.view(-1, num_candidates).argmax(1)  -> argmax_i Q(s, a_i)
```

The causal chain, start to finish. Offline RL diverges because bootstrapping queries the value of out-of-distribution actions; the in-sample fix replaces the explicit `max_{a'} Q(s',a')` with a `tau`-expectile of `Q` over dataset actions on a separate value net, plus a SARSA TD backup on `Q` — stable, multi-step, never touching an OOD action. But that leaves "which policy does the critic evaluate?" unanswered, so I generalized the value loss to an arbitrary convex `f` with `f'(0)=0`, set its derivative in `V` to zero, and used `f'(x) = |f'(x)| x/|x|` to rewrite the optimality condition as `E_{pi_imp}[Q - V*] = 0`, proving `V*` is the value of the implicit actor `pi_imp ∝ mu * |f'(Q-V*)|/|Q-V*|` — IQL is an actor-critic method. The expectile loss gives the sign-dependent constant weight `|tau - 1(Q<V*)|`, the quantile loss gives the same sign weight divided by `|Q-V*|`, and the exponential loss gives `w_exp = alpha |exp(alpha(Q-V_exp))-1|/|Q-V_exp|` while also producing the log-partition value `V_exp = (1/alpha) log sum exp(alpha Q + log mu)` for the exponential/KL policy whose behavior-to-policy KL is `E[alpha(V_exp-Q)]`. The implicit actor is a reweighting of `mu` and hence can be multimodal, which a unimodal Gaussian cannot reliably represent — and since the decoupled critic never adapts to a weak actor, the extraction has to be expressive. Training an expressive model with importance weights fails for the reason importance-weighted deep likelihood can fail generally, so instead I train an expressive model for `mu` alone by pure diffusion BC, and apply critic weights at eval by sampling `N` candidates: exact derivative-ratio weights for the theorem actor, a softmax over advantages for the practical exponential-style sampler, or the argmax-`Q` candidate for a deterministic eval policy. To keep the candidates in-support, the score network is a high-capacity, LayerNorm-regularized residual MLP, with a small number of diffusion steps for fast control sampling. The result drops into the standard offline harness as three slots: an IQL critic rule, a weight-free diffusion BC actor, and a critic-reweighted candidate selection at inference.
