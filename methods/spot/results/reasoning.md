Let me start from what actually breaks. I have a fixed batch `D = {(s,a,r,s')}` from some unknown behavior policy `pi_beta`, no environment to poke, and I want the best continuous-control policy I can squeeze out of it. The reflex is to run TD3 on the batch: fit twin critics `Q_theta` to the Bellman target and train a deterministic actor to maximize `Q_theta(s, pi_phi(s))`. The known failure mode is specific enough that I should write it down before I reach for any cure. The Bellman target is `r + gamma * Q_target(s', pi_phi(s'))`. That term evaluates the critic at `pi_phi(s')` — an action the *actor* chose, not one the dataset contains. Online, that's fine: if the actor proposes something silly the environment punishes it and the data corrects the critic. Offline there is no correction loop. When `pi_phi(s')` lands on an action the batch never saw, `Q_theta` is extrapolating, and a neural net's extrapolation off the data manifold is not anchored to anything — it can be wildly, arbitrarily high. The actor is a maximizer, so it is *attracted* to exactly those out-of-distribution actions with spuriously inflated value. Those inflated values get bootstrapped back through the backup into neighboring states, the next actor update chases them even harder, and the whole thing runs away. So the disease is not "the dataset is small" or "the reward is sparse" in the first instance — it is that the `max`/`argmax` over actions reaches outside the data and the critic lies out there. The fix has to forbid the actor from leaving the region where the behavior policy actually put mass.

What is that region, exactly? The accepted answer is the *support* of `pi_beta`: in state `s`, only allow actions with `pi_beta(a|s) > eps` for some threshold. Kumar and colleagues made this rigorous with a distribution-constrained backup — restrict the maximization in the Bellman operator to a set of policies `Pi` instead of all actions — and proved a bound that exposes a genuine tradeoff. If I specialize `Pi` to the support set `Pi_eps = {pi : pi(a|s)=0 whenever pi_beta(a|s)<eps}` and call its backup `T_eps`, the fixed point `Q*_eps` of `T_eps` is the supported optimal value function, and I can ask how far it sits from the true `Q*`. Let me actually do that bound, because it's the thing that tells me the constraint is *safe*. Write `alpha(eps) = ||T Q* - T_eps Q*||_inf = max_{s,a}|T Q*(s,a) - T_eps Q*(s,a)|`, the worst-case gap between one unconstrained backup and one support-restricted backup, applied to the true optimum. Then

  `||Q* - Q*_eps||_inf = ||T Q* - T_eps Q*_eps||_inf`,

using that `Q*` is `T`'s fixed point and `Q*_eps` is `T_eps`'s. Insert and remove `T_eps Q*` and apply the triangle inequality:

  `||T Q* - T_eps Q*_eps||_inf <= ||T_eps Q* - T_eps Q*_eps||_inf + ||T Q* - T_eps Q*||_inf`.

The first term is `T_eps` applied to two different value functions; `T_eps`, like any Bellman backup, is a `gamma`-contraction in sup-norm, so it's `<= gamma ||Q* - Q*_eps||_inf`. The second term is exactly `alpha(eps)`. So

  `||Q* - Q*_eps||_inf <= gamma ||Q* - Q*_eps||_inf + alpha(eps)`,

and solving, `||Q* - Q*_eps||_inf <= alpha(eps)/(1-gamma)`. There it is: restricting to the support costs at most `alpha(eps)/(1-gamma)` in value. A tighter support (bigger `eps`) shrinks the OOD/extrapolation risk but raises `alpha(eps)` — the restriction perturbs the backup more, so the supported optimum is more suboptimal. A looser support does the reverse. `eps` is the single knob trading "stay where the critic is trustworthy" against "keep enough room to be good." This is reassuring: the support constraint is not a hack, it provably loses only a controlled, contraction-amplified amount. The supported optimal policy is then the greedy one over supported actions, `pi*_eps(s) = argmax_{a: pi_beta(a|s)>eps} Q*_eps(s,a)`.

Now, how do people *enforce* this in the function-approximation world, and where exactly does each approach fall short? Two camps. The parameterization camp — BCQ and its descendants — bakes the support into the policy's architecture. BCQ fits a conditional VAE generative model of `pi_beta`, samples `N` candidate actions from it, perturbs each by a small learned bounded `xi_phi(s, a_i)`, and *defines* the policy as the argmax of `Q` over those perturbed candidates. PLAS runs the policy in the VAE's latent space and decodes; EMaQ drops the perturbation and just argmaxes `Q` over VAE samples. These genuinely respect a density/support notion — the candidates come from the behavior model, so they're in-distribution by construction. But look at the cost: *every single action selection*, at training and at deployment, has to sample the generative model, perturb, and score with the critic. That's an intrusive, slow inference path; it welds the algorithm to its generative component; and as the minimalist-offline-RL critique pointed out, it tangles up which component is actually earning the performance and blocks me from just dropping in a better online RL algorithm later.

The other camp — regularization — keeps the policy a plain network and instead adds a penalty to the actor loss that pulls it toward the behavior policy. This is *pluggable*: one forward pass of the policy at inference, a one-or-two-line change to a standard algorithm. BEAR penalizes by the sampled maximum mean discrepancy between actions from `pi_phi` and dataset actions, with a Lagrange multiplier holding it under a threshold. TD3+BC, even simpler, just adds a behavior-cloning term: maximize `lambda Q(s, pi(s)) - (pi(s) - a)^2`, with a clever normalization I'll come back to. I love the pluggability. But here's the wall, and it's the same wall for both: a *divergence* (or a BC squared-error) measures how close the learned *distribution* is to the behavior *distribution*, and that is not the same thing as the *support* condition the theory actually asked for. The support condition says "put your action where `pi_beta` has density above `eps`" — it says nothing about matching `pi_beta`'s shape. Matching the distribution is strictly more restrictive: if `pi_beta` is broad or near-uniform, forcing `pi_phi` close to it in KL or MMD or L2 drags the policy toward random behavior even when the data clearly supports a sharp, strong policy. BEAR tries to argue its MMD is really a *support* matcher, but that claim leans on an empirical curiosity — that at *small sample counts* the sampled MMD between a distribution and a uniform distribution over its support can dip below the MMD between the distribution and itself. That's fragile, sample-count-dependent, and in practice the constraint comes out loose: OOD actions leak through and the method is unstable. TD3+BC's `(pi(s)-a)^2` is even more directly an *imitation* of the single logged action, so on suboptimal or multimodal data it over-constrains and can't use the freedom the support genuinely permits.

So I'm cornered. Parameterization gets the *right notion* (density/support) but is intrusive. Regularization is *pluggable* but enforces the *wrong notion* (distributional closeness). I want both at once: a pluggable penalty term that nonetheless directly encodes the density-based support condition. The mismatch is the clue. The reason every regularizer is "indirect" is that it measures a *distance between distributions* and hopes that controls the support. But the constraint I wrote down — `pi_beta(a|s) > eps` — isn't a distance at all. It's a statement about the *value of the behavior density* at the specific action the policy takes. So why am I measuring a divergence? Why not just... evaluate the behavior density at `pi_phi(s)` and require it be large? The constraint literally is: at the action my deterministic policy outputs, the behavior density should exceed `eps`. Written as an optimization over the policy parameters, with log for convenience,

  `max_phi E_{s~D}[Q_theta(s, pi_phi(s))]  subject to  min_s log pi_beta(pi_phi(s)|s) > log eps`.

This is exactly the support constraint, *directly* — no divergence, no proxy. The log is just so the density enters additively and the products turn into sums; `log eps =: eps_hat`.

But that constraint is impractical as written. It's a `min` over states — a separate constraint at *every* state in a continuous, effectively infinite state space. I can't enforce infinitely many constraints. The standard move, used in TRPO for the KL constraint, in advantage-weighted regression, in BEAR — is to relax the per-state hard constraint to a constraint on the *average* over the state distribution:

  `max_phi E_{s~D}[Q_theta(s, pi_phi(s))]  subject to  E_{s~D}[log pi_beta(pi_phi(s)|s)] > eps_hat`.

It's a heuristic relaxation — I'm trading "every state is in-support" for "in-support on average" — but it's the tractable, well-trodden version, and it keeps the *density* as the object being constrained. Now convert to an unconstrained problem the usual way: form the Lagrangian, treat the constraint term as a penalty with multiplier `lambda`, and since I'll minimize a loss I flip signs:

  `J_pi(phi) = E_{s~D}[ -Q_theta(s, pi_phi(s)) - lambda * log pi_beta(pi_phi(s)|s) ].`

That's the whole idea of the regularizer, and it's pluggable by construction — it's just an extra term on the standard actor loss, the policy is still a plain network, inference is one forward pass. The penalty *is* the behavior log-density of the action taken, so it's a *direct* density/support constraint, not a divergence. `lambda` plays the role of `eps`: crank it up and the policy is pushed onto high-density actions (tight support); turn it down and the value term dominates (loose support). It's the same constraint-strength lever the theory's bound talked about, now a concrete coefficient.

One catch: I don't have `pi_beta`. I only have samples from it — the dataset actions. So I need to *estimate* the behavior density at arbitrary `(s, a)` points, including the off-data actions my policy will probe. This is plain density estimation, and I want a model flexible enough to capture whatever `pi_beta` looks like — and offline behavior policies are routinely multimodal (mixtures of experts, replay of many policies, the Adroit `cloned` mixtures of expert-plus-noise). A single Gaussian can't represent that; it would smear a bimodal behavior into one blob and call the valley between the modes "in support." The conditional VAE is the natural fit — it's already the standard behavior model in offline RL precisely because its latent-variable structure captures near-arbitrary, multimodal conditional distributions. So model `pi_beta(a|s) ~ p_psi(a|s) = ∫ p_psi(a|z,s) p(z|s) dz` with a fixed prior `p(z|s) = N(0, I)`.

But the marginal `p_psi(a|s)` is intractable — that integral over `z`. The VAE's whole trick is to lower-bound it. Introduce an approximate posterior `q_varphi(z|a,s)` and write the evidence lower bound:

  `log p_psi(a|s) = log E_{q_varphi}[ p_psi(a,z|s) / q_varphi(z|a,s) ] >= E_{q_varphi}[ log p_psi(a,z|s) / q_varphi(z|a,s) ]`

by Jensen, since `log` is concave. Expand the integrand: `log p_psi(a,z|s)/q = log p_psi(a|z,s) + log p(z|s) - log q_varphi(z|a,s)`, and the last two terms in expectation are exactly `-KL(q_varphi(z|a,s) || p(z|s))`. So

  `log p_psi(a|s) >= E_{q_varphi(z|a,s)}[log p_psi(a|z,s)] - KL(q_varphi(z|a,s) || p(z|s)) =: -L_ELBO(s,a; varphi, psi).`

The gap of this bound is precisely `KL(q_varphi(z|a,s) || p_psi(z|a,s)) >= 0` — that's the difference `log p - (-L_ELBO)`, and it's nonnegative, so `-L_ELBO` is a genuine *lower* bound on the log-density. That's convenient for me in a specific way: I'm using `log pi_beta` as a constraint that I want *above* a threshold, so substituting a lower bound `-L_ELBO` is conservative in the right direction — if the lower bound is high, the true density is at least that high. So I'll plug `-L_ELBO` in for `log pi_beta` in the penalty.

I should ask whether the looseness of the bound matters. The gap is that KL; I can tighten it. Burda's importance-weighted estimator puts the `L` samples *inside* the log:

  `log_pi_beta_hat(a|s; varphi, psi, L) = E_{z_l ~ q_varphi}[ log (1/L) sum_{l=1}^{L} p_psi(a, z_l|s) / q_varphi(z_l|a,s) ].`

This is still a lower bound (by Jensen on the averaged ratio), and it provably tightens monotonically as `L` grows, recovering `log p_psi` as `L -> inf`. So I have a knob `L` controlling the tightness of my density estimate. Tempting to use a big `L`. But let me think about whether I actually need it in the actor update. The density term is a soft constraint signal, not a reporting metric; I need it to rise on in-support actions and fall off-support, and I need its gradient to be stable. Setting `L=1` collapses the importance-weighted estimator back to the plain ELBO estimator, `-L_ELBO`. There is a real bonus at `L=1`: I can keep the KL term *analytic* (it has a closed form for Gaussian `q` and Gaussian prior) instead of estimating it by sampling, which means lower variance in the gradient. So `L=1` is simpler and lower-variance; larger `L` is there if I decide the tighter density estimate is worth the sampling cost. I'll use the `L=1` ELBO loss as the actor's density penalty. Call it `neg_log_beta = L_ELBO(s, pi_phi(s))`; minimizing it pushes `pi_phi(s)` toward high behavior density. The penalty in the actor loss is `+ lambda * neg_log_beta`.

Let me pin down the ELBO loss concretely, because the form matters for the code. The encoder maps `(s,a)` to a Gaussian posterior `q_varphi(z|a,s) = N(mean, std^2)` via a shared MLP trunk and two heads `mean`, `log_std` (clamped for stability). Reparameterize `z = mean + std * noise`. The decoder maps `(s,z)` back to a reconstructed action `u = max_action * tanh(MLP(s,z))`. With a Gaussian decoder of fixed variance, the reconstruction term `E_q[log p_psi(a|z,s)]` is, up to constants and a positive scale, a negative squared error, so the reconstruction *loss* is `recon = mean((u - a)^2)`. The KL of two Gaussians (posterior vs `N(0,I)` prior) has the standard closed form `KL = -0.5 * mean(1 + log std^2 - mean^2 - std^2)`. The practical VAE loss is `recon + beta * KL` — `beta` weighting the KL, a beta-VAE-style balance; I'll take `beta = 0.5`, down-weighting the KL so the model spends enough capacity on faithful reconstruction rather than over-regularizing the latent. This same `recon + beta*KL` is both what I *train* the VAE on and what I evaluate as `neg_log_beta` at `pi_phi(s)`: in the implementation it is proportional to the negative ELBO estimator up to constants, so it has the right sign and serves as the density-constraint signal.

Now the base algorithm. I said pluggable on top of off-policy; *which* off-policy algorithm? The two candidates are TD3 (deterministic actor, twin clipped critics, target smoothing, delayed updates) and SAC (stochastic actor, entropy bonus). My enemy is overestimation on OOD actions, and TD3 was literally built to suppress overestimation: the `min` of twin critics in the target caps the bootstrap, target policy smoothing keeps the critic from latching onto a sharp spurious peak, delayed updates let the value settle before the actor chases it. Every one of those is exactly what I want offline. SAC's stochastic policy is the opposite of what I want here on two counts. First, a stochastic actor *samples* actions, and the tails of that sampling distribution will reach into OOD regions and pull in erroneous values, biasing and adding variance to the policy gradient. Second, the entropy bonus *rewards spreading out* — it actively pushes the policy toward the edges of (and past) the support, which is precisely the behavior I'm trying to forbid. Removing entropy from a stochastic actor would also remove the main reason to keep a stochastic policy in the actor objective; the variance parameters would no longer buy me a support-control mechanism, while the sampled actor still keeps the critic exposed to off-support tails. So: TD3 base, deterministic actor, twin critics returning `(batch, 1)`.

There's one more thing I want from TD3+BC: the normalization trick, because it solves a problem I'm about to have. My actor loss is `-Q + lambda * neg_log_beta`. The two terms live on totally different scales: `Q` has the scale of returns, which depends on the reward magnitude of the task, while `neg_log_beta` (an ELBO) has the scale of a log-density. If `Q` is large (high-reward task), the value term swamps the penalty and `lambda` would have to be re-tuned per task; if `Q` is small, the penalty dominates. TD3+BC fixed this by normalizing the value term by the mean absolute `Q` over the minibatch: they wrote `lambda = alpha / ((1/N) sum_i |Q(s_i, a_i)|)` and used it to scale the *value* term. Translating to my loss: divide `-Q.mean()` by `(1/N) sum_i |Q(s_i, pi(s_i))|`, detached so I don't differentiate through the normalizer (it's a scaling, not part of the objective). That makes the value term roughly order-1 regardless of reward scale, so a single `lambda` schedule transfers across tasks. Define `norm_q = 1 / |Q|.mean().detach()` and the actor loss becomes

  `J_pi = -norm_q * Q.mean() + lambda * neg_log_beta.mean().`

This normalization is not meant to change which action has larger value inside a minibatch; its job is to keep the value gradient on a comparable scale to the density penalty. That cross-task `lambda` stability is important enough that I keep it as the default option, with a switch to turn it off when I want the raw TD3 actor scale.

Putting the pieces in order gives a two-phase algorithm. Phase one: pretrain the VAE on `(s, a)` pairs from the dataset by minimizing `recon + beta*KL`, for enough iterations (order `10^5`) to get a faithful behavior-density estimator — this happens *before* policy training, because the constraint term is meaningless until the density model is good, and freezing a trained estimator keeps the constraint stationary. Phase two: ordinary TD3 — twin-critic Bellman update with clipped double-Q target and target smoothing every step, and every `policy_freq=2` steps the actor update with my augmented loss plus soft target updates. The VAE's parameters (`750`-wide, 3-layer encoder/decoder, latent dim `2 * action_dim`, lr `1e-3`, `beta=0.5`) and TD3's (`256`-wide nets, optional actor dropout, `tau=0.005`, `gamma=0.99`, `policy_noise=0.2`, `noise_clip=0.5`, Adam `3e-4`) are the standard settings; `lambda` is the one constraint-strength hyperparameter to tune.

Now the part the offline-to-online setting really cares about: fine-tuning. Notice something about the actor loss that falls out for free. Set `lambda = 0`. The penalty vanishes and the loss is just `-norm_q * Q.mean()` — which is *exactly* the standard TD3 actor objective (normalization aside). So my offline method *is* TD3 with a knob that, at zero, returns it to plain online TD3. That's the seamless-transition property: there's no architectural gap between my offline algorithm and a well-established online one; they're the same algorithm at two settings of one coefficient. This is the concrete pay-off of having insisted on a *pluggable regularizer* instead of a parameterization — a parameterized policy can't smoothly "turn off" its constraint, but a coefficient on a penalty can.

So how do I fine-tune? Offline I've pretrained with a strong constraint (large `lambda`) to keep the critic honest on the fixed data. Once I start collecting fresh online interactions, the data distribution shifts toward the policy's own actions — exactly the actions the critic *can* now learn about from real feedback — so the reason for the constraint erodes. If I held `lambda` fixed, the policy would stay pinned near the *offline* behavior support and fail to exploit the new online data; the conservatism that was protective offline becomes a cap on improvement online. So I *cool* `lambda` over the online phase: decay it from its offline value toward a small floor, relaxing the support constraint as real feedback accumulates and lets the policy safely venture beyond the original support. A simple linear decay does it: `lambda_t = lambda * max(lambda_end, 1 - online_it / max_online_steps)`, so it ramps down linearly and then holds at `lambda * lambda_end`. I keep a *floor* `lambda_end > 0` rather than going to zero because in the hardest sparse-reward, high-dimensional tasks bootstrapping error stays dangerous even with online data — a residual constraint keeps the critic from going haywire mid-finetune. (On those tasks one also wants a larger discount, e.g. `gamma = 0.995`, to propagate the sparse reward over the long horizon.) And one more decision: I *freeze the VAE* during online fine-tuning. Behavior models are hard to update online — the online data is a moving, policy-dependent distribution, not the fixed `pi_beta` the density model was meant to capture — so re-fitting it online would chase a target that no longer means "behavior support." The frozen offline VAE keeps defining a stable notion of "where the offline data was," and the *decaying* `lambda` is what controls how much that notion still binds.

Let me write the actual code, filling the slots of the offline-to-online actor-critic harness. First the VAE — encoder, decoder, and the ELBO loss that doubles as the density-constraint signal:

```python
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class VAE(nn.Module):
    """Conditional VAE density estimator for the behavior policy pi_beta(a|s).
    Trained with an ELBO-based loss; the practical loss is proportional to
    the negative ELBO estimator used as the density penalty."""

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750):
        super().__init__()
        # encoder q_varphi(z | a, s): shared trunk + mean / log_std heads
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        # decoder p_psi(a | z, s)
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim

    def encode(self, state, action):
        h = F.relu(self.e1(torch.cat([state, action], -1)))
        h = F.relu(self.e2(h))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-4, 15)   # clamp for numerical stability
        return mean, torch.exp(log_std)

    def decode(self, state, z):
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))   # Gaussian-decoder mean, bounded

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)            # reparameterization
        return self.decode(state, z), mean, std

    def elbo_loss(self, state, action, beta, num_samples=1):
        """L_ELBO = recon + beta * KL. This is neg_log_beta: minimizing it
        pushes (s, action) toward high estimated behavior density. At L=1 this
        keeps the KL analytic and gives the low-variance default penalty."""
        mean, std = self.encode(state, action)
        # draw num_samples latents per item (L=1 by default)
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_s + std_s * torch.randn_like(std_s)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        u = self.decode(state_r, z)
        recon = ((u - action_r) ** 2).mean(dim=(1, 2))    # Gaussian recon: -log p(a|z,s) up to const
        kl = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon + beta * kl                          # per-sample, shape (batch,)

    def iwae_loss(self, state, action, beta, num_samples=10):
        return -self.importance_sampling_estimator(state, action, beta, num_samples)

    def importance_sampling_estimator(self, state, action, beta, num_samples=500):
        mean, std = self.encode(state, action)
        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_enc + std_enc * torch.randn_like(std_enc)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        mean_dec = self.decode(state_r, z)
        std_dec = torch.ones_like(mean_dec) * math.sqrt(beta / 4)
        log_qzx = td.Normal(mean_enc, std_enc).log_prob(z)
        log_pz = td.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)
        log_pxz = td.Normal(mean_dec, std_dec).log_prob(action_r)
        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        return w.logsumexp(dim=-1) - math.log(num_samples)
```

Then the TD3 actor and critic. The critic is one module with two Q heads, because the implementation updates both heads with one optimizer and uses `Q1` alone for the actor gradient:

```python
def weights_init_(m, init_w=3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dropout=None,
                 hidden_dim=256, init_w=None):
        super().__init__()
        if dropout:
            self.l1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Dropout(dropout))
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        if init_w:
            weights_init_(self.l3, init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return self.max_action * torch.tanh(a) if self.max_action is not None else a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=None):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        if init_w:
            weights_init_(self.l3, init_w)
            weights_init_(self.l6, init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)
```

And the algorithm itself — a pretrained frozen VAE is passed in, the offline `train` path uses the fixed `lambda`, and the online path optionally cools it:

```python
class SPOT_TD3:
    def __init__(self, vae, state_dim, action_dim, max_action, discount=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 beta=0.5, lambd=1.0, lr=3e-4, actor_lr=None,
                 without_Q_norm=False, num_samples=1, iwae=False,
                 actor_hidden_dim=256, critic_hidden_dim=256, actor_dropout=0.1,
                 actor_init_w=None, critic_init_w=None,
                 lambd_cool=False, lambd_end=0.2, max_online_steps=1_000_000,
                 device="cuda"):
        self.device = device
        self.total_it = 0
        self.actor = Actor(state_dim, action_dim, max_action, dropout=actor_dropout,
                           hidden_dim=actor_hidden_dim, init_w=actor_init_w).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden_dim,
                             init_w=critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.vae = vae.eval()              # trained separately, frozen during policy training
        self.max_action = max_action
        self.discount, self.tau = discount, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.beta, self.lambd, self.num_samples = beta, lambd, num_samples
        self.iwae = iwae
        self.without_Q_norm = without_Q_norm
        self.lambd_cool, self.lambd_end = lambd_cool, lambd_end
        self.max_online_steps = max_online_steps
        self.online_it = 0

    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            s = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(s).cpu().data.numpy().flatten()
            self.actor.train()
            return action

    def _train_step(self, replay_buffer, batch_size, lambd):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.Q1(state, pi)
            neg_log_beta = (self.vae.iwae_loss(state, pi, self.beta, self.num_samples)
                            if self.iwae else
                            self.vae.elbo_loss(state, pi, self.beta, self.num_samples))
            if self.without_Q_norm:
                actor_loss = -q.mean() + lambd * neg_log_beta.mean()
            else:
                norm_q = 1.0 / q.abs().mean().detach()
                actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train(self, replay_buffer, batch_size=256):
        self._train_step(replay_buffer, batch_size, self.lambd)

    def train_online(self, replay_buffer, batch_size=256, max_online_steps=None):
        self.online_it += 1
        max_online_steps = max_online_steps or self.max_online_steps
        lambd = (self.lambd * max(self.lambd_end, 1.0 - self.online_it / max_online_steps)
                 if self.lambd_cool else self.lambd)
        self._train_step(replay_buffer, batch_size, lambd)
```

Naive off-policy actor-critic on a fixed batch fails because the Bellman max reaches OOD actions where the critic extrapolates to garbage, the actor is pulled there, and the error bootstraps and runs away — so the actor must be confined to the behavior policy's support. The support-constrained backup is provably only `alpha(eps)/(1-gamma)` suboptimal, so confinement is safe, with `eps` trading conservatism against optimality. Prior enforcement splits into intrusive-but-density-correct parameterization and pluggable-but-divergence-wrong regularization; the mismatch revealed that a divergence is the wrong object — the support condition is a statement about the *value of the behavior density at the action taken*, so I made the regularizer that density itself, `-lambda log pi_beta(pi_phi(s)|s)`, relaxed the intractable per-state constraint to an average one, and Lagrangianized it into a pluggable penalty whose coefficient `lambda` is the constraint-strength lever. Since I only have samples of `pi_beta`, I estimated `log pi_beta` with a conditional VAE — flexible enough for multimodal behavior, unlike a Gaussian — and used its ELBO `-L_ELBO` as a lower bound on the log-density, taking `L=1` because the constraint doesn't need a tight density and `L=1` is simpler and lower-variance with an analytic KL. I built it on TD3 rather than SAC because TD3's deterministic actor, clipped twin critics, and target smoothing suppress exactly the OOD overestimation I'm fighting, while SAC's stochastic sampling and entropy bonus push toward the support's edge. I added TD3+BC's Q-normalization so the single `lambda` transfers across reward scales. And because `lambda = 0` collapses the whole loss back to plain TD3, offline-to-online fine-tuning is just cooling `lambda` toward a small floor as online data arrives and keeping the VAE frozen so the support notion stays stable, letting the policy improve on real feedback without the early Q-collapse that kills naive fine-tuning.
