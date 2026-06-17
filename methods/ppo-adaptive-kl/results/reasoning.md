Let me start from what actually hurts. I have a stochastic policy π_θ(a|s), a network that takes a state and emits a distribution over continuous actions, and I want to make it better by gradient ascent on expected return. The data I have are rollouts from the *current* policy, and the instant I nudge θ those rollouts describe a policy that no longer exists. Collecting rollouts is the expensive part — every batch is a fresh set of environment interactions — so I badly want to squeeze many gradient steps out of each batch instead of one. But there's a trap I've watched spring: take a batch, run several epochs of plain gradient ascent on the obvious surrogate, and the policy lurches so far that the batch stops describing it, the next rollouts are worse, and it never recovers. Not a graceful loss of accuracy — a collapse. So the whole game is: out of one batch of on-policy samples, manufacture as much *reliable* improvement as I can without ever stepping outside the region where the batch still means something.

What do I do today? The score-function policy gradient. For J(θ) = E_τ[R(τ)] I can write the gradient as something I can sample,

  g = E_t[ ∇_θ log π_θ(a_t|s_t) · Â_t ],

with Â_t an advantage estimate from a learned value baseline — generalized advantage estimation gives me a clean λ-weighted sum of TD residuals δ_t = reward_t + γV(s_{t+1}) − V(s_t), Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}, and λ dials bias against variance. Implementations realize the gradient by differentiating a surrogate L^PG(θ) = E_t[ log π_θ(a_t|s_t) Â_t ], whose gradient at θ_old is exactly g. Fine. But look hard at what L^PG *is*. Its gradient equals the policy gradient only *at the point θ_old where the samples were drawn*. Differentiate it once, step, and you've used the estimator where it's valid. Differentiate it again on the *same* batch at the new θ — which is what I'm tempted to do to get my multiple epochs — and there is nothing in L^PG that knows the samples came from π_old. The log π_θ term just keeps climbing; multiple epochs on L^PG drive the policy wherever raises log-prob of the sampled actions weighted by their advantage, with no leash. That's the collapse, and now I can see it isn't bad luck — L^PG simply has no representation of "you have left the data's neighborhood." Wall, but an informative one: whatever I build has to make multi-epoch reuse of a fixed batch *legitimate*, and to do that it has to be an objective that knows the batch is from π_old.

So how do I write down an objective whose value, evaluated at a new θ, is still an honest estimate of how good π_θ is, using only π_old samples? That's just importance sampling. The expected advantage under π_θ, estimated from π_old rollouts, re-weights each sample by the likelihood ratio,

  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t),   L^CPI(θ) = E_t[ r_t(θ) Â_t ].

This is the conservative-policy-iteration surrogate of Kakade & Langford. At θ = θ_old every r_t = 1 and L^CPI = E_t[Â_t]; its gradient there is again the policy gradient, so it agrees with L^PG to first order — but unlike L^PG it carries the ratio, so it actually *is* a (one-sample) off-policy estimate of π_θ's advantage. Now I can take many gradient steps on it from one batch, because the ratio keeps re-weighting toward the policy I'm currently at. So why isn't this the answer? Because I can maximize it too well. Suppose a sample has Â_t > 0. Then r_t Â_t is increasing in r_t without bound — the optimizer will happily push π_θ(a_t|s_t) up until r_t is enormous, far past where π_old's sample was a faithful proxy for π_θ's behavior. Where Â_t < 0 it shoves r_t toward 0. Either way it sprints out of the region where importance sampling on a finite batch means anything; a handful of high-advantage samples get their ratios blown up and the estimate, and the policy, detonate. Same collapse, now I see *why*: the surrogate is only a faithful proxy while π_θ stays close to π_old, and nothing stops it from leaving.

What does "close" even mean for two policies? Not Euclidean distance in θ — two very different parameter vectors can give nearly the same action distribution and vice versa. The thing that actually governs whether π_old's samples still describe π_θ is how different the two *distributions* are, and the natural measure of that is the KL divergence between the action distributions, averaged over the states I visit, E_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ]. That's the leash I want — not on parameters, on the distribution.

This is exactly the trust-region idea: maximize the surrogate but forbid the policy from moving more than δ in KL,

  maximize_θ E_t[ r_t(θ) Â_t ]   subject to   E_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ] ≤ δ.

And there's real theory under it — a KL-penalized version of the surrogate lower-bounds the true return, so improving the surrogate while keeping KL small guarantees monotonic improvement of the actual policy, modulo the bound's slack. So the constrained problem is principled. But how is it solved in practice? Linearize the objective, take a quadratic (Fisher-information) approximation to the KL constraint, and solve the resulting trust-region quadratic program with conjugate gradient plus a line search. And there's the rub for me. That's second order: I need Fisher-vector products, an inner conjugate-gradient loop every update, a careful line search. It's heavy to implement, expensive, and — this is the part that really bites — it does not coexist with the architectures I care about. If I share parameters between the policy and the value function, or use dropout, or hang an auxiliary loss off the network, the clean Fisher-matrix story breaks. And it doesn't naturally give me what I came for: multiple epochs of plain SGD on the batch. The method that best controls the update size is precisely the one that fails my scalability and simplicity goals. Wall.

So I want the trust region's *effect* — keep π_θ close to π_old, multi-epoch reuse stays honest, no collapse — with nothing but first-order SGD on a differentiable objective. Two ways to attack that, and I'll chase both.

The first: the theory already pointed at a penalty rather than a constraint. Drop the hard constraint and fold KL into the objective,

  L^KLPEN(θ) = E_t[ r_t(θ) Â_t − β KL[π_old(·|s_t), π_θ(·|s_t)] ].

Now it's a single unconstrained objective I can hand to Adam for K epochs. The penalty fights the surrogate's urge to blow r_t up: every bit of KL the policy spends costs β per unit, so the optimizer settles where the marginal advantage gain equals the marginal KL cost. Clean. But what is β? Here's the thing that killed the fixed-penalty idea before and kills it again: there is no single β that holds the policy at a sensible step size. Pick β too small and the KL still runs away to a destructive update; pick it too large and the policy barely moves and learning crawls. Worse, the *right* β isn't even constant within one run. Early in training the policy is far from optimal and a given β lets it move a lot in KL; late in training, near convergence, the same β barely moves it. The amount a fixed objective-penalty translates into actual distributional movement drifts over training. So a fixed β is the wrong size at some point in essentially every run. That's why the trust-region method used a hard constraint instead of a penalty in the first place — it couldn't pick β either.

But I don't have to *pick* β and freeze it. I have a target I actually care about — a desired per-update KL, call it d_targ — and after each update I can *measure* the KL the update produced and compare it to the target. That converts "choose the right β" into a control problem: if the realized KL came out too big, the penalty was too weak, so raise β; if it came out too small, the penalty was too strong, so lower β. Concretely, after a policy update compute d = E_t[ KL[π_old, π_θ] ] and adjust

  if d > d_targ · 1.5:   β ← β · 2        (overshot the trust region → penalize harder next time)
  if d < d_targ / 1.5:   β ← β / 2        (barely moved → loosen the penalty next time)

with a dead band between d_targ/1.5 and d_targ·1.5 so noise in the KL estimate doesn't make β chatter. The multiplicative form is deliberate — β must stay positive and I want a symmetric proportional response on a log scale, not an additive nudge that could drive β negative or react sluggishly when β is tiny. The factors 1.5 (the band) and 2 (the adjustment) are heuristic, and they don't need to be precise: the loop is self-correcting, so as long as it pushes β the right direction with a sane step, β converges toward whatever value yields KL ≈ d_targ for the *current* phase of training, and re-converges as that phase shifts. The initial β stops mattering — the controller walks it to the right place within a few updates. So the drifting-β problem dissolves: I never picked the right β, I built a feedback loop that finds it continuously. This is the adaptive-KL-penalty method, and it's a genuine first-order, plain-SGD, K-epochs-per-batch method with a trust-region *effect*.

Now the second route, because I'm not sure the penalty is the cleanest way to get the effect, and chasing it teaches me something. The penalty term needs the KL computed and differentiated through the network every step, plus the β bookkeeping. What if I bake the leash directly into the surrogate's *shape* so the objective itself stops rewarding the policy for leaving the neighborhood, with no separate penalty term at all? Stare at where L^CPI = E_t[r_t Â_t] goes wrong: for Â_t > 0 it keeps paying out as r_t climbs past 1; for Â_t < 0 it keeps paying as r_t drops below 1. The pathology is the *unbounded incentive to move r_t away from 1*. So clip the ratio inside the objective: don't let the surrogate give any extra credit once r_t leaves a band [1−ε, 1+ε]. Define

  L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ].

Let me check this term by term, because the min and the clip have to combine to do the right thing in both advantage signs, and it's easy to get backwards. Take Â_t > 0. The unclipped term r_t Â_t rises with r_t. If r_t < 1−ε, the clipped term is (1−ε)Â_t, which is larger than r_t Â_t, so the min keeps the unclipped term: lowering the probability of a good action is still fully penalized. If 1−ε ≤ r_t ≤ 1+ε, clip does nothing and the two terms agree. If r_t > 1+ε, the clipped term (1+ε)Â_t is smaller, so the objective flattens — pushing r_t past 1+ε buys nothing. Good: a positive-advantage action gets reinforced, but only up to a ratio of 1+ε, then the gradient through that sample dies. Now Â_t < 0. Unclipped r_t Â_t *decreases* as r_t grows and *increases* (toward 0, less negative) as r_t shrinks; the optimizer wants r_t small. Clipped term: clip floors r_t at 1−ε, so for r_t < 1−ε the clipped term is (1−ε)Â_t, a constant. The min of r_t Â_t and (1−ε)Â_t when Â_t<0: r_t Â_t is more negative than (1−ε)Â_t exactly when r_t > 1−ε... let me just plug numbers, Â_t = −1, ε = 0.2. At r_t = 0.5: unclipped = −0.5, clipped = clip(0.5,0.8,1.2)·(−1) = 0.8·(−1) = −0.8, min(−0.5,−0.8) = −0.8. At r_t = 0.9 (in band): unclipped = −0.9, clipped = 0.9·(−1) = −0.9, min = −0.9. At r_t = 1.5: unclipped = −1.5, clipped = 1.2·(−1) = −1.2, min(−1.5,−1.2) = −1.5. So for r_t < 1−ε the objective is pinned at (1−ε)Â_t = −0.8 (flat — no further reward for driving the action's probability down past 1−ε), and once r_t is back above the lower edge the unclipped r_t Â_t is active again, equal to the clipped term inside the band and worse than the clipped term above the upper edge. That's the right behavior for a bad action: push its probability down, but only to 1−ε, then stop; if the update instead raises the probability of a bad action, the objective keeps the full penalty rather than hiding it behind the upper clip.

The min is doing something subtle I want to name precisely. Where the clip would cap an improvement, the min takes the clipped value only when that value is smaller, so clipping removes credit and never adds it. Where the policy moves in the bad direction — lowering the probability of a good action or raising the probability of a bad one — the unclipped, worse term stays active. Per sample, L^CLIP never exceeds the unclipped r_t Â_t; it agrees with it to first order at r=1 and departs only in the directions where the raw surrogate would over-reward leaving the neighborhood. That's the whole leash, encoded in the objective's geometry: no KL term, no β, no second-order machinery, just a min and a clip, piecewise-smooth and easy for autodiff to handle over repeated minibatch epochs. And ε is the trust-region size — a single, interpretable knob; ε = 0.2 is the scale I will use for continuous control rather than making the band so tight that learning crawls or so loose that the policy can drift too far.

So now I have two first-order methods that both deliver the trust-region effect: the adaptive-KL penalty, and the clipped surrogate. The clipped one is the cleaner main objective because the leash is inside the sample-wise geometry instead of in a separately tuned penalty term. But the penalty/KL-control idea is not wasted — it gives me a *second* knob I'll come back to, because there's a piece of the picture I've been ignoring: even with the clipped objective bounding each *per-sample* contribution, the realized KL of a whole update still depends on the optimizer's stepsize and the number of epochs, and on a hard problem with a high learning rate the policy can still move more than I'd like across K epochs. The clip caps the *incentive* to move past the band on each sample, but it doesn't directly fix the *total distance* the optimizer travels. So I still want a handle on the overall update size, and the KL-feedback machinery I just built for β is exactly the handle — I just point it at a different actuator.

With the clipped objective already supplying the in-objective leash, the remaining free trust-region knob is the Adam stepsize itself. The thing that physically determines how far Adam walks the policy is the stepsize, so I can use the same measured-KL feedback on that actuator: inside the minibatch update, measure the KL between the policy that collected the batch and the current policy; if the realized KL overshot the target, the steps were too big, shrink the learning rate; if it undershot, grow it. For the stepsize controller I use a wider dead band than the β rule, matching the implementation:

  d = E_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ]
  if d > d_targ · 2.0:   lr ← max(lr_min, lr / 1.5)
  if 0 < d < d_targ / 2.0:   lr ← min(lr_max, lr · 1.5)

The asymmetry from the β version is gone because lr and the realized KL move the *same* direction (bigger lr → bigger KL), whereas β moved the *opposite* direction (bigger β → smaller KL); so where β doubled on overshoot, lr instead *divides* on overshoot. Let me sanity-check the direction: KL too big means I moved too far, so I want smaller steps next time, divide lr. KL too small means I'm leaving improvement on the table, multiply lr. The factor 1.5 is a damped multiplicative step so the controller doesn't overshoot and oscillate; the dead band [d_targ/2, d_targ·2] keeps noise from chattering the rate. And I clamp lr to [lr_min, lr_max] = [1e-5, 1e-2]: a rate driven to zero would freeze learning, a rate driven too high would diverge before the next KL measurement could pull it back. A target d_targ ≈ 0.01 is a small per-update distributional move, the right order of magnitude for a trust-region leash rather than a parameter-space leash.

I should make sure I can actually *compute* that KL cheaply, because if it needs extra sampling the whole controller gets expensive. The policy is a diagonal Gaussian: π_old = N(μ_old, σ_old²) and π_θ = N(μ_new, σ_new²) per action dimension, independent across dimensions. The KL between two univariate Gaussians has a closed form; let me derive it so I trust the code. For N(μ_0,σ_0²) ‖ N(μ_1,σ_1²),

  KL = E_{x∼N_0}[ log p_0(x) − log p_1(x) ].

Write log p_i(x) = −½log(2πσ_i²) − (x−μ_i)²/(2σ_i²). The difference is
  log p_0 − log p_1 = log(σ_1/σ_0) − (x−μ_0)²/(2σ_0²) + (x−μ_1)²/(2σ_1²).
Take the expectation under x∼N_0, using E[(x−μ_0)²] = σ_0² and E[(x−μ_1)²] = σ_0² + (μ_0−μ_1)²:
  KL = log(σ_1/σ_0) − σ_0²/(2σ_0²) + (σ_0² + (μ_0−μ_1)²)/(2σ_1²)
     = log(σ_1/σ_0) + (σ_0² + (μ_0−μ_1)²)/(2σ_1²) − ½.
Summed over the independent action dimensions, with old = the policy that collected the batch (subscript 0) and new = the current policy (subscript 1),

  KL = Σ_i [ log(σ_new,i/σ_old,i) + (σ_old,i² + (μ_old,i − μ_new,i)²) / (2 σ_new,i²) − ½ ].

Zero variance, no sampling, just the stored (μ_old, σ_old) and the recomputed (μ_new, σ_new). The formula itself is exact; in code I can add a tiny ε inside the log argument as a numerical guard. Good — the controller costs almost nothing.

Now the rest of the loss, because the network does more than the policy. I fit a value function for the GAE baseline, and I want to train it on the same batch. Add a squared-error value loss L^VF = (V_θ(s) − V_t^targ)², where the target is the GAE return. One wrinkle: over K epochs the critic can over-fit a transient target, so I mirror the policy's trust region on the value too — clip the value *change* to within ±ε of the old value before squaring, and take the max of clipped and unclipped squared errors, so a single epoch can't yank the value estimate too far. And to keep exploration alive I can add an entropy bonus S[π_θ] with a small coefficient (for the continuous-control policy with a learned std I often set it to zero; it matters more for the shared-network discrete case). The combined per-iteration objective is

  L = L^CLIP − c1 L^VF + c2 S[π_θ]  (maximized),

equivalently minimize  −L^CLIP + c1 L^VF − c2 S. I also normalize the advantages to zero mean and unit std over the batch before forming the surrogate, so ε and the learning rate have a consistent meaning across batches regardless of the reward scale, and I clip the global gradient norm to 1.0 as a cheap guard against the rare exploding gradient.

The outer structure is the parallel-actor harness I already have: each iteration, run the current policy for T steps in each of N environments, compute truncated-GAE advantages and returns, then optimize L with K epochs of minibatch SGD (Adam) over the NT collected transitions, and only then set π_old ← π_θ. The clipped objective and KL controller are the only new pieces, and both live inside the per-minibatch update.

Let me put the whole reasoning into the code I'd actually run, filling the empty slot in the optimizer harness — the minibatch update rule with the KL-driven learning-rate controller.

```python
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyOptimizer:
    """First-order on-policy update: clipped surrogate + clipped value loss + entropy,
    with a learning rate driven by the realized KL toward a target trust-region size."""

    def __init__(self, actor_critic, num_learning_epochs=5, num_mini_batches=4,
                 clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0,
                 entropy_coef=0.01, learning_rate=1e-3, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
                 device="cpu"):
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.device = device
        # trust-region controller
        self.desired_kl = desired_kl          # target per-update KL (d_targ ~ 0.01)
        self.schedule = schedule              # "adaptive" turns the KL->lr controller on
        self.learning_rate = learning_rate
        # surrogate / loss knobs
        self.clip_param = clip_param          # epsilon: in-objective trust-region band
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma, self.lam = gamma, lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches,
                                                      self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, actions_batch, old_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch) in generator:

            # re-evaluate the CURRENT policy distribution on the stored states/actions
            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # --- KL-driven learning-rate controller ---
            # measure how far this update moved the diagonal-Gaussian policy, in closed form,
            # then nudge the Adam stepsize toward the target trust-region size desired_kl.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch)
                           + torch.square(old_mu_batch - mu_batch))
                          / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )                                   # analytic KL(old || new) per sample
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:         # moved too far -> smaller steps
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:  # room -> bigger
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # --- clipped surrogate (the in-objective trust region) ---
            ratio = torch.exp(actions_log_prob_batch
                              - torch.squeeze(old_actions_log_prob_batch))   # r_t = pi_new/pi_old
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()   # min(.) on +L

            # --- clipped value loss (mirror the trust region on the critic) ---
            if self.use_clipped_value_loss:
                value_clipped = old_values_batch + (value_batch - old_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates
```

The causal chain, start to finish: I was stuck because the plain policy-gradient surrogate is valid only at the sampling point, so reusing a batch for several epochs drives the policy out of the region the batch describes and collapses it. Importance-weighting the surrogate by the likelihood ratio r_t made it an off-policy estimate I could reuse, but with nothing to stop r_t from blowing up it collapsed just the same — the surrogate is only faithful while the new policy stays close to the old in *distribution*, measured by KL. The principled fix, a hard KL-constrained trust region, controls the move but is second-order, hard to scale, and incompatible with parameter sharing and multi-epoch SGD. So I sought the trust-region *effect* with first-order SGD two ways: a KL penalty whose coefficient I cannot fix but *can* drive with a feedback loop toward a target KL; and a clipped surrogate that bakes the leash into the objective's shape — the min-of-clipped-and-unclipped makes it a pessimistic lower bound that gives zero credit for pushing r_t past 1±ε, in either advantage sign, with no penalty term or second-order machinery. The clip handles the per-sample incentive; the overall update distance over K epochs is regulated by reusing the same KL-feedback controller on the natural actuator — the Adam learning rate — shrinking it by 1.5 when the realized (closed-form Gaussian) KL overshoots the target band and growing it when it undershoots, clamped to a sane range, so the trust-region size stays steady across problems and across a run without ever hand-tuning a fixed knob. Wrapped around a value loss (clipped the same way), advantage normalization, an entropy bonus, and gradient-norm clipping, the whole thing is a handful of lines over a vanilla policy-gradient implementation, first-order, K epochs per batch, dropping straight into the parallel-actor harness.
