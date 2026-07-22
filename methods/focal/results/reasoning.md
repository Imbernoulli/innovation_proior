Let me start from what actually makes this hard, because the constraint is unusually severe and I want to feel its weight before I reach for any machinery. I have a distribution of tasks — each one an MDP, sharing the same state and action spaces but with its own transition function and its own reward function — and for every task all I get is a fixed, pre-collected pile of transitions logged by some behavior policy I don't even have access to. I never touch the environment. Not while meta-training, and crucially not at test time either: when a new, unseen task arrives, I'm handed a small batch of its logged transitions and I have to adapt and act well from those alone. So whatever I build has to do two things that fight each other. It has to learn a control policy from static data without the value functions exploding, and it has to figure out *which task it's in* from a few transitions, with no ability to poke the environment to disambiguate.

Take the offline half first, because it has a failure mode that will wreck everything else if I ignore it. Value-based RL bootstraps: I regress `Q(s,a)` toward `r + γ Q(s', a')` where `a'` comes from my current learned policy. When I can collect data, an over-optimistic `Q` at some weird `(s', a')` gets corrected the next time I actually try that action and see the real return. Offline, there is no next time. If my policy improvement step asks `Q` about an action far outside what the behavior policy ever did, the network — which generalizes badly off the data manifold — hands back some inflated number, and that inflated number flows straight back into the next backup, and the next, and there's no feedback anywhere to pull it down. This isn't a small bias; people have watched offline `Q` values run off to `10^11`. So an offline learner *has* to be tethered to the behavior policy's support. I can't let the policy wander to actions the data never covered.

How do I tether it? The cleanest framing I know is to add a divergence between my learner `π_θ` and the behavior policy `π_b` directly into the actor-critic objective — penalize the policy for straying off the data distribution. Wu, Tucker and Nachum's behavior-regularized actor-critic gives me the two places to put it. One option is a value penalty: bake the divergence into the value function itself,
```
V^D_π(s) = Σ_t γ^t E[ R_π(s_t) − α D(π_θ(·|s_t), π_b(·|s_t)) ],
```
so the target Q used in the critic regression carries a `−γ α D̂` term and the policy is discouraged, through the values, from leaving the support. The other option is to drop the penalty from the value update (`α = 0` there) and only regularize the actor. Either way the loss pair is
```
L_critic = E[ ( r + γ Q̄^D(s', a') − Q(s, a) )² ],
L_actor  = − E[ E_{a''∼π_θ}[ Q(s, a'') ] − α D̂ ],
```
with `Q̄` a target network without gradients. That gives me a skeleton for stable offline control, and reassuringly BCQ and BEAR both fall out of it as special cases, so I'm not inventing a one-off.

The thing I have to settle is what `D` is and how I estimate it, because I do not have `π_b` — only samples from it, the logged actions. If I pick a KL divergence `D_KL(π_θ ‖ π_b) = E_{a∼π_θ}[ log π_θ(a) − log π_b(a) ]`, I'd need the behavior density `π_b(a|s)`, which means fitting a "cloned policy" by max-likelihood on the data and then evaluating it — an extra generative model that can be badly miscalibrated. I'd rather not. There's a way around it: any f-divergence has a dual (Fenchel) form,
```
D_f(p, q) = max_g  E_{x∼p}[ g(x) ] − E_{x∼q}[ f*(g(x)) ],
```
where `f*` is the convex conjugate of `f`. For KL, `f(x) = −log x`, and I should pin down `f*` rather than copy it from memory. By definition `f*(t) = sup_{x>0} [ t x − f(x) ] = sup_{x>0} [ t x + log x ]`. The derivative `t + 1/x = 0` gives an interior maximizer `x* = −1/t`, which only exists for `t < 0`; substituting back, `f*(t) = t(−1/t) + log(−1/t) = −1 − log(−t)`. So `f*(t) = −log(−t) − 1`. (Quick numeric sanity: at `t = −0.5` the maximizer is `x* = 2`, the objective there is `−0.5·2 + log 2 = −0.30685`, and `−log(0.5) − 1 = −0.30685` — they agree.) Good — so instead of estimating a density I learn a discriminator `g` by minimax, where `g` plays the role of the log-density-ratio, and I get a sample-based divergence estimate with no cloned policy at all. I'll keep `g` honest with a gradient penalty so the minimax doesn't run away. That settles the control side: a SAC-style twin-Q actor-critic with a dual-form KL behavior penalty. I'll even keep the maximum-entropy term, which feels odd offline since I'm not exploring — but in something like Ant different actions can produce the same next state and reward, and the entropy bonus stops the policy from collapsing onto one of several equivalent actions, so it's worth keeping. There is a real cost here I should flag to myself: this whole family is sensitive to the reward scale and to the regularization strength `α`. I'll be tuning `α` per environment, and the spread is going to be wide.

Now the hard half — task inference — and this is where the offline constraint really bites, because it kills the standard approach. The strongest off-policy meta-RL method I know, PEARL, infers the task as a latent variable `z`. Its encoder `q_φ(z|c)` reads the context `c` (a set of transitions) and produces a posterior over `z`; the policy and value functions condition on `z`. Three design choices make it work on-policy. The encoder is probabilistic and permutation-invariant: each transition `c_n` produces a Gaussian factor `Ψ_φ(z|c_n) = N(μ_n, σ_n)`, and they're fused by a product of Gaussians,
```
q_φ(z|c_{1:N}) ∝ Π_n Ψ_φ(z|c_n),    σ² = 1 / Σ_n σ_n^{−2},   μ = σ² Σ_n μ_n σ_n^{−2},
```
which is order-invariant by construction. There's a KL information bottleneck, `+ β KL(q_φ(z|c) ‖ N(0,I))`, that squeezes `z` down to the minimal task-relevant information and hands me a prior to sample from. And that probabilistic posterior is the engine of posterior-sampling exploration: at test time PEARL samples `z` from the prior, acts for an episode, updates the posterior from what it saw, and acts again as the belief narrows — temporally-extended exploration of an unknown task. The encoder itself is trained by the Bellman gradients of the critic.

Stare at that for a second in my setting. The information bottleneck and the whole probabilistic posterior exist *to support exploration* — to represent and then narrow uncertainty by acting. But I never act to disambiguate. At test time I get a fixed batch of logged transitions and that's it; there is no episode-by-episode belief-narrowing because there's no interaction. So a large chunk of PEARL's design is machinery for a thing I'm forbidden from doing. That's not just wasted — modeling uncertainty I'll never resolve by exploration is modeling noise. And the encoder being trained *only* by Bellman gradients worries me here for a second reason: I just argued the offline Bellman backups can become ill-behaved, even divergent. If my task representation learns exclusively through that signal, then when the value learning goes unstable, the embedding could go unstable with it, and the tasks would stop being distinguishable. The task inference would be hostage to the very instability I'm fighting on the control side. So porting PEARL to offline — same encoder, same bottleneck, just no exploration — looks like it would leave me with a representation that learns slowly through a fragile channel and carries uncertainty structure I can't use. That pushes me to rethink task inference from the ground up, for *this* setting, rather than port.

Let me go back to first principles and ask what the latent variable even *is* here. The tasks differ only in `P` and `R`. Suppose I restrict to deterministic MDPs — `P(s'|s,a) = δ(s' − t(s,a))` for a transition map `t` — which is exactly what the standard continuous-control benchmarks are. And suppose the tasks satisfy what I'll call task-transition correspondence: for any two tasks and any `(s,a)`,
```
P_1(·|s,a) = P_2(·|s,a)  and  R_1(s,a) = R_2(s,a)   ⟺   T_1 = T_2.
```
In words: the pair `(P, R)` *identifies* the task — no two distinct tasks agree on transition and reward everywhere. Under determinism this means: given `(s,a)` and the task, there's a unique outcome `(s', r)`, so each task defines a map `f_T(s,a) = (s', r)`. The whole family `{f_T}` lives on the space of transitions `S × A × S × ℝ` — which is precisely the context space. So under this assumption the encoder isn't inferring some fuzzy belief; it's embedding the function `f_T`.

And here's what that buys me, as long as I keep the strength of the assumption in view. Because `(P, R)` identify the task pointwise, a transition tuple `(s, a, s', r)` can in principle pin down which task I'm in; if two tasks gave the same outcome for the same `(s,a)`, the assumption would say they are the same task. I don't need a belief state that narrows only through future exploration; the logged context is meant to carry task identity directly. Two consequences drop out. The encoder should be permutation-invariant — order can't matter if transitions are just samples from the task-specific map. And it should be *deterministic* — in this regime the task is treated as recoverable from context, not as uncertainty to be explored away. The probabilistic posterior, the product of Gaussians, the information bottleneck — all of that was answering a question ("how uncertain am I, and how do I explore to reduce it?") that doesn't arise once I'm in the deterministic, correspondence-satisfying, no-exploration regime. So: a deterministic encoder. Each transition gets embedded, and I aggregate the per-transition embeddings into one `z`. The natural permutation-invariant aggregation is just the mean — which is also exactly what the metric-learning few-shot people do when they form a class prototype as the mean of embedded supports, and they have a principled reason (the mean is the Bregman representative of a cluster under squared Euclidean distance). I'll take the mean over the transition embeddings.

But "deterministic encoder, mean-aggregated" only says how I *combine* embeddings; it says nothing about what makes the embedding *good*. If I just let the encoder learn through Bellman gradients like PEARL, I'm back in the trap. I want a *direct* signal that shapes the geometry of `z` so that same-task transitions land together and different-task transitions land apart — decoupled from the value learning entirely. So the question becomes: what objective do I put on the embedding space itself?

Before I pick one, let me make sure direct separation is genuinely *necessary* and not just aesthetically nice, because if it isn't I shouldn't add a whole objective for it. The value function `Q_θ(s, a, z)` is a continuous neural network. Continuity says: for every `ε > 0` there's an `η > 0` such that `|z_1 − z_2| < η ⟹ |Q̂_θ(s, a, z_1) − Q̂_θ(s, a, z_2)| < ε`. Now take two *different* tasks whose embeddings happen to be close, `|z_1 − z_2| < η`. Then the network is *forced* to give them nearly equal Q-values, `|Q̂_θ(s,a,z_1) − Q̂_θ(s,a,z_2)| < ε`. But the two tasks have different transition and reward functions, so their *true* Q-values, `Q(s,a,z_1)` and `Q(s,a,z_2)` via `R_z(s,a) + γ E_{s'∼P_z}[V(s')]`, need not be close — they can differ a lot. A single continuous network simply cannot output two well-separated true values at two nearly-identical inputs. So if the encoder lets distinct tasks' embeddings sit close together, the conditioned value functions for those tasks become unrepresentable — the approximator can't fit both. I think that's the real reason the geometry of `z` matters: keeping distinct tasks' embeddings apart isn't cosmetic, it's a precondition for the conditioned value functions to even exist. So I do need an objective that enforces separation.

So I want an objective on `z` that clusters same-task and separates different-task. This is plain distance metric learning, and the canonical loss is the contrastive loss,
```
L_cont(x_i, x_j) = 1{y_i = y_j} ‖q_i − q_j‖₂²  +  1{y_i ≠ y_j} max(0, m − ‖q_i − q_j‖₂)²,
```
where `q = q_φ(x)` and `y` is the task label. Same-task pairs feel an attractive quadratic pulling them together; different-task pairs feel a margin-`m` hinge pushing them apart up to distance `m`. Let me just try this and see if it works. I embed transitions, label pairs by task identity, descend this loss.

It degenerates. The clusters don't cleanly separate — I get blobs that contain transitions from *several* tasks mixed together. Let me figure out *why* instead of just observing it, because the why is going to tell me the fix. Two things look wrong, and they compound. First, look at the repulsive term. Hadsell's spring picture is honest about it: `max(0, m − ‖q_i − q_j‖)²` is a spring that acts *only within radius `m`*. Once a different-task pair is already farther than `m` apart it contributes exactly zero gradient. At the other end, where my tanh-bounded latent space and random initialization can put many embeddings close to the origin, the hinge force is still only bounded — its magnitude scales like `(m − d)` rather than blowing up as `d → 0`. So it can push, but it has no singular urgency when distinct tasks are nearly on top of each other; it is a capped spring, not a short-range repulsive potential.

The second thing is deeper, and it's about the attractive/quadratic part — really about using *any* positive power of distance as the spread objective. Let me compute what minimizing a sum of squared distances actually does, on a single dataset `X = {x_i}`. The accumulative squared-distance objective is
```
Σ_{i≠j} (x_i − x_j)².
```
Expand it: each unordered pair appears twice, and
```
Σ_{i≠j} (x_i − x_j)² = Σ_{i≠j} (x_i² − 2 x_i x_j + x_j²)
                     = 2 (N−1) Σ_i x_i²  −  2 Σ_{i≠j} x_i x_j
                     = 2 ( (N−1) Σ_i x_i²  −  Σ_{i≠j} x_i x_j ).
```
For the first equality I used that summing `x_i²` over all `j ≠ i` for each fixed `i` gives `(N−1) x_i²`, and likewise for `x_j²`, so together `2(N−1)Σ_i x_i²`. Now the variance of `X`:
```
Var(X) = (1/N) Σ_i x_i²  −  ( (1/N) Σ_i x_i )²
       = (1/N) Σ_i x_i²  −  (1/N²) ( Σ_i x_i )².
```
And `(Σ_i x_i)² = Σ_i x_i² + Σ_{i≠j} x_i x_j`, so
```
Var(X) = (1/N) Σ_i x_i²  −  (1/N²)( Σ_i x_i² + Σ_{i≠j} x_i x_j )
       = (1/N²)( N Σ_i x_i² − Σ_i x_i² − Σ_{i≠j} x_i x_j )
       = (1/N²)( (N−1) Σ_i x_i² − Σ_{i≠j} x_i x_j ).
```
That's the identical bracket. So
```
Σ_{i≠j} (x_i − x_j)² = 2 N² Var(X).
```
The accumulative squared-distance loss *is* the dataset variance, up to a constant. If that's right, then a squared-distance metric objective is just variance maximization — and variance is a *global* scalar. Many embedding distributions share the same variance, including degenerate ones, so maximizing it need not separate every pair of distinct tasks. That's a claim I can actually test rather than wave at, and I want to, because the whole choice of loss hinges on it.

Let me build the smallest example that could expose the failure, in 1-D. Put four task embeddings on the interval `[−1, 1]` — one point per task, so *every* pair is a different-task pair that the repulsive term should be separating. Compare two configurations. Configuration A piles two tasks at `−1` and two tasks at `+1`: `X_A = (−1, −1, +1, +1)`. This is the Bernoulli-type degenerate I'm worried about — half the mass at each extreme, distinct tasks *merged* into the two piles. Configuration B spreads all four out evenly: `X_B = (−1, −1/3, +1/3, +1)`, every task distinct and resolved.

First check the identity itself on these. `Var(X_A) = 1` (mean 0, every point at distance 1), so `2N²Var = 2·16·1 = 32`; computing `Σ_{i≠j}(x_i−x_j)²` directly over the 12 ordered pairs also gives `32`. For B, `Var(X_B) = (1 + 1/9 + 1/9 + 1)/4 = 0.5556`, so `2N²Var = 32·0.5556 = 17.78`, and the direct sum is `17.78` as well. So the identity holds numerically in both cases — good, the algebra was right.

Now the part that matters. The squared-distance *spread* I'd be maximizing is, over unordered pairs, `16` for A and `8.89` for B. So the squared objective rates the **merged** configuration A as *better* than the resolved configuration B — it actively prefers piling distinct tasks at the extremes, because that's what maximizes variance. The RMS pairwise distance tells the same misleading story: `1.633` for A versus `1.217` for B, again ranking the merged config higher. That is exactly the degenerate blob I saw in training, now reproduced on four points: a global spread statistic is happily maximized by a distribution that crams several tasks into each of two piles. So the diagnosis is confirmed, and confirmed in the strong form — it's not that squared-distance is merely *indifferent* to the merge, it positively rewards it.

So what do I actually need? A repulsive term between different-task pairs whose penalty is largest at short range and fades as the distance grows — one that *sees* the merged pair instead of averaging it away. The unregularized function with that profile is an inverse power of distance, `1 / dᵏ`; in practice I need an `ε` floor, so the penalty is finite at zero but still focused on close pairs. Replace the positive-power repulsion with a *negative* power:
```
L_dml(x_i, x_j) = 1{y_i = y_j} ‖q_i − q_j‖₂²  +  1{y_i ≠ y_j} · β / ( ‖q_i − q_j‖₂ⁿ + ε ),
```
with `ε > 0` to keep the denominator off zero and `n ≥ 0` the power. I have to handle the cases carefully: `n = 0` gives the constant `β/(1+ε)` for every different-task pair, so it has no separation gradient and is not a useful repulsion. For `n > 0`, the different-task term is a short-range repulsive penalty, not a margin spring. Because the penalty is applied per pair, a pair of distinct tasks sitting in the same pile is directly penalized even if the global spread is already large.

Let me run the same four-point test against this loss to make sure it actually fixes the ranking, since that's the whole point. Take `n = 2` (inverse-square) with `ε = 0.1`, and compute the repulsive loss `Σ_{i<j} 1/(d² + ε)` — now a quantity to *minimize*. For configuration A, two of the six pairs are the merged pairs at distance 0, and each contributes `1/(0 + 0.1) = 10`; adding the rest gives a total of `20.98`. For the spread configuration B the total is `6.82`. So the inverse-square loss ranks B *below* A — i.e. it prefers the resolved configuration, the exact opposite of the squared objective's verdict. And the mechanism is visible in the arithmetic: the two zero-distance merged pairs in A pour `1/ε = 10` each into the loss, which is precisely what the squared spread threw away (a distance-0 pair contributes `0` to `Σd²`). The short-range penalty is doing the job the variance objective structurally cannot. That settles which direction the power has to go: negative, not positive.

It also fits a physical picture, which makes me trust it more. Treat each different-task embedding as a point charge of the same sign sitting in the bounded cube. The `n = 1` case is the Coulomb analogy, with the caveat that `ε` caps the potential at zero distance. A system of like charges confined to a box tends to migrate to the boundary, concentrating at edges and corners. So I'd *expect* an encoder trained with this loss to push task embeddings toward the faces, edges, and vertices of `(−1,1)^l` — that's a prediction I'd want to confirm on the actual t-SNE/PCA embeddings rather than assert, but it's at least consistent with the four-point result, where the resolved low-loss configuration B already sits with its extreme points at the boundary `±1`. The bounded tanh space isn't incidental — it's the box that keeps the repulsion from sending norms to infinity. Same-task transitions, meanwhile, are pulled together by the attractive `‖q_i − q_j‖²` term, so each task should form a tight cluster and the clusters scatter toward the extremes. That is the geometry the continuity argument said I needed: distinct tasks kept far enough apart for the conditioned value functions to be representable.

Which power `n`? The general negative-power family is the design space, and I should think about the cases. `n = 1` is the Coulomb-style inverse potential, capped by `ε`; `n = 2` is sharper away from the floor and matches the inverse-square form used in Cauchy graph embedding, `Σ_{ij} W_{ij} / (‖Y_i − Y_j‖² + σ²)`, which was proposed to preserve local topology better than the Laplacian/quadratic form `Σ_{ij} W_{ij} ‖Y_i − Y_j‖²`. I'll treat `n` as a knob, with `n = 1` (inverse) and `n = 2` (inverse-square) as the two useful cases here; I'd let an ablation pick between them, and I'd be unsurprised if inverse-square wins given its sharper short-range bite, which is the property the four-point test rewarded. The `β` coefficient in the analytic loss sets the relative scale of attractive and repulsive terms, and I'd choose it so the candidate power laws are comparable around half the per-dimension range on `(−1,1)^l`; `ε` floors the denominator. In the implementation, I use a normalized inverse-square form: the positive term is `sqrt(mean(diff²)+ε)` and the negative term is `1/(mean(diff²)+100ε)`, so I should not pretend the runnable loss is a verbatim copy of the symbolic formula.

Now I have to decide how this encoder objective relates to the value-learning gradients, and this is the third design choice. PEARL decoupled task inference from control by *sampling strategy* — different data for the encoder than for the RL batch — because on-policy it needed the encoder data to look like exploration data. I have no exploration, so that particular decoupling is moot; the encoder and the actor-critic can read the same sampled transitions. But there's a different decoupling I *do* think I need, and the continuity-and-divergence analysis tells me why. I argued the offline Bellman backups can grow to enormous magnitude on the harder tasks even when regularized — the value functions can sit at scales like `10⁷`. If I let the encoder receive gradients from the critic as well as from `L_dml`, then the Bellman gradient, scaled by those huge value magnitudes, would likely swamp the distance-metric gradient. The carefully shaped repulsion gets drowned out, the embedding collapses, and I'm back to merged clusters. So I decouple by *gradients*: the encoder is trained *only* by `L_dml`. When the actor and critic consume `z`, they get it *detached* — `z̄`, no gradient flowing back into the encoder. The distance-metric objective is clean, bounded, and scale-controlled; the value objective is the unstable one; keeping them in separate gradient paths protects task inference from the value learning's instability. Three objectives, three optimizers, updated separately:
```
φ ← φ − α₁ ∇_φ Σ_{ij} L_dml(c_i, c_j)        # encoder, distance metric only
θ ← θ − α₂ ∇_θ Σ_i L_actor(b_i, z̄_i)         # actor, z detached
ψ ← ψ − α₃ ∇_ψ Σ_i L_critic(b_i, z̄_i)         # critic, z detached
```
That's the full skeleton: a deterministic mean-aggregating encoder trained by the inverse-power distance metric loss, detached from a behavior-regularized SAC actor-critic that conditions on the inferred task variable.

Let me write the embedding and the loss concretely, the way I'd actually run it. The encoder is a plain MLP from one transition `(s, a, r)` — or `(s, a, r, s')` if next-state helps tell tasks apart — to a `latent_dim` vector. I embed every transition in a task's context batch and take the mean to get that task's `z`. At the analytic level, same-task pairs use squared attraction and different-task pairs use inverse-power repulsion; in the runnable loss below I use the normalized inverse-square variant.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OfflineMetaAgent(nn.Module):
    """Repository-shaped context agent: IB disabled, tanh encoder, mean z."""

    def __init__(self, latent_dim, context_encoder, policy, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = context_encoder
        self.policy = policy
        self.use_ib = kwargs.get('use_information_bottleneck', False)
        self.recurrent = kwargs.get('recurrent', False)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

    def clear_z(self, num_tasks=1):
        self.z_means = self.z.new_zeros(num_tasks, self.latent_dim)
        self.z_vars = self.z.new_zeros(num_tasks, self.latent_dim)
        self.z = self.z_means

    def detach_z(self):
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def infer_posterior(self, context, task_indices=None):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if self.use_ib:
            raise NotImplementedError("This construction disables PEARL's IB path.")
        self.z_means = torch.mean(params, dim=1)
        self.z_vars = torch.std(params, dim=1)
        self.z = self.z_means

    def forward(self, obs, context, task_indices=None):
        self.infer_posterior(context, task_indices=task_indices)
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in self.z], dim=0)
        in_ = torch.cat([obs_flat, task_z], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)
        task_z_vars = torch.cat([z.repeat(b, 1) for z in self.z_vars], dim=0)
        return policy_outputs, task_z, task_z_vars
```

And the distance-metric loss — the heart. The symbolic objective uses squared Euclidean attraction for same-task pairs and inverse-power repulsion for different-task pairs. The implementation uses the inverse-square case in a normalized form: same-task pairs get root-mean-squared attraction, different-task pairs get `1/(mean(diff²)+100ε)`. I compute the per-task embeddings, then loop over task pairs in the meta-batch:

```python
def repo_z_loss(task_z, indices, b, epsilon=1e-3):
    """Repository z_loss on task embeddings.
    task_z is stacked per-task embeddings; indices[i] is the task id of block i.
    Same-task pairs: pull together by normalized RMSE.
    Different-task pairs: inverse-square repulsion with a 100*epsilon floor."""
    pos_loss, neg_loss = 0.0, 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(len(indices)):
        zi = task_z[i * b]
        for j in range(i + 1, len(indices)):
            zj = task_z[j * b]
            d_sq = torch.mean((zi - zj) ** 2)                  # mean squared diff / dim
            if indices[i] == indices[j]:                       # same task -> attract
                pos_loss += torch.sqrt(d_sq + epsilon)
                pos_cnt += 1
            else:                                              # different task -> repel
                neg_loss += 1.0 / (d_sq + epsilon * 100)
                neg_cnt += 1
    return pos_loss / (pos_cnt + epsilon) + neg_loss / (neg_cnt + epsilon)
```

This hard-codes the inverse-square case and uses mean squared distance per latent dimension. The two `ε` floors keep both terms finite. Nothing here touches the value functions; this gradient flows only into the encoder.

Now the offline actor-critic update, conditioned on the *detached* `z`, with the behavior penalty. I learn a dual-form KL discriminator `c` (the BRAC `g`), estimate the divergence from samples, and fold it into the critic target (value penalty) and/or the actor:

```python
class OfflineMetaAlgorithm:
    """Decoupled training: encoder by z_loss only; SAC+BRAC on detached z."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer,
                 qf1, qf2, vf, c_network, divergence, config):
        self.agent = agent
        self.qf1, self.qf2, self.vf = qf1, qf2, vf
        self.target_vf = vf.copy()
        self.train_tasks = train_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.c = c_network
        self.divergence = divergence
        self.batch_size = config.get('batch_size', 256)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)         # SAC/offline sensitivity
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.z_loss_weight = config.get('z_loss_weight', 10.0)      # whole z_loss multiplier
        self._alpha_var = torch.tensor(config.get('alpha_init', 500.0), requires_grad=True)
        self.alpha_max = config.get('alpha_max', 2000.0)
        self.alpha_lr = config.get('alpha_lr', 1.0)
        self.target_divergence = config.get('target_divergence', 0.05)
        self.use_brac = config.get('use_brac', True)
        self.use_value_penalty = config.get('use_value_penalty', False)
        self.max_entropy = config.get('max_entropy', True)
        self.allow_backward_z = config.get('allow_backward_z', False)
        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer  = optim.Adam(agent.policy.parameters(),          lr=lr)
        self.qf1_optimizer     = optim.Adam(qf1.parameters(),                   lr=lr)
        self.qf2_optimizer     = optim.Adam(qf2.parameters(),                   lr=lr)
        self.vf_optimizer      = optim.Adam(vf.parameters(),                    lr=lr)
        self.c_optimizer       = optim.Adam(self.c.parameters(),                lr=1e-4)

    @property
    def get_alpha(self):
        return torch.clamp(self._alpha_var, 0.0, self.alpha_max)

    def _take_step(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        # encode context -> per-task z (gradients to encoder live here)
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_f, act_f, next_f = (x.view(t * b, -1) for x in (obs, actions, next_obs))

        # --- BRAC dual-form KL: train discriminator, estimate divergence ---
        div_estimate = self.divergence.dual_estimate(obs_f, new_actions, act_f, task_z)
        c_loss = self.divergence.dual_critic_loss(obs_f, new_actions, act_f, task_z)
        self.c_optimizer.zero_grad(); c_loss.backward(retain_graph=True); self.c_optimizer.step()

        # --- encoder update: z_loss ONLY ---
        self.context_optimizer.zero_grad()
        z_loss = self.z_loss_weight * repo_z_loss(task_z, indices, b)
        z_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # --- critic update on detached z, with behavior value penalty ---
        z_for_q = task_z if self.allow_backward_z else task_z.detach()
        q1 = self.qf1(t, b, obs_f, act_f, z_for_q)
        q2 = self.qf2(t, b, obs_f, act_f, z_for_q)
        v_pred = self.vf(t, b, obs_f, z_for_q)
        with torch.no_grad():
            target_v = self.target_vf(t, b, next_f, task_z)
            if self.use_brac and self.use_value_penalty:
                target_v = target_v - self.get_alpha * div_estimate
        rewards_f = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_f = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_f + (1. - terms_f) * self.discount * target_v
        qf_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        qf_loss.backward(); self.qf1_optimizer.step(); self.qf2_optimizer.step()

        # --- value function update (max-entropy target, kept even offline) ---
        min_q = torch.min(self.qf1(t, b, obs_f, new_actions, task_z.detach()),
                          self.qf2(t, b, obs_f, new_actions, task_z.detach()))
        v_target = min_q - log_pi if self.max_entropy else min_q     # entropy term
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        soft_update(self.vf, self.target_vf, self.soft_target_tau)

        # --- policy update on detached z, with behavior policy regularization ---
        policy_loss = (log_pi - min_q + self.get_alpha.detach() * div_estimate).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

        alpha_loss = -(self._alpha_var * (div_estimate - self.target_divergence).detach()).mean()
        alpha_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            self._alpha_var.grad.zero_()
```
