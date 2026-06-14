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
with `Q̄` a target network without gradients. Good — that's the skeleton of stable offline control, and BCQ and BEAR both fall out of it as special cases, so I'm not inventing a one-off.

The thing I have to settle is what `D` is and how I estimate it, because I do not have `π_b` — only samples from it, the logged actions. If I pick a KL divergence `D_KL(π_θ ‖ π_b) = E_{a∼π_θ}[ log π_θ(a) − log π_b(a) ]`, I'd need the behavior density `π_b(a|s)`, which means fitting a "cloned policy" by max-likelihood on the data and then evaluating it — an extra generative model that can be badly miscalibrated. I'd rather not. There's a way around it: any f-divergence has a dual (Fenchel) form,
```
D_f(p, q) = max_g  E_{x∼p}[ g(x) ] − E_{x∼q}[ f*(g(x)) ],
```
where `f*` is the convex conjugate of `f`. For KL, `f(x) = −log x` gives `f*(t) = −log(−t) − 1`. So instead of estimating a density I learn a discriminator `g` by minimax — it plays the role of the log-density-ratio — and I get a sample-based divergence estimate with no cloned policy at all. I'll keep `g` honest with a gradient penalty so the minimax doesn't run away. That settles the control side: a SAC-style twin-Q actor-critic with a dual-form KL behavior penalty. I'll even keep the maximum-entropy term, which feels odd offline since I'm not exploring — but in something like Ant different actions can produce the same next state and reward, and the entropy bonus stops the policy from collapsing onto one of several equivalent actions, so it's worth keeping. There is a real cost here I should flag to myself: this whole family is sensitive to the reward scale and to the regularization strength `α`. I'll be tuning `α` per environment, and the spread is going to be wide.

Now the hard half — task inference — and this is where the offline constraint really bites, because it kills the standard approach. The strongest off-policy meta-RL method I know, PEARL, infers the task as a latent variable `z`. Its encoder `q_φ(z|c)` reads the context `c` (a set of transitions) and produces a posterior over `z`; the policy and value functions condition on `z`. Three design choices make it work on-policy. The encoder is probabilistic and permutation-invariant: each transition `c_n` produces a Gaussian factor `Ψ_φ(z|c_n) = N(μ_n, σ_n)`, and they're fused by a product of Gaussians,
```
q_φ(z|c_{1:N}) ∝ Π_n Ψ_φ(z|c_n),    σ² = 1 / Σ_n σ_n^{−2},   μ = σ² Σ_n μ_n σ_n^{−2},
```
which is order-invariant by construction. There's a KL information bottleneck, `+ β KL(q_φ(z|c) ‖ N(0,I))`, that squeezes `z` down to the minimal task-relevant information and hands me a prior to sample from. And that probabilistic posterior is the engine of posterior-sampling exploration: at test time PEARL samples `z` from the prior, acts for an episode, updates the posterior from what it saw, and acts again as the belief narrows — temporally-extended exploration of an unknown task. The encoder itself is trained by the Bellman gradients of the critic.

Stare at that for a second in my setting. The information bottleneck and the whole probabilistic posterior exist *to support exploration* — to represent and then narrow uncertainty by acting. But I never act to disambiguate. At test time I get a fixed batch of logged transitions and that's it; there is no episode-by-episode belief-narrowing because there's no interaction. So a large chunk of PEARL's design is machinery for a thing I'm forbidden from doing. That's not just wasted — modeling uncertainty I'll never resolve by exploration is modeling noise. And the encoder being trained *only* by Bellman gradients is worse than wasteful here: I just argued the offline Bellman backups can become ill-behaved, even divergent. If my task representation learns exclusively through that signal, then when the value learning goes unstable, the embedding goes unstable with it, and the tasks stop being distinguishable. The task inference is hostage to the very instability I'm fighting on the control side. So porting PEARL to offline — same encoder, same bottleneck, just no exploration — should leave me with a representation that learns slowly through a fragile channel and carries uncertainty structure I can't use. I need to rethink task inference from the ground up, for *this* setting.

Let me go back to first principles and ask what the latent variable even *is* here. The tasks differ only in `P` and `R`. Suppose I restrict to deterministic MDPs — `P(s'|s,a) = δ(s' − t(s,a))` for a transition map `t` — which is exactly what the standard continuous-control benchmarks are. And suppose the tasks satisfy what I'll call task-transition correspondence: for any two tasks and any `(s,a)`,
```
P_1(·|s,a) = P_2(·|s,a)  and  R_1(s,a) = R_2(s,a)   ⟺   T_1 = T_2.
```
In words: the pair `(P, R)` *identifies* the task — no two distinct tasks agree on transition and reward everywhere. Under determinism this means: given `(s,a)` and the task, there's a unique outcome `(s', r)`, so each task defines a map `f_T(s,a) = (s', r)`. The whole family `{f_T}` lives on the space of transitions `S × A × S × ℝ` — which is precisely the context space. So the encoder isn't inferring some fuzzy belief; it's embedding the function `f_T`.

And here's what that buys me. Because `(P, R)` identify the task pointwise, a *single* transition tuple `(s, a, s', r)` in principle already pins down which task I'm in — `f_T` is determined by its values, and one value is a real constraint on which task it could be. I don't need to integrate evidence across many transitions to slowly become confident; any one transition carries task identity. Two consequences drop out immediately. The encoder should be permutation-invariant — order can't matter if each transition independently reveals the task. And it should be *deterministic* — there's no irreducible uncertainty to represent, because the task is recoverable, not guessed. The probabilistic posterior, the product of Gaussians, the information bottleneck — all of that was answering a question ("how uncertain am I, and how do I explore to reduce it?") that doesn't arise once I'm in the deterministic, correspondence-satisfying, no-exploration regime. So: a deterministic encoder. Each transition gets embedded, and I aggregate the per-transition embeddings into one `z`. The natural permutation-invariant aggregation is just the mean — which is also exactly what the metric-learning few-shot people do when they form a class prototype as the mean of embedded supports, and they have a principled reason (the mean is the Bregman representative of a cluster under squared Euclidean distance). I'll take the mean over the transition embeddings.

But "deterministic encoder, mean-aggregated" only says how I *combine* embeddings; it says nothing about what makes the embedding *good*. If I just let the encoder learn through Bellman gradients like PEARL, I'm back in the trap. I want a *direct* signal that shapes the geometry of `z` so that same-task transitions land together and different-task transitions land apart — decoupled from the value learning entirely. So the question becomes: what objective do I put on the embedding space itself?

Before I pick one, let me make sure direct separation is genuinely *necessary* and not just aesthetically nice, because if it isn't I shouldn't add a whole objective for it. The value function `Q_θ(s, a, z)` is a continuous neural network. Continuity says: for every `ε > 0` there's an `η > 0` such that `|z_1 − z_2| < η ⟹ |Q̂_θ(s, a, z_1) − Q̂_θ(s, a, z_2)| < ε`. Now take two *different* tasks whose embeddings happen to be close, `|z_1 − z_2| < η`. Then the network is *forced* to give them nearly equal Q-values, `|Q̂_θ(s,a,z_1) − Q̂_θ(s,a,z_2)| < ε`. But the two tasks have different transition and reward functions, so their *true* Q-values, `Q(s,a,z_1)` and `Q(s,a,z_2)` via `R_z(s,a) + γ E_{s'∼P_z}[V(s')]`, are *not* guaranteed to be close — they can differ a lot. A single continuous network simply cannot output two well-separated true values at two nearly-identical inputs. So if the encoder lets distinct tasks' embeddings sit close together, the conditioned value functions for those tasks become unlearnable — the approximator can't represent both. That settles it: I *must* keep the embeddings of distinct tasks far apart, or control fails downstream no matter how good the actor-critic is. The geometry of `z` is not cosmetic; it's a precondition for the value functions to exist.

Good. So I want an objective on `z` that clusters same-task and separates different-task. This is plain distance metric learning, and the canonical loss is the contrastive loss,
```
L_cont(x_i, x_j) = 1{y_i = y_j} ‖q_i − q_j‖₂²  +  1{y_i ≠ y_j} max(0, m − ‖q_i − q_j‖₂)²,
```
where `q = q_φ(x)` and `y` is the task label. Same-task pairs feel an attractive quadratic pulling them together; different-task pairs feel a margin-`m` hinge pushing them apart up to distance `m`. Let me just try this and see if it works. I embed transitions, label pairs by task identity, descend this loss.

It degenerates. The clusters don't cleanly separate — I get blobs that contain transitions from *several* tasks mixed together. Let me figure out *why* instead of just observing it, because the why is going to tell me the fix. Two things are wrong, and they compound. First, look at the repulsive term. Hadsell's spring picture is honest about it: `max(0, m − ‖q_i − q_j‖)²` is a spring that acts *only within radius `m`*. Once a different-task pair is already farther than `m` apart it contributes exactly zero gradient — fine — but the real problem is at the *other* end. My latent space is squashed by a tanh to the bounded cube `(−1, 1)^l`, and the encoder starts randomly initialized, so at the beginning *every* embedding is bunched near the origin and *every* pairwise distance is small. Right where I most need a strong shove apart — distances near zero — the margin hinge's gradient with respect to distance is `−2(m − d)`, which is bounded; it's a fixed-stiffness spring, not a strong one, and it has no special urgency when `d` is tiny. So the repulsion is weakest in exactly the regime, small distances, where I need it most.

The second thing is deeper, and it's about the attractive/quadratic part — really about using *any* positive power of distance as the objective. Let me compute what minimizing a sum of squared distances actually does, on a single dataset `X = {x_i}`. The accumulative squared-distance objective is
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
The accumulative squared-distance loss *is* the dataset variance, up to a constant. That's the whole problem, laid bare. A squared-distance metric objective is just variance maximization. And variance is a *global* scalar — there are many embedding distributions with the same variance, including badly degenerate ones. Concretely, on each axis a Bernoulli-type distribution that piles half the mass at `+1` and half at `−1` maximizes variance on the bounded interval, and such a distribution can have *several tasks crammed into each of the two piles* — the variance is huge but the clusters are merged. Variance being large says nothing about whether *every* pair of distinct task clusters is separated. That's exactly the degenerate blobs I saw: the optimizer is happily maximizing variance by spreading mass to the extremes while letting distinct tasks share a pile. And it's not a quirk of the power `2` — any positive power inherits the same disease, because positive powers reward *spread*, and spread is a global aggregate that a coarse, lumpy distribution can satisfy without fine-grained separation.

Let me sanity-check that this distinction is real and not an artifact of my reading, with a measurement I can make on the embedding space without ever touching the proposed method's downstream returns. Define the effective separation rate: the fraction of different-task embedding pairs whose distance exceeds what randomly placed vectors would have — on `(−1,1)^l` the expected random pairwise distance is `√(2l/3)`. And separately measure plain RMS pairwise distance. The square-distance objective and an inverse-distance objective give *nearly the same RMS distance* but very different separation rates — the square loss leaves a much smaller fraction of distinct-task pairs actually well-separated. That's the quantitative fingerprint of "high variance, poor separation": RMS, a global average, can't tell the two apart; the separation rate can. So the diagnosis holds.

So what do I actually need? A repulsive term between different-task pairs whose force is *strong when the distance is small* and *fades as the distance grows* — the opposite profile of the margin spring, which is weak at small distance and dead beyond `m`. I want something that screams when two distinct tasks are on top of each other and relaxes once they've separated. The function with exactly that profile is an inverse power of distance: `1 / dᵏ` blows up as `d → 0` and decays as `d` grows. So replace the positive-power margin term with a *negative* power:
```
L_dml(x_i, x_j) = 1{y_i = y_j} ‖q_i − q_j‖₂²  +  1{y_i ≠ y_j} · β / ( ‖q_i − q_j‖₂ⁿ + ε ),
```
with `ε > 0` to keep the denominator off zero and `n ≥ 0` the power. Now the different-task term is a genuine repulsive *potential*, not a capped spring. Two distinct-task embeddings that start near the origin together feel an enormous push apart; once they're well separated the push relaxes and stops fighting the rest of the configuration. And critically, because the force is per-pair and explodes at short range, it cannot be satisfied by the global variance trick — you can't hide two tasks in the same pile, because the pairs *inside* that pile are at small distance and the `1/dⁿ` term is screaming at them. Every pair of distinct clusters is forced apart. That's exactly the guarantee the squared-distance / variance objective couldn't give.

It clicks into place when I picture it physically. Treat each different-task embedding as a point charge of the same sign sitting in the bounded cube. Then `β / dⁿ` with `n = 1` *is* the Coulomb potential — like charges repel with a potential that grows without bound as they approach. What does a system of like charges in a conducting box do at equilibrium? By Gauss's law the charges migrate to the boundary, and they pile up where the local curvature is highest — the edges and the corners. So I should *predict*, before running anything, that an encoder trained with this loss will push the task embeddings out toward the faces, edges, and vertices of `(−1,1)^l`, spreading them as far apart as the bounded space allows, with the highest concentration near the corners. The bounded tanh space isn't incidental — it's the conducting box that makes the repulsion settle into a well-separated equilibrium instead of flinging everything to infinity. Same-task transitions, meanwhile, are pulled together by the attractive `‖q_i − q_j‖²` term, so each task collapses to a tight cluster and the clusters scatter to the extremes. That is precisely the geometry the continuity argument demanded: distinct tasks maximally far apart so the conditioned value functions are representable.

Which power `n`? The general negative-power family is the design space, and I should think about the cases. `n = 1` is literal Coulomb — clean physical intuition, the repulsion decays gently. `n = 2` is sharper: the force falls off faster with distance, so it concentrates its effort on the *closest* offending pairs and stops meddling with already-separated ones sooner. And `n = 2` has a pedigree — it's exactly the inverse-square form of Cauchy graph embedding (Luo et al. 2011), which maximizes `Σ_{ij} W_{ij} / (‖Y_i − Y_j‖² + σ²)` and was shown to preserve local topology better than the Laplacian/quadratic objective `Σ_{ij} W_{ij} ‖Y_i − Y_j‖²`. That's the same lesson I just re-derived from scratch — a quadratic distance objective over-weights pushing *dissimilar* points apart and under-serves the *local* structure of similar points, while the inverse form emphasizes the right thing. So `n = 2` isn't arbitrary; it's the metric whose whole point is the local-topology preservation I need. I'll treat `n` as a knob, with `n = 1` (inverse) and `n = 2` (inverse-square) as the two I care about, expecting the inverse-square to give the tightest, best-separated clusters. The `β` is just a scale to match the magnitude of the attractive and repulsive terms — I want them comparable when distances are at a reasonable scale, around half the per-dimension range on `(−1,1)^l`, so neither term trivially dominates; and `ε` floors the denominator. There's a subtlety in how I write the per-pair distances in code: I'll use the mean squared difference per dimension under the root so the scale is dimension-normalized, but the structure is exactly this — squared distance for same-task, inverse power for different-task.

Now I have to decide how this encoder objective relates to the value-learning gradients, and this is the third design choice. PEARL decoupled task inference from control by *sampling strategy* — different data for the encoder than for the RL batch — because on-policy it needed the encoder data to look like exploration data. I have no exploration, so that particular decoupling is moot; the encoder and the actor-critic can read the same sampled transitions. But there's a different decoupling I *do* need, and the continuity-and-divergence analysis tells me why. I argued the offline Bellman backups can grow to enormous magnitude on the harder tasks even when regularized — the value functions can sit at scales like `10⁷`. If I let the encoder receive gradients from the critic as well as from `L_dml`, then the Bellman gradient, scaled by those huge value magnitudes, will simply swamp the distance-metric gradient. The carefully shaped repulsion gets drowned out, the embedding collapses, and I'm back to merged clusters. So I decouple by *gradients*: the encoder is trained *only* by `L_dml`. When the actor and critic consume `z`, they get it *detached* — `z̄`, no gradient flowing back into the encoder. The distance-metric objective is clean, bounded, and scale-controlled; the value objective is the unstable one; keeping them in separate gradient paths protects task inference from the value learning's instability. Three objectives, three optimizers, updated separately:
```
φ ← φ − α₁ ∇_φ Σ_{ij} L_dml(c_i, c_j)        # encoder, distance metric only
θ ← θ − α₂ ∇_θ Σ_i L_actor(b_i, z̄_i)         # actor, z detached
ψ ← ψ − α₃ ∇_ψ Σ_i L_critic(b_i, z̄_i)        # critic, z detached
```
That's the full skeleton: a deterministic mean-aggregating encoder trained by the inverse-power distance metric loss, detached from a behavior-regularized SAC actor-critic that conditions on the inferred task variable.

Let me write the embedding and the loss concretely, the way I'd actually run it. The encoder is a plain MLP from one transition `(s, a, r)` — or `(s, a, r, s')` if next-state helps tell tasks apart — to a `latent_dim` vector. I embed every transition in a task's context batch and take the mean to get that task's `z`. For the loss, I have a meta-batch of tasks; I take the per-task embeddings, and for each pair of tasks I apply the squared distance if they're the same task (within-task pairs, pulling together) and the inverse-power if they're different tasks (pushing apart), then average:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


class FOCALAgent(nn.Module):
    """Deterministic context encoder + mean aggregation; SAC heads condition on z."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        # deterministic encoder: one transition -> latent_dim (no IB, no Gaussian params)
        self.context_encoder = build_mlp(context_input_dim, latent_dim,
                                         hidden_dim=200, n_layers=3)
        # tanh on the encoder output bounds z to (-1, 1)^l -> the "conducting box"
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def clear_z(self, num_tasks=1):
        self.z = torch.zeros(num_tasks, self.latent_dim)
        self._context = None

    def infer_posterior(self, context):
        # embed every transition, then MEAN over transitions -> deterministic z per task
        embeddings = self.context_encoder(context)
        embeddings = embeddings.view(context.size(0), -1, self.latent_dim)
        self.z = torch.mean(embeddings, dim=1)                 # (num_tasks, latent_dim)

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def detach_z(self):
        self.z = self.z.detach()                               # cut gradient path to encoder

    def get_action(self, obs, deterministic=False):
        in_ = torch.cat([torch.as_tensor(obs)[None], self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.vf, self.target_vf]
```

And the distance-metric loss — the heart. Same-task pairs get the squared (here root-of-mean-squared, dimension-normalized) distance; different-task pairs get the inverse-power repulsion. I compute the per-task embeddings, then loop over task pairs in the meta-batch:

```python
def dml_loss(task_z, indices, b, beta=1.0, eps=1e-3, n=2.0):
    """Distance metric learning loss on the task embeddings.
    task_z is stacked per-task embeddings; indices[i] is the task id of block i.
    Same-task pairs: pull together (squared distance).
    Different-task pairs: push apart (inverse-power repulsion ~ Coulomb potential)."""
    pos_loss, neg_loss = 0.0, 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(len(indices)):
        zi = task_z[i * b]
        for j in range(i + 1, len(indices)):
            zj = task_z[j * b]
            d_sq = torch.mean((zi - zj) ** 2)                  # mean squared diff / dim
            if indices[i] == indices[j]:                       # same task -> attract
                pos_loss += torch.sqrt(d_sq + eps)
                pos_cnt += 1
            else:                                              # different task -> repel
                neg_loss += beta / (d_sq ** (n / 2.0) + eps * 100)
                neg_cnt += 1
    return pos_loss / (pos_cnt + eps) + neg_loss / (neg_cnt + eps)
```

The `d_sq ** (n/2)` is just `‖q_i − q_j‖ⁿ` written through the squared distance: `n = 2` makes it `d_sq` (inverse-square, the Cauchy form), `n = 1` makes it `sqrt(d_sq)` (inverse, Coulomb). The two `ε` floors keep both terms finite. Nothing here touches the value functions; this gradient flows only into the encoder.

Now the offline actor-critic update, conditioned on the *detached* `z`, with the behavior penalty. I learn a dual-form KL discriminator `c` (the BRAC `g`), estimate the divergence from samples, and fold it into the critic target (value penalty) and/or the actor:

```python
class FOCALAlgorithm:
    """Decoupled training: encoder by DML loss only; SAC+BRAC on detached z."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer, config):
        self.agent = agent
        self.train_tasks = train_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.batch_size = config.get('batch_size', 256)
        self.meta_batch = config.get('meta_batch', 16)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)         # SAC/offline sensitivity
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.z_loss_weight = config.get('z_loss_weight', 10.0)      # weight on L_dml
        self.alpha = config.get('alpha', 500.0)                     # behavior reg strength
        self.use_value_penalty = config.get('use_value_penalty', False)
        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer  = optim.Adam(agent.policy.parameters(),          lr=lr)
        self.qf1_optimizer     = optim.Adam(agent.qf1.parameters(),             lr=lr)
        self.qf2_optimizer     = optim.Adam(agent.qf2.parameters(),             lr=lr)
        self.vf_optimizer      = optim.Adam(agent.vf.parameters(),              lr=lr)
        self.c_optimizer       = optim.Adam(self.divergence.parameters(),       lr=1e-4)

    def _take_step(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        # encode context -> per-task z (gradients to encoder live here)
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_f, act_f, next_f = (x.view(t * b, -1) for x in (obs, actions, next_obs))

        # --- BRAC dual-form KL: train discriminator, estimate divergence ---
        div_estimate = self.divergence.dual_estimate(obs_f, new_actions, act_f, task_z)
        c_loss = self.divergence.dual_critic_loss(obs_f, new_actions, act_f, task_z)
        self.c_optimizer.zero_grad(); c_loss.backward(retain_graph=True); self.c_optimizer.step()

        # --- encoder update: DML loss ONLY (z detached everywhere downstream) ---
        self.context_optimizer.zero_grad()
        z_loss = self.z_loss_weight * dml_loss(task_z, indices, b)
        z_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # --- critic update on detached z, with behavior value penalty ---
        with torch.no_grad():
            target_v = self.agent.target_vf(next_f, task_z.detach())
            if self.use_value_penalty:
                target_v = target_v - self.alpha * div_estimate     # BRAC value penalty
        q1 = self.agent.qf1(obs_f, act_f, task_z.detach())
        q2 = self.agent.qf2(obs_f, act_f, task_z.detach())
        rewards_f = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_f = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_f + (1. - terms_f) * self.discount * target_v
        qf_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        qf_loss.backward(); self.qf1_optimizer.step(); self.qf2_optimizer.step()

        # --- value function update (max-entropy target, kept even offline) ---
        min_q = torch.min(self.agent.qf1(obs_f, new_actions, task_z.detach()),
                          self.agent.qf2(obs_f, new_actions, task_z.detach()))
        v_pred = self.agent.vf(obs_f, task_z.detach())
        v_target = min_q - log_pi                                    # entropy term
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        soft_update(self.agent.vf, self.agent.target_vf, self.soft_target_tau)

        # --- policy update on detached z, with behavior policy regularization ---
        policy_loss = (log_pi - min_q + self.alpha * div_estimate.detach()).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()
```

The constants follow from the pieces. The latent dimension is small — 5 for the reward-varying tasks like Half-Cheetah-Vel and Sparse-Point-Robot, larger (20) for the harder dynamics-varying ones — because the task variable only has to encode `(P, R)` structure, not the full state. The encoder is a width-200 depth-3 MLP, the SAC heads width 300; the latent is tanh-bounded so the repulsion has a box to settle in. The DML weight sits around 10 so the metric signal is firmly present, and since it's the *only* thing training the encoder I don't have to worry about it competing with the critic gradient. The behavior-regularization `α` is the one I have to set per environment, and the spread is exactly as wide as I feared the sensitivity would make it: zero on the easy Sparse-Point-Robot (where bootstrapping doesn't bite and no constraint is needed), a few hundred on the cheetah tasks, up to `10⁶` on Ant-Fwd-Back where the value functions would otherwise diverge outright. I'll tune `α` adaptively, raising it when the measured divergence exceeds a small threshold and lowering it below, so I don't hand-search the whole range. The reward scale (5 on the locomotion tasks, larger on the sparse task) is the other SAC-inherited knob.

So the chain closes. I started with a setting that forbids exploration at both train and test time and pre-collected data only, which kills the standard offline-RL feedback that corrects out-of-distribution value estimates and kills the exploration that probabilistic meta-RL relies on. On the control side I tether the policy to the behavior support with a behavior-regularized actor-critic, estimating the divergence by a dual-form KL discriminator so I never need a cloned behavior density. On the inference side I noticed that under deterministic dynamics and task-transition correspondence the task is *recoverable* from individual transitions, not merely *guessable*, which makes the encoder deterministic and permutation-invariant and makes the whole probabilistic-bottleneck apparatus pointless. A continuity argument then showed that distinct tasks whose embeddings sit close are *unlearnable* by the conditioned value network, so I need to separate them directly — and the contrastive loss I reached for first degenerated, which I traced to the algebraic fact that a squared-distance metric objective is exactly variance maximization, a global statistic that a coarse, lumpy distribution satisfies while merging tasks. Flipping the repulsive term to a negative power of distance — a Coulomb-like potential that is strongest at short range and per-pair, the inverse-square case coinciding with Cauchy graph embedding — forces *every* pair of distinct task clusters apart and drives the embeddings to the corners of the bounded latent cube, exactly the geometry the continuity argument demanded. Finally, because the offline value functions can run to enormous scale, I keep the encoder's distance-metric gradients fully separate from the value-learning gradients — `z` detached when it feeds the actor and critic — so task inference is never swamped by the unstable Bellman signal. A deterministic mean-aggregating encoder, an inverse-power distance metric loss, gradient-level decoupling, and a behavior-regularized SAC underneath: an end-to-end, model-free offline meta-RL learner.
