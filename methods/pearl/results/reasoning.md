Let me start from the thing that actually hurts. I have a distribution of tasks `p(T)`, each one an MDP with its own reward or dynamics, and I want an agent that, after training across many of them, walks into a brand-new task and is good at it after a handful of trajectories. That is the meta-learning promise. But two costs are pulling in opposite directions and I keep paying both. There is the cost at *test* time — how many interactions before I'm doing well on the new task — and there is the cost at *meta-training* time — how many environment samples I burn while learning how to adapt in the first place. Everyone advertises the first; almost nobody talks about the second, and the second is brutal: I'm training across many tasks, many trials each, and if I can't reuse data, the total sample count is millions per task times a big task set. So the real goal is both at once: a meta-learner that is cheap to train because it reuses old data, and a test-time agent that adapts from almost nothing.

What does "reuse old data" mean concretely? It means off-policy: keep a replay buffer per task, learn from transitions collected by old policies. That is exactly the lever that makes single-task RL sample-efficient — soft actor-critic learns continuous control from a replay buffer and is far cheaper and more stable than any on-policy method I have. So my instinct is just: take the best off-policy actor-critic I have, condition it on some summary of the task, and meta-train it. And every efficient meta-RL method I know is on-policy — RL² runs a recurrent net over the on-policy stream and trains it with policy gradients; MAML and ProMP do inner and outer policy-gradient steps on on-policy trajectories; MAESN adapts a latent by gradient descent on on-policy data. They all spend on the order of `1e8` timesteps in meta-training. If I could just make the meta-learner off-policy I'd cut that by one or two orders of magnitude. Why hasn't everyone done it?

Here's the wall, and it's a principled one, not an engineering nuisance. Meta-learning rests on the idea that the data distribution you adapt from at *test* time should match the data distribution you were meta-trained to adapt from. A five-shot image classifier is meta-trained on five-example episodes because that's what it sees at test. Now transpose that to RL. At test time the agent adapts from the experience it gathers by *exploring* the new task — that experience is on-policy by construction: it's whatever the current adapting policy chose to do. The matching principle then says: meta-train to adapt from on-policy data too. But off-policy replay data — old transitions from stale policies, the whole point of which is that it's *not* what the current policy would do — is systematically unlike the on-policy data the adapted agent will face. So if I naively meta-train the adaptation mechanism on replayed off-policy transitions, I'm training it to adapt from a distribution it will never see at test, and it breaks. That's why off-policy meta-RL is "non-trivial." I actually tried the naive thing — stick a recurrent context encoder on top of an off-policy value learner — and couldn't get it to optimize. Recurrent DDPG-style memory control had only ever been shown on much simpler or discrete problems, and on these continuous-control meta-RL families it just doesn't train.

So I have a contradiction to resolve, not a knob to turn. Off-policy efficiency wants me to train on the whole replay buffer; the matching principle wants the adaptation data to be on-policy. Let me look at *what* has to match and *what* doesn't, instead of treating "the algorithm" as one indivisible thing. At test time, what does the agent actually do? It collects some exploration experience, turns that experience into a belief about the task, and then acts conditioned on that belief. There are two distinct jobs there: *inferring the task* from experience, and *acting* given the task. The matching principle only bites on the *input to the inference* — the experience the agent infers from must look like what it'll infer from at test. It says nothing direct about how I train the *control* part, the policy and value function. The policy just needs to be a good policy for whatever task it's told it's in; it doesn't care whether the transitions teaching it Q-values came from the current policy or a stale one — that's the ordinary off-policy claim, and it's fine.

That's the crack to pry open. Disentangle inference from control. Make the task belief an explicit object — call it a latent context variable `z` — that I infer from collected experience, and condition the policy `π(a|s,z)` and the critic `Q(s,a,z)` on it, treating `z` as just another part of the state. Then I can train the policy and critic *fully off-policy* on everything in the replay buffer (control doesn't need the matching), and train the *encoder* — the thing that maps experience to `z` — on data that *does* respect the matching, i.e. recent, near-on-policy experience. The data that trains the encoder need not be the data that trains the policy. That single decoupling is what lets me have off-policy efficiency for the expensive part (the RL) while keeping the distribution match exactly where it's load-bearing (the inference). I'll have to be careful about what "the encoder's data" is, but structurally the contradiction is gone.

Now, what *is* `z` and how do I infer it? I want a belief about the task from experience. The clean way to learn a mapping from data to a latent, when the true posterior is intractable, is amortized variational inference: train a recognition network `q_φ(z|c)` — `c` is the context, the collected transitions — to approximate the posterior `p(z|c)`, and reparameterize the sample `z = μ_φ(c) + σ_φ(c) ⊙ ε`, `ε ~ N(0,I)`, so I can backprop through the sampled `z`. The objective in that frame is a variational lower bound,

  `E_T [ E_{z ∼ q_φ(z|c^T)}[ R(T, z) ] − β · KL( q_φ(z|c^T) ‖ p(z) ) ]`,

with a unit-Gaussian prior `p(z) = N(0,I)` and `R(T,z)` some reconstruction/likelihood term. So I maximize an objective that rewards a `z` good for the task while keeping `q_φ(z|c)` near the prior. I should pause on that KL term because it's doing more than one job. It is, of course, the VAE regularizer. But read it the other way: `KL(q_φ(z|c) ‖ p(z))` is a variational upper bound on the mutual information `I(Z;C)` between the latent and the context. Penalizing it is an information bottleneck — it pressures `z` to keep only the bits of the context I actually need to act well and to discard the rest. To see that the cost is real and not just a slogan: a belief that has actually learned something, say `N(1.6, 0.8)` on one coordinate, pays `KL(N(1.6,0.8) ‖ N(0,1)) ≈ 1.29` nats, while a belief that stayed at the prior pays exactly `0`. So every bit of task information `z` carries comes out of a budget, and β sets the price. Why do I care? Because I'm training on a finite set of tasks, and a fat unconstrained `z` could memorize idiosyncrasies of the training tasks and overfit. Charging for information should push `z` toward something close to a minimal task-relevant statistic. I can't prove from here that this is what generalizes to held-out tasks — that's an empirical question I'd want to settle by sweeping β on the test tasks — but the mechanism is at least pointed the right way: β is the dial on how much the task belief is allowed to remember, and that should trade directly against overfitting to the training set.

Next question: what should `R(T,z)` be — what do I actually train the encoder to make `z` good *for*? There are three candidates and I genuinely don't know up front which wins, so let me lay out what each one optimizes and what could go wrong. The generative-modeling reflex is: reconstruct the MDP — predict the reward and next state from `(s,a,z)`, train `z` to make a reward/dynamics model accurate. That's clean and self-supervised. But it has a failure mode I can name concretely: in the reward-varying locomotion families the dynamics are *identical* across tasks and only the reward differs, so an encoder straining to reconstruct transitions burns capacity modeling next-state physics that carries zero task information, and the actual task signal — the reward function — is a small part of the reconstruction loss it's minimizing. Option two is to train `z` to maximize the actor's returns directly, but that couples the encoder's gradient to the high-variance policy-gradient objective, which is exactly the noise I went off-policy to avoid. Option three: the thing the agent needs in order to *act* is the value, so train the encoder so that `z` makes the *critic* accurate — `R(T,z)` is the (negative) critic loss, and the encoder gets its gradients from the Bellman update for the Q-function. That aims the representation at "what do I need to know to value actions correctly," uses the same low-variance off-policy signal the critic already runs on, and ignores task-irrelevant dynamics by construction. The argument favors the critic objective, but the reconstruction-vs-critic question is really empirical — I'd want to run all three as an ablation and compare held-out return before trusting it. I'll carry the critic-trained encoder as the design and let the architecture follow.

Now the architecture of `q_φ(z|c)`, and this is where I want to be careful rather than just reach for an RNN like RL² did. What is the context, really? It's the experience collected on the task, a bunch of transitions `{(s_i, a_i, s'_i, r_i)}`. Here's the structural fact: to identify an MDP — its reward and its dynamics — the *order* of those transitions doesn't matter. The Markov model is made of local transition and reward laws; a trajectory can be temporally correlated, but permuting the observed local tuples does not change the information they contain about those laws. The context is a *set*, not a sequence. An RNN imposes an order that isn't there: it has to learn to be order-invariant, wastes capacity doing so, is slow to optimize over long horizons, and can overfit to spurious temporal patterns. So I want a *permutation-invariant* encoder.

The standard recipe for embedding a set permutation-invariantly is the prototypical-network move: embed each element with a shared network and then *aggregate* with a symmetric function — they average the per-example embeddings. I could do that: run each transition `c_n` through `f_φ`, average, predict `z`. But I want `z` to be *probabilistic* — that's the whole point, I want a belief with uncertainty — and I want the aggregation to behave like *evidence accumulation*: more transitions should make me more certain. A plain mean of point embeddings doesn't carry uncertainty and doesn't sharpen as `N` grows. So let me make each transition produce not a point but a *Gaussian factor* over `z` — `Ψ_φ(z|c_n) = N(f_φ^μ(c_n), f_φ^σ(c_n))` — each transition's little vote on what the task is, with its own mean and spread. Now how do I combine Gaussian votes into one belief? The encoder models the approximate posterior over `z` given the context as proportional to the *product* of the per-transition factors:

  `q_φ(z|c_{1:N}) ∝ ∏_{n=1}^N Ψ_φ(z|c_n).`

A product is a symmetric function of the factors, so this is automatically permutation-invariant — exactly the set structure I argued for. And a product of Gaussians is itself Gaussian, so it has a closed form I can derive. Take diagonal factors `N(μ_n, σ_n²)` per coordinate and multiply the densities: in the exponent I'm summing `−(z−μ_n)²/(2σ_n²)`. Collecting the quadratic and linear terms in `z`, the precision (inverse variance) of the product comes out as the *sum* of the per-factor precisions, and the natural parameter `μ/σ²` as the *sum* of the per-factor natural parameters:

  `1/σ² = Σ_n 1/σ_n²`,  i.e.  `σ² = 1 / Σ_n (1/σ_n²)`,
  `μ = σ² · Σ_n (μ_n / σ_n²)`.

Before I lean on that I should make sure I didn't drop a factor of two or mix up where the weights go — it's an easy place to fool myself. Let me take two concrete factors on a single coordinate, `N(2, 1)` and `N(0, 4)`, push them through the formula, and separately compute the *actual* product of the two densities on a fine grid and read off its mean and variance. The formula gives precision `1/1 + 1/4 = 1.25`, so `σ² = 0.8` and `μ = 0.8·(2/1 + 0/4) = 1.6`. Multiplying the densities numerically and renormalizing gives mean `1.600000` and variance `0.800000` — they agree to six places, so the closed form is right and I have the weighting in the right place. The mean `1.6` sits much closer to the `σ²=1` factor's `2` than to the `σ²=4` factor's `0`: the confident factor pulled harder, which is the behavior I'd want from evidence fusion. To be sure that's a real effect and not luck, I push it: with means `5` (var `0.25`) and `0` (var `4`) the precision-weighted mean is `4.71`, not the plain average `2.5` — it's dragged almost all the way to the confident vote.

The other claim I need to actually check is that the belief *sharpens* with evidence rather than just shifting. Since precisions add, every extra factor strictly increases the total precision, so `σ²` strictly decreases. Numerically: appending a third factor (var `0.5`) to the two above drops the posterior variance from `0.8` to `0.31`. And for `N` identical factors of variance `v`, the formula gives posterior variance `1/(N/v) = v/N` — I check `v=2` and get `2, 1, 0.5, 0.25` for `N = 1, 2, 4, 8`, exactly the `1/N` contraction I'd expect from accumulating independent observations. So the more transitions I feed in, the narrower the belief, and an empty context (no factors) leaves it at the unit-Gaussian prior. That's not an arbitrary pooling op — it's behaving like a Bayesian filtering update, implemented by a learned inference network: the encoder predicts each transition's Gaussian vote, the product fuses them order-free, and the variance does the right thing as evidence comes in.

Having a *probabilistic* belief rather than a point estimate turns out to buy something I hadn't been using, and it matters most exactly where I'm most worried — the sparse-reward case. Think about test time on a sparse task: no reward signal until I happen to touch the goal. Per-timestep action noise is useless here — it jitters around and never commits to going *anywhere* to check. What I need is to pick a *hypothesis* about where the goal is and go test it, coherently, for a whole episode. Now look at what I already have: a posterior `q_φ(z|c)` over the task. I can sample one `z` from it, hold that `z` *fixed for the whole episode*, and let the policy act optimally for that single hypothesis. That's a committed, coherent attempt at one task — exactly the temporally extended probe I said I needed. Then I take what I observed, update the posterior, sample again, and try a hypothesis consistent with what I've learned; as the belief narrows, behavior shifts from exploring to exploiting on its own. Written out, this is the classical posterior-sampling exploration scheme — keep a posterior over MDPs, draw one, act optimally for it for an episode — recovered here for free, because acting optimally for a sampled `z` *is* acting optimally for a sampled task-hypothesis.

That recovery hinges on `z` being probabilistic, and it's worth being precise about why, because it makes a falsifiable prediction. If I collapsed `z` to a point estimate, the per-episode `z` would be deterministic given the context, so the only randomness left during a sparse episode would be the policy's own action-level stochasticity — independent jitter at each step, not a persistent commitment to one hypothesis. With no committed hypothesis there's no coherent multi-step probe, so on the sparse half-circle navigation task I'd expect a deterministic-`z` variant to mostly fail to ever find the goal, while the probabilistic version reaches it. I can't confirm that from the desk, but it's a clean ablation — point-estimate `z` vs. sampled `z` on the sparse task — and if the deterministic variant did *not* collapse, my whole story about where the exploration comes from would be wrong. So the probabilistic latent is load-bearing, not decoration.

Now let me actually build the control side on top of soft actor-critic and write the losses, because the details of *which gradients touch the encoder* turn out to matter. SAC gives me: a critic `Q(s,a)`, a value net `V(s)` with a slow target `V̄`, a squashed-Gaussian actor, two Q-nets I take the min of to fight overestimation, and the max-entropy objective with reward scaled by the inverse temperature. I append `z` to the state everywhere. The critic regresses the soft Bellman target:

  `L_critic = E_{(s,a,r,s',d) ∼ B, z ∼ q_φ(z|c)} [ ( Q(s,a,z) − ( ηr + (1−d)γV̄(s', z̄) ) )² ]`,

where `η` is the reward scale, `γ` is the discount, `d` is the terminal flag, and `z̄` means I do *not* backprop into the encoder through the target — the target is a fixed regression target, same logic as a target network. The actor loss is SAC's, with `z` as an extra input and detached so the actor doesn't reshape the task belief:

  `L_actor = E_{s ∼ B, a ∼ π, z ∼ q_φ(z|c)} [ D_KL( π(a|s, z̄) ‖ exp(Q(s,a,z̄)) / 𝒵(s) ) ]`,

and the encoder's own regularizer is the KL to the prior,

  `L_KL = β · KL( q_φ(z|c) ‖ p(z) )`.

Here's the subtle part. I argued the encoder should be shaped by the *critic*, not the actor or the value reconstruction. So in the code I must make `z` carry gradients into the encoder *only* through the Q-loss (and the KL), and detach it everywhere else — in the value loss and the policy loss the `z` is detached. If I let the policy and value gradients also flow into the encoder, the task representation gets pulled in three directions and the inference signal muddies; keeping it driven by the Bellman/critic objective is what keeps `z` a clean, control-relevant task summary. So: the Q-optimizer step and the context-encoder-optimizer step happen together, on the same backward pass, with `z` live for the Q's but detached for `V` and `π`.

Now the load-bearing decoupling, made concrete — *what data does each part see*. The actor and critic are off-policy: I sample minibatches of *decorrelated transitions from the entire replay buffer*. That's where all the sample-efficiency comes from, and decorrelation matters — sampling whole correlated trajectories for the RL batch destabilizes the value updates. The encoder is the part that must respect the matching principle. If I let the context sampler also draw from the entire buffer — fully off-policy context — I reintroduce exactly the train/test mismatch I was trying to kill: at test time the encoder sees on-policy exploration data, so feeding it ancient off-policy transitions during meta-training trains it on the wrong distribution and it should hurt. So I sample the context from *recently collected* data, re-collected periodically, and crucially from a batch *distinct* from the RL batch. It doesn't have to be strictly on-policy — an in-between of "recent" keeps the distribution close enough while still letting me reuse — but it cannot be the whole buffer. The separate context sampler `S_c` is the operational form of "the encoder's data ≠ the policy's data," and it's the thing that makes off-policy meta-RL actually train.

So the meta-training inner step, walking the gradients in order: sample a set of tasks for the minibatch; for each, sample a context batch from its recent data and infer `q_φ(z|c)`, sampling `z` by reparameterization; sample a separate off-policy RL batch from the full buffer. Run the agent forward to get `z` and the policy outputs. Compute the KL loss and backward it into the encoder (retain the graph, because `z` is shared). Compute the Q-loss with live `z`, backward it, and step the Q-optimizers and the encoder optimizer together — the encoder sees critic gradients plus the KL. Then compute the value loss against `min(Q1,Q2)` on fresh policy actions, with `z` detached, and step `V`; soft-update the target `V̄`. Then the policy loss, `z` detached, plus SAC's mean/std regularizers, and step the actor. Detach `z` between mini-batches so backprop doesn't run away through the shared latent.

For data collection each iteration: for a few sampled tasks, clear that task's encoder buffer (so context is recent), then collect some steps using `z` sampled from the *prior* (pure exploration, before the agent knows the task), and some steps using `z` from the posterior `q_φ(z|c)` after a bit of context has accumulated — that mix gives the encoder both prior-conditioned and posterior-conditioned data. Initial data is collected from the prior for all tasks to seed the buffers.

And meta-testing is the posterior-sampling loop I designed: start with empty context, sample `z` from the prior, roll out a trajectory acting optimally for that `z`, accumulate only that task's trajectory into the context, re-infer `q_φ(z|c)`, sample again, repeat — committing to a coherent hypothesis each episode and sharpening the belief as evidence comes in. For evaluation I make the number of pure exploration trajectories configurable (`num_exp_traj_eval`, default 1, commonly 2 for MuJoCo configs and 5 for sparse point robot), log online returns by rollout, and report the last rollout within the evaluation budget as the final return.

Let me put the whole thing into the code I'd actually run, filling the context-encoder and meta-gradient slots that were left open. The agent holds the permutation-invariant probabilistic encoder (the product-of-Gaussians fusion) and a SAC actor-critic conditioned on `z`; the algorithm holds the decoupled samplers and the gradient order I just worked out.

```python
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    # Fuse per-transition Gaussian factors into one belief: precisions ADD,
    # natural params (mu/sigma^2) ADD. Permutation-invariant; variance shrinks
    # with more transitions -> Bayesian filtering.
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class MetaRLAgent(nn.Module):
    """Probabilistic permutation-invariant context encoder + z-conditioned SAC."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim, self.action_dim, self.latent_dim = obs_dim, action_dim, latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = kwargs.get('sparse_rewards', False)

        # Encoder input: one transition (o, a, r) [, next_o]; output 2*latent_dim
        # = per-coordinate mean + (pre-softplus) variance of this transition's vote.
        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        context_output_dim = latent_dim * 2
        self.context_encoder = build_mlp(context_input_dim, context_output_dim,
                                         hidden_dim=200, n_layers=3)
        self.context_encoder_output_size = context_output_dim

        # z-conditioned actor-critic (SAC backbone)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        # current belief q(z|c): mean, var, and a sample
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.ones(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        # reset belief to the prior N(0, I) and drop collected context
        self.z_means = ptu.zeros(num_tasks, self.latent_dim)
        self.z_vars = ptu.ones(num_tasks, self.latent_dim)
        self.sample_z()
        self._context = None

    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    @property
    def context(self):
        return self._context

    def update_context(self, inputs):
        # append one online transition to the running context (called during rollout)
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info.get('sparse_reward', r)
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        data = torch.cat([o, a, r, no], dim=2) if self.use_next_obs_in_context \
            else torch.cat([o, a, r], dim=2)
        self._context = data if self._context is None \
            else torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        # each transition -> Gaussian factor (mu, sigma^2); fuse by product
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder_output_size)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])   # positive variance
        z_params = [_product_of_gaussians(m, s)
                    for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        self.z_means = torch.stack([p[0] for p in z_params])
        self.z_vars = torch.stack([p[1] for p in z_params])
        self.sample_z()

    def sample_z(self):
        # reparameterized sample from q(z|c) so gradients flow into the encoder
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s))
                      for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        self.z = torch.stack([d.rsample() for d in posteriors])

    def adapt(self):
        # task inference from accumulated context (called after exploration)
        if self._context is not None:
            self.infer_posterior(self._context)

    def compute_kl_div(self):
        # KL( q(z|c) || N(0, I) ): the information bottleneck on z
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var))
                      for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kls = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        return torch.sum(torch.stack(kls))

    def get_action(self, obs, deterministic=False):
        # condition the policy on the current sampled z (held fixed over an episode)
        in_ = torch.cat([ptu.from_numpy(obs[None]), self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def forward(self, obs, context):
        # infer z from context, condition policy on (s, z) for a batch
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in task_z], dim=0)
        in_ = torch.cat([obs, task_z.detach()], dim=1)   # actor sees z but doesn't shape it
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
        return policy_outputs, task_z

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.vf, self.target_vf]


class MetaRLAlgorithm:
    """Decoupled-sampler off-policy meta-training of the agent."""

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent, self.env = agent, env
        self.train_tasks, self.eval_tasks = train_tasks, eval_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.config = config
        self.sampler = InPlacePathSampler(env=env, policy=agent,
                                          max_path_length=config['max_path_length'])

        self.batch_size = config.get('batch_size', 256)
        self.meta_batch = config.get('meta_batch', 16)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)   # 1/entropy-temperature (SAC)
        self.kl_lambda = config.get('kl_lambda', 0.1)         # beta: bottleneck strength
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.sparse_rewards = config.get('sparse_rewards', False)
        self.use_next_obs_in_context = config.get('use_next_obs_in_context', False)
        self.embedding_batch_size = config.get('embedding_batch_size', 64)
        self.embedding_mini_batch_size = config.get('embedding_mini_batch_size', 64)
        self.num_tasks_sample = config.get('num_tasks_sample', 5)
        self.num_steps_prior = config.get('num_steps_prior', 400)
        self.num_steps_posterior = config.get('num_steps_posterior', 0)
        self.num_extra_rl_steps_posterior = config.get('num_extra_rl_steps_posterior', 400)
        self.num_train_steps_per_itr = config.get('num_train_steps_per_itr', 2000)
        self.update_post_train = config.get('update_post_train', 1)

        lr = 3e-4
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
        self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)

    def collect_initial_data(self):
        # seed every task's buffers with prior-conditioned exploration
        n = self.config.get('num_initial_steps', 2000)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(self.agent, self.env, self.sampler,
                         self.replay_buffer, self.enc_replay_buffer,
                         idx, n, 1, np.inf, add_to_enc_buffer=True, config=self.config)

    def train_iteration(self, iteration_idx):
        # collect: clear each sampled task's encoder buffer (keep context RECENT),
        # gather prior-z exploration, then posterior-z exploitation
        for _ in range(self.num_tasks_sample):
            idx = np.random.choice(self.train_tasks)
            self.env.reset_task(idx)
            self.enc_replay_buffer.task_buffers[idx].clear()
            if self.num_steps_prior > 0:
                collect_data(self.agent, self.env, self.sampler,
                             self.replay_buffer, self.enc_replay_buffer,
                             idx, self.num_steps_prior, 1, np.inf, config=self.config)
            if self.num_steps_posterior > 0:
                collect_data(self.agent, self.env, self.sampler,
                             self.replay_buffer, self.enc_replay_buffer,
                             idx, self.num_steps_posterior, 1, self.update_post_train,
                             config=self.config)
            if self.num_extra_rl_steps_posterior > 0:
                collect_data(self.agent, self.env, self.sampler,
                             self.replay_buffer, self.enc_replay_buffer,
                             idx, self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                             add_to_enc_buffer=False, config=self.config)

        for _ in range(self.num_train_steps_per_itr):
            indices = np.random.choice(self.train_tasks, self.meta_batch)
            self._take_step(indices)
        return {}

    def _take_step(self, indices):
        mb = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb
        # CONTEXT from recently collected data, DISTINCT from the RL batch below
        context_batch = sample_context_from_buffer(
            self.enc_replay_buffer, indices, self.embedding_batch_size,
            sparse_rewards=self.sparse_rewards,
            use_next_obs_in_context=self.use_next_obs_in_context)
        self.agent.clear_z(num_tasks=len(indices))
        for i in range(num_updates):
            context = context_batch[:, i * mb: i * mb + mb, :]
            self._update(indices, context)
            self.agent.detach_z()   # truncate backprop through the shared z

    def _update(self, indices, context):
        num_tasks = len(indices)
        # RL batch: DECORRELATED transitions from the ENTIRE replay buffer (off-policy)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)

        # z carries gradients into the encoder ONLY through the Q-loss
        q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
        q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
        v_pred = self.agent.vf(obs_flat, task_z.detach())
        with torch.no_grad():
            target_v = self.agent.target_vf(next_obs_flat, task_z)

        # KL (information bottleneck) -> encoder
        self.context_optimizer.zero_grad()
        kl_loss = self.kl_lambda * self.agent.compute_kl_div()
        kl_loss.backward(retain_graph=True)

        # critic (soft Bellman) -> Q-nets AND encoder
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step(); self.qf2_optimizer.step()
        self.context_optimizer.step()          # encoder steps with the critic

        # value: regress onto min_Q - log_pi (z detached: encoder unaffected)
        min_q = torch.min(self.agent.qf1(obs_flat, new_actions, task_z.detach()),
                          self.agent.qf2(obs_flat, new_actions, task_z.detach()))
        v_target = min_q - log_pi
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        ptu.soft_update_from_to(self.agent.vf, self.agent.target_vf, self.soft_target_tau)

        # policy: SAC actor loss (z detached) + mean/std regularizers
        policy_loss = (log_pi - min_q).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

    @property
    def networks(self):
        return self.agent.networks
```

The causal chain, end to end. I wanted off-policy efficiency for meta-training and structured fast adaptation for test, and the matching principle said those fight: test-time adaptation is on-policy, so meta-training should be on-policy too, killing data reuse. The way out was to notice that the matching only constrains *task inference*, not *control* — so I disentangled them, making the task belief an explicit latent `z` that the policy and critic condition on, training the actor-critic fully off-policy from the whole replay buffer while training the encoder on recent, near-on-policy context drawn by a *separate* sampler. I inferred `z` by amortized variational inference with a KL-to-prior that doubles as an information bottleneck, pricing the information `z` carries so it's pushed toward a minimal task-relevant statistic — the lever I'd expect to control overfitting to the training tasks, pending a β sweep on held-out tasks — and I trained that encoder from the *critic's* gradients, the candidate I argued for over reconstruction and policy-return but would still pin down by ablation. Because the context is an unordered set of transitions, I made the encoder permutation-invariant by having each transition emit a Gaussian vote and fusing them by *product* — which I checked is a closed-form Gaussian whose precision is the sum of precisions, so the belief contracts like `1/N` and behaves like Bayesian filtering that sharpens with evidence. Making `z` probabilistic gave me posterior sampling: sample a task hypothesis, commit to it for an episode for temporally extended exploration, update the belief, repeat — the mechanism that lets sparse-reward tasks be cracked at all, and the reason a deterministic `z` would fail. I built the control on soft actor-critic — `z` appended to the state, two Q-nets and their min, a slow target value net, reparameterized squashed-Gaussian actor — and detached `z` everywhere except the critic so the task representation stays clean. The decoupled samplers, the product-of-Gaussians belief, the IB-regularized critic-trained encoder, and the posterior-sampling rollout are the four pieces, and they drop into the off-policy actor-critic harness as a context encoder plus a meta-gradient step.
