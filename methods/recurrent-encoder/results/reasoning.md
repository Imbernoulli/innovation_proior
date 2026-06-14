Let me start from what actually hurts. I drop an agent into an MDP I've never seen, sampled from a family `p(M)` where the states, actions, discount and horizon are shared but the reward `R` and transition `T` differ and are hidden. I don't get to converge first and then be graded; I'm graded on the return I pile up *while I'm still figuring out which task this is*. So from the very first step I'm paying real reward to learn, and the only way to do well is to explore exactly as much as resolving my uncertainty is worth, and no more. That's a precise thing, not a vibe. If I held a posterior over which MDP I'm in and, at every step, chose the action that maximizes expected return *under that posterior over the steps I have left*, I'd be Bayes-optimal: I'd take an information-seeking detour only when the expected future payoff from clearing up my uncertainty beats the cost of the detour. That is the target. The problem is that this is hopeless to compute directly.

Let me be honest about *why* it's hopeless, because the obstacles are going to dictate the design. The clean formalization is the Bayes-adaptive MDP. Put a prior `b_0 = p(R,T)` over the unknown reward and transition functions; after seeing the history `τ_{:t} = {s_0, a_0, r_1, s_1, …, s_t}` I hold the posterior `b_t(R,T) = p(R,T | τ_{:t})`, my belief. Glue the belief onto the state to make a hyper-state `s⁺_t = (s_t, b_t)`. The hyper-state's dynamics split: the environment part is the posterior-averaged transition `E_{b_t}[T(s_{t+1}|s_t,a_t)]`, and the belief part updates *deterministically* by Bayes' rule, `b_{t+1} = p(R,T|τ_{:t+1})`, so

  T⁺(s⁺_{t+1} | s⁺_t, a_t, r_t) = E_{b_t}[ T(s_{t+1}|s_t,a_t) ] · δ(b_{t+1} = p(R,T|τ_{:t+1})),

and the reward on hyper-states is `E_{b_{t+1}}[R(s_t,a_t,s_{t+1})]`. Maximizing `J⁺(π) = E[Σ_t γ^t R⁺]` over the BAMDP horizon `H⁺` gives the Bayes-optimal policy, and the Bayes-optimal policy automatically trades off exploration and exploitation — that's the whole appeal. Notice `H⁺` can be several MDP episodes, `H⁺ = N·H`, because how aggressively I should explore depends on how much time I have left to cash in what I learn. This is a special case of a belief MDP, but with one gift: in a general POMDP the hidden thing drifts every step, whereas here the hidden thing — the task `(R,T)` — is *constant within a task*. The belief only ever sharpens; it never has to track a moving target.

So why can't I just run this? Three walls, and each one is going to force a choice. First, I don't even know the parametric form of the true `R` and `T`. Second, the belief update `p(R,T|τ_{:t})` is intractable — for deep-RL-scale `R` and `T` there's no closed-form posterior. Third, even if I magically had the belief, planning in belief space is intractable. Three intractabilities stacked on top of each other. The usual escape from the third one is posterior sampling, à la Strens and Osband: at the start of an episode, sample one hypothesis MDP from the posterior, act optimally for *that one MDP* until you resample, then update. Planning is now on a plain MDP, cheap. But this explores badly. If I sample one goal hypothesis and sprint to it, find nothing, resample another, sprint there — I revisit cells I'd already ruled out, I don't *systematically* shrink my candidate set the way a belief-conditioned plan would. Picture the gridworld with an unknown goal: the Bayes-optimal agent sweeps the still-possible cells along the route that eliminates the most candidates per step; posterior sampling darts at one sampled goal at a time. So posterior sampling buys me tractability and costs me efficiency. I want its tractability and Bayes-optimal's efficiency at once. That tension is the real problem.

Here's the lever I haven't used: I get a meta-training phase. I can sample as many related tasks from `p(M)` as I like and train offline. That changes everything, because it means I don't have to *derive* the belief update analytically — I can *learn* it. Amortize it. Instead of running a fresh intractable inference in each new task, train a network once that maps experience to a posterior, and at test time it's one forward pass. That's exactly the move the variational auto-encoder makes for ordinary latent-variable models: when the true posterior `p_θ(z|x)` is intractable, introduce a recognition network `q_φ(z|x)` and train it to approximate the posterior, amortized across all data. So the plan crystallizes: learn an inference network that reads my experience and outputs a belief, and learn a policy that conditions on that belief — both meta-trained, so that at test time there's no planning, just forward passes.

But a belief over *what*? Wall one said I don't know the parameterization of `R` and `T`, and inferring a posterior over millions of neural-net weights of a transition model would be insane anyway. Let me look harder at the structure. Across the family, `R` and `T` share almost everything — the same kind of dynamics, the same kind of reward shape — and only a *task identity* varies (which goal, which target velocity, which system parameters). So I don't need a belief over the whole functions; I need a belief over the small thing that distinguishes one task from another. Represent that with a learned low-dimensional stochastic latent `m`, and write the per-task functions as

  R_i(r_{t+1}|s_t,a_t,s_{t+1}) ≈ R(r_{t+1}|s_t,a_t,s_{t+1}; m_i),
  T_i(s_{t+1}|s_t,a_t)        ≈ T(s_{t+1}|s_t,a_t; m_i),

with `R` and `T` *shared* across tasks and only `m_i` carrying the task identity. This is a huge simplification: the shared `R` and `T` get to learn from data pooled over *all* tasks, and the only thing I do inference over per task is the little vector `m`. Now the belief is `p(m|τ_{:t})`, a posterior over a low-dimensional latent rather than over a function space. Tractable to represent, and exactly the kind of thing an amortized encoder can output.

Now I have to actually train this. What's the learning signal for the encoder `q_φ(m|τ_{:t})` and the shared model? I want the latent to capture the task, and the natural objective is: the model of the environment should explain the trajectories I actually see. So maximize `E_{ρ(M,τ)}[ log p_θ(τ_{:H⁺}) ]`, the log-likelihood of the trajectory under my learned model, where `ρ` is the distribution over tasks-and-trajectories induced by my policy. (I'll suppress the conditioning on actions to keep notation clean — the model produces trajectories given actions, which come from the policy.) But that marginal log-likelihood is intractable: `log p_θ(τ) = log ∫ p_θ(τ, m) dm`, and marginalizing over the latent task is exactly the integral I can't do. This is the same shape as the VAE's intractable evidence. So I do the same trick: lower-bound it with the encoder. Let me grind the bound out rather than quote it, because I want to see precisely which encoder it asks for. Multiply and divide by `q_φ(m|τ_{:t})` inside the integral:

  E_ρ[ log p_θ(τ_{:H⁺}) ] = E_ρ[ log ∫ p_θ(τ_{:H⁺}, m) (q_φ(m|τ_{:t}) / q_φ(m|τ_{:t})) dm ]
                          = E_ρ[ log E_{q_φ(m|τ_{:t})}[ p_θ(τ_{:H⁺}, m) / q_φ(m|τ_{:t}) ] ].

Now `log` of an expectation; Jensen pushes the concave `log` inside to get a lower bound:

  ≥ E_{ρ, q_φ(m|τ_{:t})}[ log ( p_θ(τ_{:H⁺}, m) / q_φ(m|τ_{:t}) ) ]
  = E_{ρ, q_φ}[ log p_θ(τ_{:H⁺}|m) + log p_θ(m) − log q_φ(m|τ_{:t}) ]
  = E_ρ[ E_{q_φ(m|τ_{:t})}[ log p_θ(τ_{:H⁺}|m) ] − KL( q_φ(m|τ_{:t}) || p_θ(m) ) ]
  =: ELBO_t.

There it is. A reconstruction term `E_q[log p(τ_{:H⁺}|m)]` — the decoder must explain the trajectory from the latent — and a `KL(q_φ(m|τ_{:t}) || p_θ(m))` pulling my posterior toward a prior. Standard VAE shape, but now stare at the subscripts, because two of them are doing something the plain VAE doesn't, and they're going to dictate the encoder.

First subscript: the encoder conditions on `τ_{:t}` — the *past up to now* — while the decoder is asked to explain `τ_{:H⁺}` — the *whole trajectory, including the future*. That asymmetry is deliberate and it's the soul of the thing. When the agent is standing at time `t`, all it has to infer the task from is what it's seen, `τ_{:t}`; that's the only honest input to the belief. But I want the belief to be *useful*, and a belief is only useful if it lets me predict things I *haven't* seen yet. So at training time, where I do have the full trajectory, I make the decoder reconstruct the future too: predict `s_{t+1}, r_{t+1}, …` for states I haven't visited, from `m`. That forces `m` to encode the task well enough to *generalize* — to say "the goal isn't in any cell I've checked, so by elimination it's likely over there," to predict reward at cells never visited. A plain VAE reconstructs only what it encoded; here I encode the past and decode past *and* future, which is exactly what teaches the latent to support Bayes-optimal reasoning about the unknown. The reconstruction factorizes over the trajectory, because given `m` the dynamics are Markov:

  log p(τ_{:H⁺}|m) = log p(s_0|m) + Σ_i [ log p(s_{i+1}|s_i,a_i,m) + log p(r_{i+1}|s_i,a_i,s_{i+1},m) ],

i.e. a learned initial-state term, a transition decoder `T'`, and a reward decoder `R'`, all conditioned on `m`. So the decoder is just the shared `R` and `T` I posited earlier, now wearing the hat of "VAE decoder." Lovely — the thing I need to plan with and the thing that supplies the training signal are the same object.

Second subscript: `t`. The bound `ELBO_t` is indexed by *when* I cut the past. Which `t` should I train on? If I only trained on `t = H⁺` (encode the whole episode, then reconstruct), the encoder would only ever learn to infer the task from a *complete* trajectory. But I need it to infer the task *online*, at every moment, getting steadily more confident as data arrives — that's the entire point of acting under uncertainty. So I have to train the encoder to produce a good posterior at *every* context length. Sum the bound over all `t`:

  L(φ, θ, ψ) = E_{p(M)}[ J(ψ, φ) + λ Σ_{t=0}^{H⁺} ELBO_t(φ, θ) ],

where `J` is the RL return of the policy `ψ` (which conditions on the belief) and `λ` trades the model objective against the RL objective. Now the encoder is rewarded for being right after one transition, after two, after ten — it learns the whole *trajectory* of beliefs, the online sharpening. (In practice if `H⁺` is large I'll subsample a handful of `t`'s per update rather than all of them, for compute.) For `t = 0`, before any data, the posterior is just the prior `q_φ(m) = N(0, I)` — maximal uncertainty, which is correct.

And what's the prior `p_θ(m)` inside each `KL(q_φ(m|τ_{:t}) || p_θ(m))`? I could fix it to `N(0,I)` for every `t`, but think about what `t` indexes: it's a *sequence* of beliefs, each one strictly more informed than the last. The right prior for the belief at time `t` is the belief I held at time `t−1` — that's just Bayesian updating, today's prior is yesterday's posterior. So set `p_θ(m) = q_φ(m|τ_{:t-1})` for `t ≥ 1`, with the initial prior `q_φ(m) = N(0,I)`. The `KL(q_φ(m|τ_{:t}) || q_φ(m|τ_{:t-1}))` term then penalizes the belief for jumping around — it should refine smoothly as each new transition arrives, not lurch. This makes the whole `Σ_t ELBO_t` read as one coherent online filtering objective.

Now the question this whole exercise has been circling: what is the *encoder* architecture? `q_φ(m|τ_{:t})` is a function of the entire history `τ_{:t} = (s_0, a_0, r_1, s_1, …, s_t)`, a growing, ordered sequence, and I need to produce its output *online* at every `t`. Let me think about what the belief actually depends on. The Bayes posterior `p(m|τ_{:t})` is a function of the whole ordered history — the order matters, because the belief at step `t` is the belief at step `t−1` updated by the one new transition. So the encoder needs to (a) consume a variable-length, ordered sequence, (b) update its summary in `O(1)` per new transition rather than re-reading the whole history each step, and (c) carry forward a running summary that *is* the sufficient statistic of the past for inferring `m`. Read that list again — it's the exact job description of a recurrent network. Maintain a hidden state `h_t` that gets updated as `h_t = f(h_{t-1}, x_t)` from the previous hidden state and the newly embedded transition; `h_t` is my running compression of `τ_{:t}`; one cheap update per step; the prior-becomes-posterior structure I wanted falls right out of "carry the hidden state forward." The recurrent hidden state literally *is* the online belief accumulator. So encode the past trajectory with an RNN, and read the latent-task parameters off its hidden state. This is the same recurrence that RL² and Wang et al. use to do model-free meta-RL — feed the network the stream of `(state, action, reward)` and let adaptation live in the recurrent dynamics — and I'm reusing it as the inference engine.

Let me make sure I'm not reaching for recurrence out of habit, by checking the obvious alternatives and watching them fail. Alternative one: re-encode the entire window from scratch at every step with a feedforward net over all `t` transitions. That's `O(t)` work per step and can't naturally take a growing input — it throws away the online structure I'm paying for. Wall. Alternative two — and this is the serious one, because it's exactly what the off-policy probabilistic-context line does — make the encoder *permutation-invariant*: pass each transition `c_n` independently through a net to get a Gaussian factor `Ψ(m|c_n) = N(μ(c_n), σ(c_n))`, then combine them as a product of Gaussians, `q(m|c_{1:N}) ∝ Π_n Ψ(m|c_n)`. This has a clean closed form and a real virtue: order-invariance means you can resample context batches from a replay buffer in any order and get the same posterior, which is exactly what you need to make the encoder play nicely inside an off-policy loop where the context data and the RL data are decoupled. But look at what order-invariance *costs* me. The product factorization treats the transitions as *conditionally independent given `m`*; the encoder is, by construction, blind to the order in which experience arrived. For pure task *identification* — "which of these fixed MDPs am I in" — independence given `m` is often fine, because the task is constant and any unordered bag of transitions pins it down. But the *belief I need for Bayes-optimal exploration* is `p(m|τ_{:t})`, an inherently sequential object: what I should do next depends on the trajectory of how my uncertainty has been shrinking, and the product-of-Gaussians encoder has thrown the sequence away. It also tends to give posterior-sampling-flavored exploration, which I already argued is less efficient than acting on the running belief. So the permutation-invariant encoder is the right tool for off-policy convenience and the wrong tool for online Bayes-optimality. I'll take the recurrent encoder and pay the price (recurrent forward/backward passes are slower, and the on-policy training that goes with it is less sample-efficient than off-policy) to get the ordered belief. That's a real, eyes-open trade, not a free lunch.

There's one more design decision I've been carrying implicitly that I should make explicit, because it's the inductive bias that separates this from plain recurrent meta-RL. RL² carries a single *deterministic* hidden vector and trains it only through the return — nothing makes that hidden state an honest representation of *which task* or *how uncertain*. But I need uncertainty: the whole Bayes-optimal story is "act on your belief *including its spread*," so the policy has to be able to read how unsure I am. So the latent `m` must be a *distribution*, not a point — a Gaussian whose variance shrinks as the task gets pinned down. That's why the encoder outputs `μ` and a (log-)variance, why the objective is a *variational* ELBO with a KL term (the KL is what gives the latent a probabilistic meaning and an uncertainty I can trust), and why I sample `m` with the reparameterization trick `m = μ + exp(½ logvar) ⊙ ε`, `ε ~ N(0,I)`, so the sampling stays differentiable and a single Monte-Carlo draw backprops cleanly. Conditioning the policy on the *posterior* `(μ, σ)` — the belief — rather than only on a sample is the direct analogue of the BAMDP hyper-state `(s, b)`: the policy sees the state and the belief. And because the latent is trained to represent *only the task* (constant within a task, by the shared-`R`,`T` construction), the belief stays put once the task is identified, which is exactly the stability that a high-dimensional deterministic recurrent hidden state lacks across episode resets — the latent doesn't have to re-infer anything when the agent is reset to the start. If I strip the decoder and the KL away entirely, this whole thing collapses back to RL²: a recurrent net trained only by the return. So the decoder-plus-KL is precisely an *auxiliary objective* bolted onto recurrent meta-RL that forces the hidden state to become a calibrated task belief, and to be predictive about unseen transitions. That framing tells me exactly what I'm adding and why.

Now a practical wall I'll hit if I'm not careful about how the losses interact. The encoder parameters `φ` are *shared* between the model (it appears in every `ELBO_t`) and the policy (the policy conditions on `q_φ`'s output). So in principle I'd backprop both the RL loss and the ELBO through `φ`. But these two gradients pull in different directions — the RL loss wants the embedding to make the *next action* good, the ELBO wants it to *reconstruct the trajectory* — and forcing one set of parameters to serve both means a delicate `λ` trade-off and gradient interference. When I check whether the RL gradient through the encoder is even necessary, it turns out it mostly isn't: the reconstruction objective alone shapes a perfectly good task representation, and not backpropagating the RL loss through the encoder both removes the interference and saves a fortune in compute. The reason it saves compute is specific and worth seeing: with an on-policy algorithm like PPO I do several minibatch passes per batch of data, and if the RL loss flowed through the recurrent encoder I'd have to re-run forward/backward through the whole RNN on every one of those passes; cutting that dependency lets me compute the embeddings once and reuse them. So I'll train the VAE (encoder + decoder) and the policy with *separate* optimizers and learning rates, and even separate data buffers — the policy on the most recent on-policy data, the VAE on a larger buffer of stored trajectories — and `λ` only matters at all because `φ` is shared, which it now barely is. Decoupling is the engineering that makes the recurrent encoder trainable at scale.

Let me also pin the small architectural choices so they're reasons, not arbitrary. The transitions are heterogeneous — a state vector, an action vector, a scalar reward, all on different scales and dimensions — so I embed each modality with its own small feature extractor before fusing them, rather than concatenating raw and hoping one linear layer sorts it out. After fusing I can put a fully-connected layer or two before the recurrent cell to mix the embeddings, then the recurrent core, then a fully-connected layer or two, then *two separate heads* for `μ` and `logvar` (two heads because the mean and the log-variance are different functions of the hidden state and shouldn't share a final projection). For the recurrent core itself: a vanilla RNN would choke on the vanishing-gradient problem over a long horizon, and an LSTM has more gates (and parameters) than I need here; a GRU is the middle choice — gated enough to carry information across the horizon, lighter than an LSTM — which is the same call RL² and Wang made, so I'll use a GRU. I'll initialize the GRU's recurrent weights orthogonally and its biases to zero, the standard recipe for keeping recurrent dynamics well-conditioned at the start of training. And the latent dimension can be *tiny* — the task variation is genuinely low-dimensional (a goal location, a target speed, a parameter vector), so a 5-dimensional latent both suffices and *concentrates*, giving a sharp, stable posterior, in contrast to a 100-plus-dimensional deterministic recurrent hidden state that has room to drift and destabilize across resets.

At meta-test time the decoder is thrown away entirely — it was only ever scaffolding to train the belief — and I just roll the policy forward: encode the streaming experience with the GRU, read off the posterior `(μ, σ)`, condition the policy, act. No gradient adaptation, no planning. The policy learned during meta-training to act approximately Bayes-optimally as a function of the belief, so test-time "learning" is just the belief sharpening inside the recurrent state as data comes in.

Let me write the encoder I'd actually ship, filling the one empty slot — the experience-to-belief module. Per-modality embeddings, the GRU core, the two Gaussian heads, the reparameterized sample, an online forward that consumes one transition and the previous hidden state, and a prior path for `t = 0`:

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    """Embed one modality (state / action / reward) before fusing them."""
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.activation = activation
        self.fc = nn.Linear(input_size, output_size) if output_size != 0 else None

    def forward(self, x):
        return self.activation(self.fc(x)) if self.fc is not None else torch.zeros(0)


class RNNEncoder(nn.Module):
    """Recurrent belief encoder q_phi(m | tau_{:t}).

    Consumes the ordered (s, a, r) stream, carries a GRU hidden state as the
    online sufficient statistic of the past, and reads the latent-task
    posterior (mean, log-variance) off the recurrent state at every step.
    """
    def __init__(self,
                 layers_before_gru=(), hidden_size=64, layers_after_gru=(),
                 latent_dim=5,
                 action_dim=2, action_embed_dim=16,
                 state_dim=2, state_embed_dim=32,
                 reward_size=1, reward_embed_size=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # per-modality embeddings: different scales/dims, so embed separately
        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # optional FC mixing before the recurrent cell
        curr = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList(
            [nn.Linear(curr if i == 0 else layers_before_gru[i - 1], layers_before_gru[i])
             for i in range(len(layers_before_gru))])
        if layers_before_gru:
            curr = layers_before_gru[-1]

        # recurrent core: GRU -- gated (handles long horizons), lighter than LSTM
        self.gru = nn.GRU(input_size=curr, hidden_size=hidden_size, num_layers=1)
        for name, p in self.gru.named_parameters():       # standard RNN init
            if 'bias' in name:
                nn.init.constant_(p, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(p)

        # optional FC after the recurrent cell
        curr = hidden_size
        self.fc_after_gru = nn.ModuleList(
            [nn.Linear(curr if i == 0 else layers_after_gru[i - 1], layers_after_gru[i])
             for i in range(len(layers_after_gru))])
        if layers_after_gru:
            curr = layers_after_gru[-1]

        # two separate heads: mu and log-variance are different functions of h
        self.fc_mu = nn.Linear(curr, latent_dim)
        self.fc_logvar = nn.Linear(curr, latent_dim)

    def _sample_gaussian(self, mu, logvar):
        # reparameterisation: m = mu + exp(0.5*logvar) * eps, differentiable in phi
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def prior(self, batch_size, device, sample=True):
        # t = 0: no data yet -> belief is the prior, read off a zero hidden state
        h = torch.zeros((1, batch_size, self.hidden_size), device=device)
        for fc in self.fc_after_gru:
            h = F.relu(fc(h))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        m = self.reparameterise(mu, logvar) if sample else mu
        return m, mu, logvar, torch.zeros((1, batch_size, self.hidden_size), device=device)

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True):
        # inputs shaped (seq_len, batch, dim); for one online step seq_len == 1
        if return_prior:
            m0, mu0, lv0, hidden_state = self.prior(actions.shape[1], actions.device)

        # embed each modality, then fuse
        h = torch.cat((self.action_encoder(actions),
                       self.state_encoder(states),
                       self.reward_encoder(rewards)), dim=2)
        for fc in self.fc_before_gru:
            h = F.relu(fc(h))

        # carry the hidden state forward: h_t = GRU(embed(transition), h_{t-1})
        output, _ = self.gru(h, hidden_state)
        gru_h = output
        for fc in self.fc_after_gru:
            gru_h = F.relu(fc(gru_h))

        # read the latent-task posterior off the recurrent state
        mu, logvar = self.fc_mu(gru_h), self.fc_logvar(gru_h)
        m = self.reparameterise(mu, logvar) if sample else mu

        if return_prior:                  # prepend the t=0 prior belief
            m = torch.cat((m0, m)); mu = torch.cat((mu0, mu)); logvar = torch.cat((lv0, logvar))
            output = torch.cat((hidden_state, output))
        return m, mu, logvar, output
```

And to fill the concrete encoder slot of a PEARL-style harness — where the surrounding loop feeds ordered context as `(task, seq, feat)` and the agent turns the encoder's output into the posterior via `softplus` on the variance and a product of Gaussians — I collapse the per-modality embedders into one per-transition MLP, take the last recurrent state, and emit `2·latent_dim` numbers (`μ` and pre-softplus `σ²`) in one head; the recurrence is the only structural change from the per-transition baseline:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule


def _identity(x):
    return x


class CustomContextEncoder(PyTorchModule):
    """Recurrent belief encoder filling the per-transition encoder slot."""
    IS_RECURRENT = True

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu,
                 output_activation=_identity, hidden_init=ptu.fanin_init,
                 b_init_value=0.1, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size              # 2 * latent_dim (mu and sigma^2)
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # per-transition MLP embedding
        in_size = input_size
        self.fcs = []
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        # recurrent core over the ordered context; hidden state = running belief
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                            num_layers=1, batch_first=True)

    def forward(self, input, return_preactivations=False):
        task, seq, feat = input.size()              # ordered context (task, seq, feat)
        out = input.view(task * seq, feat)
        for fc in self.fcs:                         # embed each transition
            out = self.hidden_activation(fc(out))
        out = out.view(task, seq, -1)

        if self.hidden.size(1) != task:             # match hidden to current task dim
            self.reset(task)
        zeros = torch.zeros(self.hidden.size()).to(ptu.device)
        out, (hn, cn) = self.lstm(out, (self.hidden, zeros))
        self.hidden = hn
        out = out[:, -1, :]                         # last hidden state -> belief

        preactivation = self.last_fc(out)           # -> (mu, sigma^2) params
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
```

Let me trace the whole chain back so I'm sure it hangs together. I wanted return-during-learning, whose principled optimum is the Bayes-optimal BAMDP policy that conditions on a belief over the task — but the BAMDP is triply intractable (unknown parameterization, intractable belief update, intractable belief-space planning), and the usual escape, posterior sampling, explores inefficiently. The meta-training distribution let me *learn* the belief instead of deriving it: amortize the intractable posterior with an inference network, the VAE move. I shrank the belief from "posterior over `R`,`T`" to "posterior over a low-dimensional latent task `m`" by sharing `R` and `T` across tasks, which made the inference tractable and let the shared model pool data over all tasks. Lower-bounding the intractable trajectory log-likelihood gave a per-context-length ELBO — reconstruction plus KL — and two subscripts in it did the real work: encode the *past* but decode past *and future* (so the belief learns to generalize to unseen states), and index the bound by every `t` with the prior set to the previous posterior (so the belief is trained to refine online). The belief is `p(m|τ_{:t})`, a function of an ordered, growing history that I must update online — which is exactly a recurrent network's job, so the encoder is a GRU whose hidden state is the running belief accumulator, with per-modality embeddings in and two Gaussian heads out; the rejected permutation-invariant product-of-Gaussians encoder is order-blind and only good for off-policy convenience. Making the latent a *distribution* (Gaussian, reparameterized, KL-regularized) is the inductive bias that turns a deterministic recurrent meta-RL net into a calibrated, uncertainty-aware task belief, and the small latent keeps that belief sharp and stable across resets. Not backpropagating the RL loss through the shared encoder removes gradient interference and the cost of re-encoding for every PPO minibatch, so the VAE and policy train with separate optimizers and buffers. At test time the decoder is discarded and the policy acts on the streaming belief with no planning — tractable, like posterior sampling, but conditioned on the running belief, so it explores approximately Bayes-optimally.
