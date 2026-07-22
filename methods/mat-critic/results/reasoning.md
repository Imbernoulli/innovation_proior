Let me start from what actually goes wrong when I train a team of agents. I have `n` agents in a cooperative Markov game, each seeing a local observation `o^i`, each picking `a^i`, and the whole team gets one shared scalar reward `R(o, a)`. I want to maximize the joint return `J(π) = E[Σ_t γ^t R(o_t, a_t)]`. The single-agent toolbox is solid: TRPO gives me a monotonic-improvement bound, `J(π̄) ≥ L_π(π̄) − C·D_KL^max(π,π̄)` with `L_π(π̄) = J(π) + E_{s∼ρ_π, a∼π̄}[A_π(s,a)]`, so maximizing the surrogate inside a small KL ball cannot decrease `J`; and PPO turns that into a cheap first-order clip, `E[min(rA, clip(r,1±ε)A)]`. So I know how to improve *one* policy with a guarantee. The trouble is that "the team" is not one policy, it's `n` of them, and I have two intertwined jobs: find an improvement direction for each agent, and combine those directions so the *joint* return actually rises. Those are different jobs, and the gap between them is where everything breaks.

The naive thing is to give all agents the same network — one shared `θ`, aggregate everyone's trajectories, optimize `Σ_i E[min(r^i A_π, clip(r^i,1±ε) A_π)]`. This is sample-efficient when the agents are interchangeable; with 25 identical marines each step trains the shared net on 25× the data. But forcing `θ^i = θ^j` for all `i,j` is a hard constraint on the joint policy space, and when the optimal joint behavior is genuinely heterogeneous it caps me, sometimes catastrophically. Picture the one-state game on `{0,1}^n` where reward is 1 only for `(0^{n/2}, 1^{n/2})` and `(1^{n/2}, 0^{n/2})` and 0 otherwise. The optimum needs *half* the agents to do one thing and half the other — pure heterogeneity. Under shared parameters every agent has the same action distribution, so the best a symmetric policy can do is hit a winning pattern only by coincidence: probability `2·[p(1−p)]^{n/2}` of landing on the two good patterns under a shared Bernoulli(`p`) policy, maximized at `p = 1/2` to give `2·(1/4)^{n/2} = 2/2^n`, versus the optimum's 1 — a ratio `2/2^n` that decays exponentially in `n`. So parameter sharing is not just inconvenient, it's provably crippling on heterogeneous tasks. Wall.

Fine, drop the sharing and give each agent its own parameters, each running its own PPO with a centralized critic for variance reduction — the COMA/MADDPG style, critic sees the joint action and global state, actors stay decentralized. Now I can represent heterogeneous policies. But I've lost the *guarantee*, and I can show it's not a technicality. Take two agents with one-dimensional Gaussian policies, means `μ_1, μ_2`, unit variance, and reward `r(a_1, a_2) = a_1 a_2`, so the expected one-step reward is `μ_1 μ_2`. Start at `μ_1 = -0.25, μ_2 = 0.25`, giving return `-0.0625`. Agent 1, holding agent 2 fixed at `+0.25`, improves reward by increasing `μ_1`; moving to `μ_1 = 0.75` would give `0.75 · 0.25 = 0.1875`. Agent 2, holding agent 1 fixed at `-0.25`, improves reward by decreasing `μ_2`; moving to `μ_2 = -0.75` would give `(-0.25) · (-0.75) = 0.1875`. For unit-variance Gaussians those moves are inside a KL ball of radius `0.5`, since `D_KL(N(μ,1), N(μ+1,1)) = 1/2`. But if both locally attractive moves happen simultaneously, the new joint return is `0.75 · (-0.75) = -0.5625`, worse than where I started. Each update was good against the old partner, and their combination is bad. That's the heart of the problem stated as cleanly as I can state it: simultaneous independent ascent has no claim on joint improvement. So independence buys me expressiveness and loses me the very guarantee that made the single-agent method trustworthy.

And there's the credit-assignment ache underneath all of this. One shared reward, `n` agents — how does any single agent know whether the team won *because of* it or *despite* it? If I just feed the joint value into a multi-agent policy gradient, the variance of each agent's estimate grows with `n`, because each agent's gradient is contaminated by the noise of everyone else's exploration. People have patched this with local value functions and counterfactual baselines, but those are patches on a monolithic critic — an MLP that eats the joint input and never represents *how the agents relate*. I keep coming back to the feeling that the structure of the problem — who depends on whom — is being thrown away and then approximately reconstructed.

So let me get precise about the object. Define the multi-agent observation-value function for an ordered subset of agents, `Q_π(o, a^{i_{1:m}}) = E[R^γ | o_0 = o, a^{i_{1:m}}_0 = a^{i_{1:m}}]` — the expected return when agents `i_1, …, i_m` are pinned to actions `a^{i_{1:m}}` and everyone else acts under `π`. When `m = n` this is the ordinary `Q`; when `m = 0` it's the ordinary `V`. And the multi-agent advantage, `A^{i_{1:m}}_π(o, a^{j_{1:h}}, a^{i_{1:m}}) = Q^{j_{1:h}, i_{1:m}}_π − Q^{j_{1:h}}_π`: how much better than average it is for agents `i_{1:m}` to take `a^{i_{1:m}}` *given that* `j_{1:h}` have already committed to `a^{j_{1:h}}`. The "given that" is the part I want to lean on, because the differential-game disaster was precisely about agents acting in ignorance of each other.

I need to know whether the *joint* advantage can be broken into *individual* advantages. If it can, making the joint advantage positive reduces to making a chain of single-agent conditioned advantages positive, and single-agent positivity I know how to chase. Let me just try to telescope it. Fix any ordering `i_1, …, i_m` and look at `A^{i_{1:m}}_π(o, a^{i_{1:m}}) = Q^{i_{1:m}}_π(o, a^{i_{1:m}}) − V_π(o)`. Now insert a telescoping chain of intermediate `Q`s, growing the pinned set one agent at a time:

```
A^{i_{1:m}}(o, a^{i_{1:m}}) = Q^{i_{1:m}}(o, a^{i_{1:m}}) − V(o)
  = Σ_{k=1}^{m} [ Q^{i_{1:k}}(o, a^{i_{1:k}}) − Q^{i_{1:k-1}}(o, a^{i_{1:k-1}}) ]
```

because the sum telescopes: every interior `Q^{i_{1:k}}` cancels, leaving `Q^{i_{1:m}}` at the top and `Q^{i_{1:0}} = V` at the bottom. And each bracket is, by the definition above, exactly a single-agent conditioned advantage: `Q^{i_{1:k}}(o, a^{i_{1:k}}) − Q^{i_{1:k-1}}(o, a^{i_{1:k-1}}) = A^{i_k}_π(o, a^{i_{1:k-1}}, a^{i_k})` — agent `i_k`'s advantage *given its predecessors `i_{1:k-1}` already played `a^{i_{1:k-1}}`*. So

```
A^{i_{1:n}}_π(o, a^{i_{1:n}}) = Σ_{m=1}^{n} A^{i_m}_π(o, a^{i_{1:m-1}}, a^{i_m}).
```

The telescoping is mechanical, so on the page it looks like it has to hold for *any* permutation, *any* state, *any* joint action, with *no* assumption that the value decomposes. But I derived it by symbol-pushing, and the claim I actually care about — that the *same* joint advantage falls out term-by-term regardless of which order I pick — is exactly the kind of thing that is easy to convince yourself of and wrong about. So before I build anything on it, let me put numbers through it on a small game I can compute by hand.

Take three agents, each with two actions, one state. Give them a base policy — `π^1 = (0.6, 0.4)`, `π^2 = (0.3, 0.7)`, `π^3 = (0.5, 0.5)` — and a reward over the eight joint actions that is deliberately *not* a sum of per-agent terms: a random value per joint action plus pairwise couplings `2 a_1 a_2 − 1.5 a_2 a_3`, so agents genuinely interact. I compute `Q^{S}(a_S)` by pinning the agents in `S` to `a_S` and averaging the reward over the rest under the base policy, and `V = Q^{∅}` by averaging over all three. For the joint action `(a_1, a_2, a_3) = (0, 1, 1)` I get `V = 0.02332` and a joint advantage `A^{1:3} = Q^{full} − V = -2.00532`. Now I telescope along the order `(3, 1, 2)`: agent 3's conditioned advantage `Q^{\{3\}} − V`, then agent 1's `Q^{\{3,1\}} − Q^{\{3\}}`, then agent 2's `Q^{\{3,1,2\}} − Q^{\{3,1\}}`, which come out `(-0.756, -0.500, -0.749)` and sum to `-2.00532`. That matches the joint advantage to all the digits I carried. And the order-independence — the part I was most suspicious of — holds too: redoing it along `(1, 2, 3)` gives entirely *different* per-agent terms `(-0.592, -0.502, -0.912)`, but they sum to the same `-2.00532`. So the per-agent attributions shuffle with the order while their total is pinned to the joint advantage. The identity is real, and it is real even when the reward is non-additive, which is the only case I care about.

That changes the team problem. The joint advantage is the sum of per-agent advantages *as long as each agent's advantage is conditioned on the actions of the agents before it in the order*. The differential game failed because the agents conditioned on *nothing*. Here, if agent `i_m` knew `a^{i_{1:m-1}}` and picked an `a^{i_m}` with nonnegative conditioned advantage, every summand would be nonnegative, so their sum — the joint advantage — would be nonnegative. That is the chain I was missing: joint improvement reduces to a sequence of conditioned single-agent improvements, and single-agent positivity is the one thing I already know how to chase.

There's a second prize hidden in here, the complexity one. Searching the joint action space directly means browsing `∏_i |A^i|` joint actions — multiplicative, hopeless for many agents. But the decomposition says I can build a good joint action *incrementally*: agent `i_1` searches its own `|A^{i_1}|` actions for a positive advantage, then `i_2` searches its `|A^{i_2}|` *conditioned on `i_1`'s choice*, and so on. The total search is `Σ_i |A^i|` — additive. The exponential blow-up collapses to linear in the number of agents, just by making the decision sequential.

Now, one line of work already runs with this: draw a random permutation each iteration and update the agents *one at a time* in that order, each agent `i_m` maximizing its clip objective `E[min(r^{i_m} A^{i_{1:m}}, clip(r^{i_m}, 1±ε) A^{i_{1:m}})]` with the expectation taken under the *already-updated* policies of `i_{1:m-1}`. The no-op policy is always available, so the conditioned trust-region term for agent `i_m` can be kept at zero; choosing a policy with a nonnegative conditioned term cannot hurt that term. Add those nonnegative terms through the telescoping identity I just checked numerically, keep the TRPO-style penalty under the per-agent KL constraints, and the lower bound on the joint return is nondecreasing. Randomizing the order makes the limit point a Nash equilibrium — at convergence no agent is incentivized to deviate, because every order has a chance to be the one that would have moved it. So this carries the monotone-improvement structure I lost when I went independent, and it handles heterogeneous agents because each has its own conditioned objective. But it has a wall of its own: it is *strictly sequential*. Agent `i_m` literally cannot be optimized until `i_1, …, i_{m-1}` have finished updating, because it needs their *new* policies for the importance-sampling term. With `n` agents that's `n` dependent stages per iteration; the wall-clock cost grows with the number of agents and you can't parallelize it away. And there's a softer dissatisfaction too: each agent is still optimizing a separately handcrafted objective; nothing in the model *is* the team. The cooperation lives in the math around the agents, not in the agents themselves.

Let me sit with the shape of what the decomposition is telling me. Agent `i_m`'s action depends on the joint observation `o^{i_{1:n}}` and on its predecessors' actions `a^{i_{1:m-1}}`. So I'm mapping a sequence of observations `(o^{i_1}, …, o^{i_n})` to a sequence of actions `(a^{i_1}, …, a^{i_n})`, where output element `m` is allowed to depend on the whole input and on output elements `1..m-1` — but explicitly *not* on `m..n`. Where else does that dependency pattern show up? It is the dependency of auto-regressive sequence-to-sequence generation: translate an input sequence into an output sequence, each output token conditioned on the full input and on the previously generated output tokens, and on nothing later. So the team need not be a set I coordinate by hand-built objectives; it can be treated as a *sequence* generated one agent at a time. If I want a model whose forward pass enforces "output `m` may read all inputs and outputs `< m`," I want an architecture with exactly that wiring: read the whole input sequence (no ordering constraint on reading), and emit outputs causally. The encoder-decoder Transformer is built to do precisely this — full self-attention in the encoder, *masked* self-attention in the decoder so output `m` attends only to outputs `< m`, plus cross-attention into the encoder. Its causal mask is not a loose analogy to the predecessor constraint; it is the same constraint — "agent `i_m` may see only `i_{1:m-1}`" — so the masking does the conditioning the decomposition requires rather than approximating it.

Sequence models have two modes, and that matters here. At *inference* I must generate auto-regressively, because agent `i_m`'s action genuinely isn't known until I've sampled `i_{1:m-1}`. But at *training* the predecessors' actions are already sitting in the replay buffer, so I can feed all the ground-truth shifted actions in at once and let the causal mask enforce the conditioning, computing every agent's clipped likelihood ratio in a *single parallel forward pass*. The monotonic argument is the same trust-region argument as the sequential scheme in the way that matters for the model — each agent's ratio is conditioned on the predecessor actions required by the decomposition — but the implementation no longer waits for previous agents' parameter updates or uses their newly updated action distributions inside the importance-sampling calculation. The model outputs all agents' objectives together, and one optimizer step updates the encoder-decoder parameters.

So let me build it. Start with the encoder, parameters `φ`. Its job is to turn the `n` per-agent observations into representations that carry not just each agent's own information but the *interactions* between agents — because the joint advantage and the value depend on the whole team, not one agent in isolation. Each agent's observation is a token; I project it up to a model dimension, then push the `n` tokens through blocks of self-attention. The self-attention here must be *unmasked* — full attention across the agent axis — because building an interaction-aware representation of agent `i` should be allowed to look at *every* other agent, not just earlier ones; there's no causal ordering on *reading* observations, only on *emitting* actions. Each block is the standard sublayer pattern: attention, then a position-wise MLP, each wrapped in a residual connection and layer norm, `x ← LayerNorm(x + Attn(x)); x ← LayerNorm(x + MLP(x))`. Residual connections so I can stack depth without the signal degrading or the gradient vanishing; layer norm to keep the per-token activations well-scaled through the blocks. The attention itself is scaled dot-product: weights `w(q^{i_r}, k^{i_j}) = ⟨q^{i_r}, k^{i_j}⟩ / sqrt(d_k)`, softmaxed, applied to values. The `1/sqrt(d_k)` factor — let me make sure I actually need it and that `sqrt(d_k)` is the right amount, rather than copying it in. Suppose `q` and `k` have independent coordinates with mean 0 and variance 1. The raw score is `q·k = Σ_{l=1}^{d_k} q_l k_l`. Each product `q_l k_l` has mean `E[q_l]E[k_l] = 0` and variance `E[q_l^2]E[k_l^2] − 0 = 1·1 = 1`, and the `d_k` terms are independent, so `var(q·k) = Σ_l 1 = d_k` and its standard deviation is `sqrt(d_k)`. At a plausible width — say `d_k = 64` — the logits feeding softmax would have spread `±8` before they even touch any learned scaling, which puts softmax deep in its saturated corner where one weight is ≈1 and the rest ≈0, and there the gradient of softmax is ≈0, so almost no signal flows back. Dividing the score by `sqrt(d_k)` turns `var = d_k` into `var(q·k / sqrt(d_k)) = d_k / d_k = 1`, i.e. spread `±1`, which is exactly the regime where softmax still has usable gradients. So the `sqrt(d_k)` is the unique scale that cancels the width dependence; it earns its place.

On top of the encoder's final representation I hang a small value head — Linear, GELU, layer norm, Linear to a scalar — producing a per-agent value `V_φ(ô^{i_m})`. Why make the *encoder* predict the value, rather than bolting on a separate critic network? Two reasons that reinforce each other. First, forcing the shared representation to be good enough to predict returns is exactly the pressure that makes it expressive — a representation that can estimate the value of the joint situation has had to encode the interactions. Second, I need a joint value anyway to compute advantages, and the cleanest joint-value estimate from per-agent values is their average, `V̂_t = (1/n) Σ_m V_φ(ô^{i_m}_t)`; feeding that through GAE gives me the joint advantage `Â_t` that the policy side will clip against. So the encoder is simultaneously the interaction model and the centralized critic. The clean objective is the empirical Bellman error on each per-agent value,

```
L_Encoder(φ) = (1/Tn) Σ_{m=1}^{n} Σ_{t=0}^{T-1} [ R(o_t, a_t) + γ V_φ̄(ô^{i_m}_{t+1}) − V_φ(ô^{i_m}_t) ]²,
```

with a target network `φ̄` (a periodically-frozen copy) supplying the bootstrap target, so the regression target doesn't chase the parameters and destabilize. In the on-policy code path the buffer already carries return targets and old value predictions, so the implemented value update is the PPO-style value regression to those returns, optionally with value clipping, value normalization, Huber loss, and active-agent masks.

Now the decoder, parameters `θ`. It consumes the encoder's representations and the *shifted* sequence of joint actions and emits, for each agent in order, the policy `π^{i_m}_θ(a^{i_m} | ô^{i_{1:n}}, a^{i_{1:m-1}})`. "Shifted" because output `m` must be conditioned on actions `1..m-1`, not on its own action — so I feed in the previous agents' actions offset by one position, and I need a start-of-sequence symbol for the very first agent `i_1`, which has no predecessor. Concretely I embed each action and prepend a start symbol: for the discrete case the action embedding is a linear map from a one-hot of width `action_dim + 1`, where the extra `+1` slot is the start token (`a^{i_0}` is the one-hot `[1, 0, …, 0]`). The plan is "input to position 0 is the start symbol, and position `m` carries the one-hot of agent `i_{m-1}`'s action," and I want to be sure the index arithmetic I'm about to write actually produces that, because an off-by-one here silently leaks each agent its own action — which would make the importance ratios meaningless. So let me trace it for `n = 3`, `action_dim = 2`, with the buffered actions `a^{i_1} = 1, a^{i_2} = 0, a^{i_3} = 1`. One-hot those into a `(3, 2)` block `[[0,1],[1,0],[0,1]]`. Allocate `shifted` of shape `(3, 3)` (width `action_dim + 1`), set `shifted[0,0] = 1` for the start token, and write `shifted[1:, 1:] = one_hot[:-1]` — i.e. the first `n-1` one-hots, dropping the last, slotted into positions `1..n-1`. That yields rows `[[1,0,0], [0,0,1], [0,1,0]]`. Reading them back: position 0 = start token; position 1 = `[0,0,1]`, which is `i_1`'s action `1` sitting in the action slots; position 2 = `[0,1,0]`, which is `i_2`'s action `0`. So position `m` does hold `i_{m-1}`'s action and the last agent's own action never appears anywhere in the input. The shift is right. Each decode block then does masked self-attention over the action tokens, then a second masked attention whose query is the encoder representation and whose keys/values are the masked action stream, then an MLP — all with residuals and layer norm. The masks are lower-triangular: position `r` attends only to positions `j ≤ r`, so combined with the shift, agent `i_m` (reading at position `m`, which carries up to `i_{m-1}`) depends only on `a^{i_{1:m-1}}` — `w(q^{i_r}, k^{i_j}) = 0` for `r < j`, the literal "may not see the future." A final head maps to action logits (discrete) or means (continuous).

The decoder's training objective is the PPO clip on the *joint* advantage, summed over agents and time:

```
L_Decoder(θ) = −(1/Tn) Σ_{m=1}^{n} Σ_{t=0}^{T-1} min( r^{i_m}_t(θ) Â_t,  clip(r^{i_m}_t(θ), 1±ε) Â_t ),
   r^{i_m}_t(θ) = π^{i_m}_θ(a^{i_m}_t | ô^{i_{1:n}}_t, a^{i_{1:m-1}}_t) / π^{i_m}_{θ_old}(a^{i_m}_t | ô^{i_{1:n}}_t, a^{i_{1:m-1}}_t).
```

I should pause on why the joint advantage `Â_t` is the practical signal to multiply each agent's ratio by, rather than building a separate estimator for every conditioned advantage. The theorem's exact summands are conditioned advantages, and the decoder's mask makes the likelihood ratio carry that same conditioning: `r^{i_m}` is the ratio for agent `i_m`'s action given its predecessors. So the model is not treating `n` independent PPO terms as if they magically coordinate; every term is a predecessor-conditioned change of the joint policy. I still estimate the scalar advantage robustly with GAE on the joint value, because trying to regress all conditioned advantage tables would bring back the combinatorial object I am trying to avoid. The trust-region argument then has the right place to attach: the policy change for `i_m` is measured under the predecessor-conditioned distribution required by the decomposition, and the clipped surrogate is the tractable PPO version of that constrained step. Every `r^{i_m}` is computed in the *same* parallel pass because the ground-truth `a^{i_{1:m-1}}` are in the buffer and the mask does the conditioning.

I keep one more requirement from the guarantee: I must *permute the agent order randomly every iteration*. The monotonic bound holds for any fixed order, but if I always used the same order, the limit policy could be a fixed point of that order and not a true equilibrium — some agent might still want to deviate if it could go first. Randomizing the order so every permutation has positive probability of leading is what makes the limit a Nash equilibrium: at convergence no agent is incentivized to change, because it had a chance to be the one that moved and didn't.

One choice I would have reflexively imported from sequence models is positional encoding — add a vector encoding "you are token at position `p`" so attention can use order. Let me think about whether it even makes sense here. The intended training protocol shuffles agents into a fresh permutation, which is the ordering condition behind the Nash result. So an agent's position index in the sequence carries no consistent meaning across iterations; the same physical agent can appear at position 3 this iteration and position 7 the next. Encoding position would inject noise — the model would try to read structure off an index that is deliberately randomized. What I actually want the model to key on is *which agent* a token is, consistently — its role, its action space, its identity. So the stable identity information belongs in the observation/state features as a one-hot agent id rather than as a Transformer position vector. Agent-id encoding is consistent across shuffling; positional encoding fights it, and once the stable identity is present the random position index has no useful semantic job left. So: agent-id encoding, drop positional encoding entirely.

A couple of sizing choices fall out of the regime. The cooperative tasks have modest agent counts, so I don't need a deep stack or many heads — a single block and a single head is the default, with `n_embd = 64`, and it is cheaper than a large language-style stack. The implementation keeps the feed-forward block compact, `Linear(d,d) → GELU → Linear(d,d)`, because the agent sequence is short; the residual/LN placement stays because those are what keep attention blocks trainable. The whole thing trains under one Adam optimizer over the transformer, total loss `policy_loss - entropy_coef·entropy + value_loss_coef·value_loss`, single backward, optional gradient-norm clip, advantages from GAE with the usual normalization `(Â − mean)/(std + 1e-5)` over active agents.

Let me also be honest about a CTDE sanity check I can run in my head. If I keep the encoder (so I keep the interaction-aware representation and the centralized value) but *replace the decoder with a fully decentralized per-agent actor* — each agent maps its own representation to its own action with no cross-agent action conditioning, and I train it on the *local* advantage instead of the joint one — I get a method that still uses the powerful centralized critic but throws away the auto-regressive coupling. That isolates the value of the decoder: if the full encoder-decoder improves over this decentralized-actor variant, the gain is specifically the auto-regressive, predecessor-conditioned action generation, not just the attention critic. I'd want exactly that comparison to confirm the decoder is doing real work and not just riding the encoder. Similarly, swapping the encoder-decoder for encoder-only (no auto-regression) or decoder-only or a GRU sequence model would tell me whether attention and the encoder-decoder split are each necessary. Those are the validations I'd run before trusting the design.

Now let me put the model into the code I'd actually ship, filling the one empty slot — the joint observation→action/value model. The encoder builds interaction-aware representations and the per-agent value; the decoder generates actions auto-regressively at inference and in parallel during training, with masked attention implementing the predecessor conditioning.

```python
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.constant_(m.bias, 0) if m.bias is not None else None
    return m


class SelfAttention(nn.Module):
    """Scaled dot-product multi-head attention over the AGENT axis.
    masked=True applies a causal mask so agent i_m sees only predecessors i_{1:m-1}."""

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # lower-triangular causal mask over the agent sequence (+1 for the start token)
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

    def forward(self, key, value, query):
        B, L, D = query.size()
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        # w(q,k) = <q,k> / sqrt(d_k): the 1/sqrt(d_k) keeps softmax out of the vanishing-grad regime
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:                                   # r < j entries set to -inf
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class EncodeBlock(nn.Module):
    """Full (unmasked) self-attention across agents + MLP, residual + LayerNorm."""

    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                 nn.GELU(), init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))              # read every other agent
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """Masked self-attention over actions + masked cross-attention into the encoder rep + MLP."""

    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(n_embd) for _ in range(3))
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                 nn.GELU(), init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))             # action self-attn, predecessors only
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))   # cross into obs rep
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    """Per-agent state/obs tokens -> interaction-aware representation + per-agent value."""

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state=False):
        super().__init__()
        self.encode_state = encode_state
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        # value head: forcing the rep to predict the return is what makes it expressive
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                  nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        x = self.state_encoder(state) if self.encode_state else self.obs_encoder(obs)
        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)                            # per-agent V(o^{i_m}): (B, n_agent, 1)
        return v_loc, rep


class Decoder(nn.Module):
    """Shifted joint actions + encoder rep -> per-agent policy logits, masked = predecessor-conditioned."""

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent):
        super().__init__()
        self.action_dim = action_dim
        # action_dim + 1: the extra slot is the start-of-sequence token a^{i_0} for agent i_1
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                  nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, action_dim)))

    def forward(self, action, obs_rep, obs):
        x = self.ln(self.action_encoder(action))
        for block in self.blocks:
            x = block(x, obs_rep)
        return self.head(x)                               # (B, n_agent, action_dim) logits


class MultiAgentTransformer(nn.Module):
    """Encoder-decoder over the agent axis: maps the team's observation sequence to its
    action sequence (auto-regressive at inference, parallel at training) and per-agent values."""

    def __init__(self, state_dim, obs_dim, action_dim, n_agent, n_block, n_embd, n_head,
                 encode_state=False, device=torch.device("cpu")):
        super().__init__()
        self.n_agent = n_agent
        self.action_dim = action_dim
        self.device = device
        state_dim = 37                                      # fixed dummy state width in the default path
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent)
        self.to(device)

    def _zero_state_like_obs(self, obs):
        # The default path ignores the supplied centralized state and encodes observations.
        return torch.zeros((*obs.shape[:-1], 37), dtype=torch.float32, device=self.device)

    def get_values(self, state, obs):
        state = self._zero_state_like_obs(obs)
        v_loc, _ = self.encoder(state, obs)               # centralized critic call
        return v_loc

    @torch.no_grad()
    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # INFERENCE: auto-regressive. a^{i_m} is sampled then fed back to produce a^{i_{m+1}}.
        B = obs.shape[0]
        state = self._zero_state_like_obs(obs)
        _, obs_rep = self.encoder(state, obs)
        shifted_action = torch.zeros((B, self.n_agent, self.action_dim + 1), device=self.device)
        shifted_action[:, 0, 0] = 1                       # start token in the +1 slot for agent i_1
        output_action = torch.zeros((B, self.n_agent, 1), dtype=torch.long, device=self.device)
        output_action_log = torch.zeros((B, self.n_agent, 1), device=self.device)
        for i in range(self.n_agent):
            logit = self.decoder(shifted_action, obs_rep, obs)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10
            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            output_action[:, i, 0] = action
            output_action_log[:, i, 0] = distri.log_prob(action)
            if i + 1 < self.n_agent:                      # feed this action in to condition the next
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=self.action_dim)
        return output_action, output_action_log

    def evaluate(self, state, obs, action, available_actions=None):
        # TRAINING: parallel. Ground-truth predecessor actions are in the buffer, so build the whole
        # shifted-action tensor at once; the causal mask still enforces predecessor-only conditioning.
        B = obs.shape[0]
        state = self._zero_state_like_obs(obs)
        v_loc, obs_rep = self.encoder(state, obs)
        one_hot = F.one_hot(action.squeeze(-1), num_classes=self.action_dim).float()
        shifted_action = torch.zeros((B, self.n_agent, self.action_dim + 1), device=self.device)
        shifted_action[:, 0, 0] = 1                       # start token
        shifted_action[:, 1:, 1:] = one_hot[:, :-1, :]    # position m carries agent i_{m-1}'s action
        logit = self.decoder(shifted_action, obs_rep, obs)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10
        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)
        return action_log, v_loc, entropy
```

And the training step that drives it — joint PPO clip on the decoder, PPO value regression on the encoder value, one optimizer, one backward:

```python
def mat_update(model, optimizer, sample, clip, value_coef, entropy_coef, max_grad_norm):
    state, obs, actions, avail, old_log_probs, returns, value_preds, adv, active_masks = sample
    adv = (adv - adv.mean()) / (adv.std() + 1e-5)         # advantage normalization

    action_log, values, entropy = model.evaluate(state, obs, actions, avail)
    ratio = torch.exp(action_log - old_log_probs)         # r^{i_m}: conditioned ratio via the mask
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    policy_loss = (-(torch.min(surr1, surr2).sum(dim=-1, keepdim=True)) * active_masks).sum()
    policy_loss = policy_loss / active_masks.sum()

    value_clipped = value_preds + (values - value_preds).clamp(-clip, clip)
    value_loss = torch.max((returns - values) ** 2, (returns - value_clipped) ** 2)
    value_loss = (value_loss * active_masks).sum() / active_masks.sum()

    dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
    loss = policy_loss + value_coef * value_loss - entropy_coef * dist_entropy
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return policy_loss, value_loss
```
