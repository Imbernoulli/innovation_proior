# Multi-Agent Transformer (MAT)

MAT casts cooperative multi-agent reinforcement learning as a **sequence modeling** problem: map the
team's per-agent observation sequence `(o^{i_1}, …, o^{i_n})` to its action sequence
`(a^{i_1}, …, a^{i_n})` with an encoder-decoder Transformer over the **agent axis**. The encoder reads
all observations with full self-attention and doubles as the centralized critic (a per-agent value
head); the decoder generates each agent's action auto-regressively with **masked** attention, so agent
`i_m` is conditioned only on its predecessors' actions `a^{i_{1:m-1}}`. That masking is the literal
implementation of the multi-agent advantage decomposition, which is what gives MAT its monotonic
joint-improvement argument while making all agents' clipped ratios computable in one parallel
training pass and updating the shared encoder-decoder with one optimizer step.

## Problem it solves

Cooperative Markov game, `n` agents, partial observability, shared reward; maximize the joint return
`J(π) = E[Σ_t γ^t R(o_t, a_t)]`. Independent per-agent ascent does not guarantee joint improvement:
with unit-variance Gaussian policies and `r(a_1,a_2)=a_1 a_2`, starting from
`(μ_1, μ_2)=(-0.25,0.25)`, agent 1's unilateral move to `0.75` and agent 2's unilateral move to
`-0.75` each improves against the old partner, but the simultaneous move changes the expected return
from `-0.0625` to `-0.5625`. Parameter sharing
guarantees nothing on heterogeneous tasks and is exponentially suboptimal there. The credit-assignment
variance of multi-agent policy gradients grows with `n`. MAT wants joint (not just individual)
improvement, additive (`Σ_i|A^i|`) rather than multiplicative (`∏_i|A^i|`) search, a monotonic
guarantee, support for heterogeneous and variable agent counts, and parallel training-time ratio
computation.

## Key idea

**Multi-agent advantage decomposition (telescoping identity).** For any permutation `i_{1:n}`, any
state, any joint action:

```
A^{i_{1:n}}_π(o, a^{i_{1:n}}) = Σ_{m=1}^{n} A^{i_m}_π(o, a^{i_{1:m-1}}, a^{i_m}),
```

proved by telescoping `Q^{i_{1:m}} − V = Σ_k [Q^{i_{1:k}} − Q^{i_{1:k-1}}]` and identifying each
bracket as agent `i_k`'s advantage conditioned on its predecessors. No value-decomposability
assumption is needed. Consequence: if each agent in order picks a positive *conditioned* advantage,
the joint advantage is positive — joint improvement, additive search.

**Sequence view.** "Agent `i_m`'s action depends on `o^{i_{1:n}}` and `a^{i_{1:m-1}}`" is exactly the
dependency of sequence-to-sequence generation. The Transformer's masked auto-regressive decoder
enforces the predecessor-only conditioning; the encoder's full self-attention builds interaction-aware
representations and the centralized value. At training time the predecessors' ground-truth actions are
in the buffer, so all agents' objectives compute in one parallel pass; at inference the decoder runs
auto-regressively. This keeps the predecessor-conditioned trust-region structure used by the
monotonic argument without the sequential scheme's `n`-stage parameter-update bottleneck.

## Architecture

- **Encoder** (`φ`): per-agent obs token → `LayerNorm → Linear → GELU` → `n_block` blocks of
  *unmasked* self-attention across agents + position-wise MLP, each `x ← LayerNorm(x + Sublayer(x))`
  (residual). Output: interaction-aware representation `ô^{i_m}` and a per-agent value head
  `Linear → GELU → LayerNorm → Linear(·,1)` giving `V_φ(ô^{i_m})`. The clean objective is the Bellman error
  ```
  L_Encoder(φ) = (1/Tn) Σ_m Σ_t [ R(o_t,a_t) + γ V_φ̄(ô^{i_m}_{t+1}) − V_φ(ô^{i_m}_t) ]²,
  ```
  with a target network `φ̄`. The canonical trainer implements the PPO value update against buffered
  returns, with optional value clipping, value normalization, Huber loss, and active-agent masks. The
  joint value for GAE is `V̂_t = (1/n) Σ_m V(ô^{i_m}_t)`.
- **Decoder** (`θ`): shifted joint actions (with a start token `a^{i_0}`) → action encoder → `n_block`
  decode blocks of *masked* self-attention over actions + a second masked attention whose query is
  the encoder representation and whose keys/values are the masked action stream + MLP, residual +
  LayerNorm → head to per-agent logits/means, i.e. the policy
  `π^{i_m}_θ(a^{i_m} | ô^{i_{1:n}}, a^{i_{1:m-1}})`. Trained with the PPO clip on the **joint**
  advantage `Â_t` (GAE):
  ```
  L_Decoder(θ) = −(1/Tn) Σ_m Σ_t min( r^{i_m}_t Â_t, clip(r^{i_m}_t, 1±ε) Â_t ),
     r^{i_m}_t = π^{i_m}_θ(a^{i_m}_t | ô^{i_{1:n}}_t, a^{i_{1:m-1}}_t)
               / π^{i_m}_{θ_old}(a^{i_m}_t | ô^{i_{1:n}}_t, a^{i_{1:m-1}}_t).
  ```
- **Attention**: `softmax(QKᵀ/sqrt(d_k))V`; `1/sqrt(d_k)` keeps softmax out of the vanishing-gradient
  regime (`var(q·k)=d_k`). Decoder masks are lower-triangular (`w(q^{i_r},k^{i_j})=0` for `r<j`).

## Design choices and why

- **Masked decoder attention** = the conditioning the decomposition requires (agent `i_m` sees only
  `a^{i_{1:m-1}}`). **Unmasked encoder attention** because building an interaction-aware obs
  representation may look at every agent (no causal order on *reading*, only on *acting*).
- **Encoder is the critic** (per-agent value head): forcing the shared representation to predict
  returns makes it expressive, and the mean of per-agent values is the joint value for GAE.
- **Joint advantage in every agent's clip term**: the mask already makes each ratio `r^{i_m}`
  conditioned, so each clipped surrogate has the predecessor-conditioned structure required by the
  decomposition; that is where the TRPO/HAPPO monotonic argument attaches.
- **Agent-id one-hot encoding, not positional encoding**: the intended agent order is randomized, so
  position is meaningless across iterations; stable identity belongs in the observation/state features,
  and position adds no structural signal once identity is already present.
- **Random agent permutation each iteration**: the monotonic bound holds for any order, but
  randomization is what makes the limit a Nash equilibrium.
- **Parallel ratio computation / auto-regressive inference**: ground-truth predecessor actions are
  buffered at train time → one parallel decoder pass and one optimizer step; at inference actions must
  be generated in sequence.
- **Residual + LayerNorm + GELU, `1/sqrt(d_k)` scaling**: standard machinery for training attention
  stacks; the default config uses `n_block = 1`, `n_head = 1`, `n_embd = 64`, and compact
  `Linear(d,d) → GELU → Linear(d,d)` feed-forward blocks.

## Algorithm

```
Initialize encoder φ, decoder θ, buffer B.
for each iteration:
    draw a random agent permutation i_{1:n}
    # rollout (auto-regressive inference)
    for t = 0..T-1:
        encode observations -> representations ô^{i_{1:n}}_t, per-agent values
        for m = 0..n-1: feed a^{i_0..i_m}_t to the decoder, sample a^{i_{m+1}}_t
        execute joint action, store (o_t, a_t, R(o_t,a_t)) in B
    # training (parallel)
    sample minibatch from B
    compute per-agent values V_φ(ô^{i_m}), joint value V̂ = mean_m V_φ, joint advantage Â via GAE
    L_Encoder(φ): Bellman MSE on per-agent values (target net φ̄)
    L_Decoder(θ): PPO clip on Â (decoder run once, masked = predecessor-conditioned)
    minimize policy_loss − entropy_coef·entropy + value_coef·value_loss (one optimizer, one backward)
```

## Working code

Core discrete-action path from the canonical implementation, with auto-regressive inference and parallel
training-time likelihood evaluation:

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.masked, self.n_head = masked, n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

    def forward(self, key, value, query):
        B, L, D = query.size()
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                 nn.GELU(), init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        return self.ln2(x + self.mlp(x))


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(n_embd) for _ in range(3))
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                 nn.GELU(), init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        return self.ln3(x + self.mlp(x))


class Encoder(nn.Module):
    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state=False):
        super().__init__()
        self.encode_state = encode_state
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                  nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        x = self.state_encoder(state) if self.encode_state else self.obs_encoder(obs)
        rep = self.blocks(self.ln(x))
        return self.head(rep), rep                         # (B, n, 1) per-agent value, rep


class Decoder(nn.Module):
    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent):
        super().__init__()
        self.action_dim = action_dim
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
        return self.head(x)


class MultiAgentTransformer(nn.Module):
    def __init__(self, state_dim, obs_dim, action_dim, n_agent, n_block, n_embd, n_head,
                 encode_state=False, device=torch.device("cpu")):
        super().__init__()
        self.n_agent, self.action_dim, self.device = n_agent, action_dim, device
        state_dim = 37                                      # fixed dummy state width in the default path
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent)
        self.to(device)

    def _zero_state_like_obs(self, obs):
        return torch.zeros((*obs.shape[:-1], 37), dtype=torch.float32, device=self.device)

    def get_values(self, state, obs):
        state = self._zero_state_like_obs(obs)
        return self.encoder(state, obs)[0]                  # centralized critic

    @torch.no_grad()
    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        B = obs.shape[0]
        state = self._zero_state_like_obs(obs)
        _, obs_rep = self.encoder(state, obs)
        shifted = torch.zeros((B, self.n_agent, self.action_dim + 1), device=self.device)
        shifted[:, 0, 0] = 1                                # start token a^{i_0}
        out_a = torch.zeros((B, self.n_agent, 1), dtype=torch.long, device=self.device)
        out_lp = torch.zeros((B, self.n_agent, 1), device=self.device)
        for i in range(self.n_agent):                       # auto-regressive
            logit = self.decoder(shifted, obs_rep, obs)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10
            distri = Categorical(logits=logit)
            a = distri.probs.argmax(-1) if deterministic else distri.sample()
            out_a[:, i, 0], out_lp[:, i, 0] = a, distri.log_prob(a)
            if i + 1 < self.n_agent:
                shifted[:, i + 1, 1:] = F.one_hot(a, num_classes=self.action_dim)
        return out_a, out_lp

    def evaluate(self, state, obs, action, available_actions=None):
        B = obs.shape[0]
        state = self._zero_state_like_obs(obs)
        v_loc, obs_rep = self.encoder(state, obs)
        one_hot = F.one_hot(action.squeeze(-1), num_classes=self.action_dim).float()
        shifted = torch.zeros((B, self.n_agent, self.action_dim + 1), device=self.device)
        shifted[:, 0, 0] = 1
        shifted[:, 1:, 1:] = one_hot[:, :-1, :]             # parallel; mask enforces conditioning
        logit = self.decoder(shifted, obs_rep, obs)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10
        distri = Categorical(logits=logit)
        return distri.log_prob(action.squeeze(-1)).unsqueeze(-1), v_loc, distri.entropy().unsqueeze(-1)
```

Training step (joint PPO clip + PPO value loss, one optimizer):

```python
def mat_update(model, optimizer, sample, clip, value_coef, entropy_coef, max_grad_norm):
    state, obs, actions, avail, old_log_probs, returns, value_preds, adv, active_masks = sample
    adv = (adv - adv.mean()) / (adv.std() + 1e-5)
    action_log, values, entropy = model.evaluate(state, obs, actions, avail)
    ratio = torch.exp(action_log - old_log_probs)
    surr1, surr2 = ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv
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

## CTDE critic-only reduction

A centralized-critic-only variant keeps a decentralized per-agent actor and uses the attention
encoder only as the value function: per-agent token = `Linear(o^i ⊕ broadcast(state))`, one
`TransformerEncoder` layer (multi-head self-attention across agents, `dim_ff = 4·d_model`, GELU) →
per-agent value head, trained on the Bellman error with local-advantage policy updates. This isolates
the value of the encoder's attention critic; the full encoder-decoder MAT adds the auto-regressive,
predecessor-conditioned decoder that carries the monotonic guarantee.
