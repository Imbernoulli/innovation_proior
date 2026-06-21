We have $n$ agents acting in a shared, partially-observed cooperative Markov game, each seeing a local observation $o^i$, each picking an action $a^i$, and the whole team collecting one shared scalar reward $R(o,a)$; the goal is to maximize the joint return $J(\pi) = \mathbb{E}[\sum_t \gamma^t R(o_t, a_t)]$. The single-agent toolbox is dependable here: TRPO gives a monotonic-improvement bound, $J(\bar\pi) \ge L_\pi(\bar\pi) - C\, D_{\mathrm{KL}}^{\max}(\pi,\bar\pi)$ with surrogate $L_\pi(\bar\pi) = J(\pi) + \mathbb{E}_{s\sim\rho_\pi, a\sim\bar\pi}[A_\pi(s,a)]$, so maximizing the surrogate inside a small KL ball cannot decrease $J$, and PPO turns that into a cheap first-order clip $\mathbb{E}[\min(rA, \mathrm{clip}(r,1\pm\varepsilon)A)]$. The trouble is that a team is not one policy but $n$ of them, and there are two intertwined jobs that come apart: finding an improvement direction for each agent, and combining those directions so the joint return actually rises. It is entirely possible for every agent to move in a locally improving direction and for the joint return to drop.

The existing options each fail at exactly one of the pieces we need. Giving all agents the same shared network is sample-efficient when the agents are interchangeable, but it imposes $\theta^i = \theta^j$ as a hard constraint, and on genuinely heterogeneous tasks this caps the achievable return, sometimes catastrophically: in the one-state game on $\{0,1\}^n$ where reward is $1$ only on $(0^{n/2}, 1^{n/2})$ and $(1^{n/2}, 0^{n/2})$, a shared $\mathrm{Bernoulli}(p)$ policy lands on a good pattern with probability $2[p(1-p)]^{n/2}$, maximized at $p=1/2$ to give $2(1/4)^{n/2} = 2/2^n$ against the optimum's $1$ — an exponentially decaying ratio. Dropping the sharing and giving each agent its own parameters with a centralized critic (the COMA/MADDPG style) restores expressiveness but loses the guarantee, and not as a technicality: with two unit-variance Gaussian policies and reward $r(a_1,a_2)=a_1 a_2$, starting at $(\mu_1,\mu_2)=(-0.25,0.25)$ with return $-0.0625$, agent 1's unilateral move to $\mu_1=0.75$ and agent 2's unilateral move to $\mu_2=-0.75$ each improve against the old partner (both inside a KL ball of radius $0.5$, since $D_{\mathrm{KL}}(\mathcal N(\mu,1),\mathcal N(\mu+1,1))=1/2$), yet the simultaneous move gives $0.75\cdot(-0.75)=-0.5625$, worse than the start. Simultaneous independent ascent has no claim on joint improvement. Underneath all of this sits the credit-assignment ache: one shared reward over $n$ agents, so feeding the joint value into a multi-agent policy gradient makes each agent's estimate variance grow with $n$, and the usual patches — local value functions, counterfactual baselines — sit on a monolithic critic that never represents how the agents relate. We want a single paradigm that delivers joint (not just individual) improvement, additive $\sum_i |A^i|$ rather than multiplicative $\prod_i |A^i|$ search cost, a monotonic guarantee, support for heterogeneous and variable agent counts, and parallel updates.

I propose the Multi-Agent Transformer (MAT), which casts cooperative MARL as a sequence-modeling problem and builds the cooperation into the model itself. The load-bearing idea is the multi-agent advantage decomposition. Define the multi-agent observation-value function for an ordered subset of agents, $Q_\pi(o, a^{i_{1:m}}) = \mathbb{E}[R^\gamma \mid o_0=o, a^{i_{1:m}}_0 = a^{i_{1:m}}]$ — the expected return when agents $i_1,\dots,i_m$ are pinned to those actions and everyone else acts under $\pi$ — which recovers the ordinary $Q$ at $m=n$ and the ordinary $V$ at $m=0$, and the conditioned multi-agent advantage $A^{i_m}_\pi(o, a^{i_{1:m-1}}, a^{i_m})$, how much better than average it is for agent $i_m$ to take $a^{i_m}$ given that its predecessors $i_{1:m-1}$ have already committed. Now telescope the joint advantage. For any permutation, any state, any joint action, insert intermediate $Q$s that grow the pinned set one agent at a time,
$$A^{i_{1:m}}_\pi(o, a^{i_{1:m}}) = Q^{i_{1:m}}_\pi(o, a^{i_{1:m}}) - V_\pi(o) = \sum_{k=1}^{m}\big[\,Q^{i_{1:k}}_\pi(o, a^{i_{1:k}}) - Q^{i_{1:k-1}}_\pi(o, a^{i_{1:k-1}})\,\big],$$
where every interior term cancels, leaving $Q^{i_{1:m}}$ at the top and $Q^{i_{1:0}}=V$ at the bottom, and each bracket is by definition agent $i_k$'s predecessor-conditioned advantage, giving the identity
$$A^{i_{1:n}}_\pi(o, a^{i_{1:n}}) = \sum_{m=1}^{n} A^{i_m}_\pi(o, a^{i_{1:m-1}}, a^{i_m}).$$
This holds for any permutation, any state, any joint action, with no value-decomposability assumption — it is pure telescoping — and it reframes everything. If each agent in order picks an action with positive conditioned advantage, the sum of positive terms is positive, so the joint advantage is positive: joint improvement, guaranteed term by term. It also collapses the search from multiplicative to additive — agent $i_1$ searches its $|A^{i_1}|$ actions, then $i_2$ searches its $|A^{i_2}|$ conditioned on $i_1$'s choice, and so on, for a total of $\sum_i |A^i|$. The earlier sequential scheme (draw a random permutation, update agents strictly one at a time, each using its predecessors' newly updated policies for the importance-sampling term) realizes this with a monotonic guarantee and Nash limit points, but it is strictly sequential: agent $i_m$ cannot be optimized until $i_{1:m-1}$ finish, so the $n$ stages cannot run in parallel and become a wall-clock wall as $n$ grows, and each agent still optimizes a separately handcrafted objective rather than being one cooperating whole.

What makes MAT work is the observation that the decomposition's dependency — output element $m$ may depend on the whole input and on output elements $1..m-1$ — is exactly the dependency structure of sequence-to-sequence generation, and the masked-decoder Transformer is the literal implementation of that conditioning. So MAT maps the team's observation sequence $(o^{i_1},\dots,o^{i_n})$ to its action sequence $(a^{i_1},\dots,a^{i_n})$ over the agent axis with an encoder-decoder. The encoder, parameters $\phi$, projects each agent's observation to a model dimension and pushes the $n$ tokens through blocks of *unmasked* self-attention plus a position-wise MLP, each wrapped as $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. The attention must be unmasked here because building an interaction-aware representation of an agent should look at *every* other agent — there is no causal order on reading observations, only on emitting actions — and the residuals and LayerNorm are what let the attention stack train at depth. The attention is scaled dot-product, $w(q,k) = \langle q,k\rangle/\sqrt{d_k}$ softmaxed onto the values; the $1/\sqrt{d_k}$ is essential because if query and key coordinates are roughly mean-0 variance-1 their dot product has $\mathrm{var}(q\cdot k)=d_k$, so without rescaling the logits grow with width and push softmax into a near-one-hot, vanishing-gradient regime. On top of the encoder's representation hangs a per-agent value head ($\mathrm{Linear}\to\mathrm{GELU}\to\mathrm{LayerNorm}\to\mathrm{Linear}(\cdot,1)$), so the encoder *is* the centralized critic — forcing the shared representation to predict returns is exactly the pressure that makes it encode the interactions, and the joint value for advantage estimation is the clean average $\hat V_t = \tfrac1n\sum_m V_\phi(\hat o^{i_m}_t)$, fed through GAE to produce the joint advantage $\hat A_t$. The clean encoder objective is the Bellman error
$$L_{\mathrm{Encoder}}(\phi) = \frac{1}{Tn}\sum_{m=1}^{n}\sum_{t=0}^{T-1}\big[\,R(o_t,a_t) + \gamma\, V_{\bar\phi}(\hat o^{i_m}_{t+1}) - V_\phi(\hat o^{i_m}_t)\,\big]^2,$$
with a periodically frozen target network $\bar\phi$ so the regression target does not chase the parameters; the shipped trainer implements this as PPO value regression to buffered returns with optional value clipping, normalization, Huber loss, and active-agent masks.

The decoder, parameters $\theta$, consumes the encoder representation and the *shifted* joint actions and emits each agent's policy $\pi^{i_m}_\theta(a^{i_m}\mid \hat o^{i_{1:n}}, a^{i_{1:m-1}})$. "Shifted" because output $m$ must condition on actions $1..m-1$, not its own, so the previous agents' actions are offset by one and the very first agent $i_1$, which has no predecessor, receives a start-of-sequence symbol — implemented by widening the action one-hot to $\text{action\_dim}+1$ and putting the start token $a^{i_0}=[1,0,\dots,0]$ in the extra slot. Each decode block does *masked* self-attention over the action stream, then a second masked attention whose query is the encoder representation and whose keys/values are the masked action stream, then an MLP, all with residuals and LayerNorm; the lower-triangular masks set $w(q^{i_r},k^{i_j})=0$ for $r<j$, the literal "may not see the future" that makes agent $i_m$ depend only on $a^{i_{1:m-1}}$ — the same constraint the decomposition theorem demands. The decoder trains on the PPO clip of the *joint* advantage,
$$L_{\mathrm{Decoder}}(\theta) = -\frac{1}{Tn}\sum_{m=1}^{n}\sum_{t=0}^{T-1}\min\!\big(r^{i_m}_t(\theta)\,\hat A_t,\; \mathrm{clip}(r^{i_m}_t(\theta), 1\pm\varepsilon)\,\hat A_t\big),\quad r^{i_m}_t(\theta) = \frac{\pi^{i_m}_\theta(a^{i_m}_t\mid \hat o^{i_{1:n}}_t, a^{i_{1:m-1}}_t)}{\pi^{i_m}_{\theta_{\mathrm{old}}}(a^{i_m}_t\mid \hat o^{i_{1:n}}_t, a^{i_{1:m-1}}_t)}.$$
The reason the single scalar $\hat A_t$ can multiply every agent's ratio, rather than estimating a separate table of conditioned advantages, is that the mask already makes each ratio $r^{i_m}$ predecessor-conditioned: the surrogate has the conditioning structure the decomposition requires, which is precisely where the TRPO-style monotonic argument attaches, while GAE on the joint value keeps the advantage estimate robust without rebuilding the combinatorial object I am trying to avoid. The decisive payoff is in the train/inference split. At inference I must generate auto-regressively, sampling $a^{i_m}$ and feeding it back to condition $a^{i_{m+1}}$, because the predecessors' actions genuinely are not known yet. But at training the predecessors' ground-truth actions already sit in the buffer, so I feed the whole shifted-action tensor in at once and let the causal mask enforce the conditioning, computing every agent's clipped ratio in a *single parallel forward pass* — the same predecessor-conditioned trust-region structure as the sequential scheme, but without its $n$-stage parameter-update bottleneck.

Two further choices fall out of the guarantee and the regime. I permute the agent order randomly every iteration: the monotonic bound holds for any fixed order, but only randomization (giving every permutation positive probability of leading) makes the limit a Nash equilibrium, since at convergence no agent is incentivized to deviate because it had a chance to be the one that moved. And I drop positional encoding entirely in favor of an agent-id one-hot in the observation features — because the random permutation makes a token's position index meaningless across iterations (the same physical agent appears at different positions), so encoding position would inject noise the model tries to read structure off, whereas the stable identity belongs with the agent and, once present, leaves the position index no semantic job. The cooperative tasks have modest agent counts, so the default is a single block, a single head, and $n_{\mathrm{embd}}=64$ with a compact $\mathrm{Linear}(d,d)\to\mathrm{GELU}\to\mathrm{Linear}(d,d)$ feed-forward, and the whole thing trains under one Adam optimizer on the combined loss $\text{policy\_loss} - \text{entropy\_coef}\cdot\text{entropy} + \text{value\_coef}\cdot\text{value\_loss}$ with a single backward, gradient-norm clipping, and GAE advantages normalized $(\hat A - \mathrm{mean})/(\mathrm{std}+10^{-5})$ over active agents.

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
