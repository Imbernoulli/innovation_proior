The substrate I am handed fixes everything except one thing: the MAPPO RNN actor, the `ppo_learner`, GAE, the `q_nstep=5` target, the optimizer — all frozen — leaving only the centralized critic as the free variable. The scaffold's default fill is the plainest centralized critic there is, a shared MLP over the global state plus an agent one-hot, $V_\phi([s, e_a])$. Precisely because it is plain it leaves the obvious lever untouched: it never lets the agents' representations *interact*. Cooperation is hard exactly because the value of the team situation depends on how the agents relate to each other, and a critic that flattens $[s, e_a]$ through an MLP has to learn those relations implicitly inside its weights, with no architectural place to put them. So the most ambitious critic I can drop in first is one whose *architecture* is built to model agent-to-agent interaction — if structure helps, I want to see it help from the start.

I propose a **MAT-style attention critic** — the critic-only, encoder-only form of the multi-agent Transformer (Wen et al. 2022). The idea is to turn each agent into a token and let self-attention compute a data-dependent mixing across the agent axis. A fixed MLP cannot express "agent $i$'s value should weight agent $j$'s features by how relevant $j$ is to $i$," because that weighting changes with the configuration — in a SMAC fight a marine's value depends on the allies and enemies near *it* and almost nothing on a unit across the map. What computes a content-addressed, data-dependent mixing over a set of tokens is exactly self-attention: for each agent (query) a softmax weighting over all agents (keys) pulls in their values,
$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$
The $1/\sqrt{d_k}$ is load-bearing, not decorative: if the query and key components are roughly unit-variance, their dot product over $d_k$ dimensions has variance $d_k$, so without the scaling the raw scores are large, the softmax saturates toward one-hot, and its gradient nearly vanishes; dividing by $\sqrt{d_k}$ keeps the scores order-1 and the softmax in a regime that actually learns.

I have to be honest about what this task exposes versus what the full method is. The full sequence-modeling method is an encoder–decoder over the agent axis: an *unmasked* encoder builds interaction-aware representations and serves as the centralized value, and a *masked* auto-regressive decoder generates each agent's action conditioned only on its predecessors. The decoder is the engine — it is the literal implementation of the identity that the joint advantage equals the sum of per-agent advantages each conditioned on the agents before it, which is what gives the method its monotonic-improvement story. But here the **actor is fixed**; the editable region is `CustomCritic` only. So I cannot build the auto-regressive decoder, cannot feed predecessor actions into action generation, and cannot realize the advantage-decomposition guarantee. What survives is the encoder half used as a critic — the unmasked self-attention over agent tokens with a per-agent value head — stripped of the decoder that is the method's actual source of power. That is the right first probe regardless: it isolates "does an attention critic beat a flat MLP critic," holding the MAPPO actor and learner fixed.

Each agent's token carries both its local detail and the global picture, so I form the per-agent input as $[o_i \oplus s]$ — the agent's observation concatenated with the broadcast global state — and project it to a model width $d_{\text{model}} = 128 = \texttt{hidden\_dim}$ with a linear map, keeping the critic comparable in capacity to the MLP baselines. I include $s$ in *every* token rather than $o_i$ alone because the value of the situation genuinely depends on global structure an agent may not see locally, and since attention mixes across agents anyway, handing each token the global state means the mixing operates on representations that already know the global context. Then a single `TransformerEncoderLayer` over the $n$ agent tokens: multi-head self-attention with 4 heads — enough to let different heads specialize on different relational patterns (allies vs. enemies) without fragmenting a width-128 model into tiny per-head subspaces — feed-forward width $4 \cdot d_{\text{model}}$ in the usual transformer proportion, GELU, and deliberately **no dropout**, because this is a value regression where I want a deterministic, low-variance estimate and dropout would inject noise into the very baseline whose job is to *reduce* variance. The attention is *unmasked*: building an interaction-aware representation of agent $i$ should look at every other agent, since there is no causal ordering on *reading* observations, only (in the full method) on *emitting* actions, and there is no emission stage. After the encoder layer, a per-token linear head maps each agent's mixed representation to a scalar value. One encoder layer, not a deep stack: SMAC teams are small (8–10 units), the relational structure is shallow, and a deeper stack on a value regression with only ~5M steps is more likely to overfit and destabilize than to help.

The shapes must line up with how the learner calls this. The batch gives `state` of shape $(B, T, \texttt{state\_dim})$ and `obs` of shape $(B, T, n, \texttt{obs\_dim})$, and the result is later `.squeeze(3)`-ed, so I must return $(B, T, n, 1)$. Inside `forward` I broadcast the state across the agent axis, concatenate with `obs`, and project to $(B, T, n, d_{\text{model}})$. The transformer encoder expects $(\text{batch}, \text{seq\_len}, d_{\text{model}})$ and *my* sequence is the agent axis, so I flatten $(B, T)$ into one batch dimension, giving $(B\!\cdot\!T, n, d_{\text{model}})$, run the encoder so attention mixes across the $n$ agents independently for each $(B, T)$ position, then reshape back and apply the value head. That flatten-then-restore is the detail that makes the self-attention operate over *agents* and not accidentally over time.

I am putting the most complex critic in first deliberately, and the value-learning conditions are not gentle. The encoder critic has many more parameters than a two-layer MLP, and it is trained as a *bootstrapped* value regression against a target copy of itself, masked, with returns *not* standardized (`standardise_returns=False`). A high-capacity critic fit to a moving bootstrapped target with unnormalized returns is exactly the setup where value learning can lock into a bad basin: the attention weights can collapse onto an unhelpful pattern the masked MSE tolerates but that produces biased per-agent advantages, or the per-agent values can flatten toward a team mean — an over-smoothing failure specific to attention critics. And a cold-started transformer under the fixed `lr=3e-4` with no warmup is seed-sensitive. So the falsifiable expectation is two-signatured: if interaction modeling pays, it should help most where coordination is hardest (MMM's 10 mixed units, 3s5z's team of 8) and at least match on 2s3z; if value-learning fragility dominates, I expect the reverse — high seed variance and *collapse* on the hard maps, possibly some seeds at exactly zero win rate while one looks fine, with 2s3z surviving as the easiest value regression. I am putting this first so the measurement tells me whether complexity or robustness is the binding constraint.

```python
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom centralized critic for MAPPO
# ======================================================================
class CustomCritic(nn.Module):
    """MAT-style attention critic — self-attention over per-agent tokens.

    Adapted from Wen et al. 2022 MAT (arXiv 2205.14953), critic-only form.
    Each agent's token encodes its local observation together with the
    global state; a single TransformerEncoder layer mixes information
    across agents via self-attention, then a per-token linear head
    produces the scalar value.
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        obs_dim = int(scheme["obs"]["vshape"])
        state_dim = int(scheme["state"]["vshape"])
        self.d_model = args.hidden_dim

        # Per-agent token projection: [obs_i ⊕ state] → d_model
        self.token_proj = nn.Linear(obs_dim + state_dim, self.d_model)

        # Single transformer encoder layer with self-attention across agents
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=4 * self.d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Per-agent value head
        self.v_head = nn.Linear(self.d_model, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
        state = batch["state"][:, ts]                                    # (B, T, state_dim)
        state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
        tokens = th.cat([obs, state], dim=-1)                            # (B, T, n, obs+state)
        tokens = self.token_proj(tokens)                                 # (B, T, n, d_model)

        # Flatten (B, T) into a single batch dim for the transformer,
        # then restore: TransformerEncoder expects (bs*, seq_len, d_model).
        b, tt, n, d = tokens.shape
        tokens = tokens.reshape(b * tt, n, d)
        attn_out = self.encoder(tokens)                                  # (B*T, n, d_model)
        attn_out = attn_out.reshape(b, tt, n, d)

        q = self.v_head(attn_out)                                       # (B, T, n, 1)
        return q
```
