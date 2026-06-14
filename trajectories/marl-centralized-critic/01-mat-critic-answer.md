**Problem.** Cooperative MARL under partial observability with a shared reward, trained CTDE. The actor
(MAPPO RNN policy), the learner, GAE/`n`-step, and the optimizer are fixed; the only editable thing is
the centralized critic. The plain default critic — a flat MLP over `[s, e_a]` — has no architectural
place to model agent-to-agent interaction, which is exactly what cooperative value depends on.

**Key idea (critic-only attention over agents).** Turn each agent into a token `[o_i ⊕ s]`, project to
`d_model`, and run a single unmasked `TransformerEncoder` layer so each agent's representation is mixed
with every other agent's via data-dependent self-attention `softmax(QKᵀ/√d_k)V`; a per-agent linear head
reads the value off the mixed representation. This is the *encoder-only, critic-only* form of the
multi-agent sequence-modeling method (Wen et al. 2022, arXiv:2205.14953): the full method also replaces
the actor with a masked auto-regressive decoder that conditions each agent's action on its predecessors
(the source of its advantage-decomposition guarantee), but the harness fixes the actor, so only the
interaction-modeling encoder survives. The probe isolates "does an attention critic beat a flat MLP?"

**Why it might work / might not.** Attention can express "weight agent `j`'s features by relevance to
`i`," which a fixed MLP cannot. But the critic is high-capacity, fit to a *bootstrapped* target via a
target copy with **unnormalized** returns (`standardise_returns=False`) and masked MSE — fragile value
learning where the attention can collapse to an over-smoothed team mean or a degenerate basin; and a
cold-started transformer under the fixed `lr=3e-4` with no warmup is seed-sensitive.

**Hyperparameters.** `d_model = hidden_dim` (128); 1 encoder layer; 4 heads; feed-forward `4·d_model`;
GELU; **no dropout** (a value baseline wants low variance); attention unmasked (full read across
agents); per-agent linear value head. Flatten `(B,T)` into the batch dim so attention runs over the
agent axis, restore, return `(B, T, n, 1)`.

**What to watch.** If interaction modeling pays, it should help most on the hard, heterogeneous maps
(MMM, 3s5z) and at least match on 2s3z. If value-learning fragility dominates, expect the reverse:
high seed variance and collapse on the hard maps (bimodal per-seed outcomes, some seeds near zero win
rate), with 2s3z surviving as the easiest value regression.

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
