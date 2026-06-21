The thing that bothers me about adapting a large pre-trained generation model — GPT-2 for table-to-text, BART for summarization — is the per-task cost of full fine-tuning. To specialize the model I copy it and update every one of its several-hundred-million parameters, ending up with a complete checkpoint for each task, with no sharing across tasks. When the goal is to serve many tasks, or many users each wanting their own behavior, that storage and the impossibility of mixing tasks in one served backbone is the whole bottleneck. What I actually want is one frozen backbone shared across everything, plus a tiny per-task object — so the real question is: what is the smallest thing I can learn that re-purposes a frozen LM for a task while keeping accuracy near full fine-tuning? The existing options each fall short. Fine-tuning only the top $k$ layers shrinks the footprint but trades away accuracy as $k$ drops. Adapters insert small bottleneck modules between the frozen layers and tune only those; that reaches parameter efficiency, but it adds depth into the backbone and changes the internal compute path rather than steering the model through its own context. And discrete prompting — even gradient-guided trigger search like AutoPrompt — is combinatorially hard to optimize and, more fundamentally, forces every prompt slot to be the embedding of an actual vocabulary word, capping how expressive the steering context can be; hand-written natural-language instructions mostly fail to steer GPT-2 and BART at all, since that capability only emerges in the very largest models.

The clue I keep returning to is that a frozen LM can be steered by its context with no weight changes: prepend "Barack" and the next-token probability of "Obama" jumps. So for a whole task there may exist some context that steers the LM to solve it — and if so I pay almost nothing per task, just storing that context. The reason a discrete prompt is hard to optimize is precisely that I am insisting each position be a real token; but the model does not consume tokens, it consumes their continuous embeddings. So I drop the constraint: let the prompt positions be free continuous vectors optimized directly by gradient descent against the task loss. This is smooth rather than combinatorial, and strictly more expressive, since a discrete prompt is just the special case where each vector happens to coincide with a real word's embedding. The trouble is that input-layer embeddings alone are too weak: they sit at the very bottom of the network, and their only handle on what happens at layer 12 is whatever the frozen layers 1 through 11 decide to do with them — a long, indirect dependency path riding on very few, very distant parameters. The expressiveness order is clear before any benchmark: discrete prompt $<$ embedding-only $<$ something that intervenes deeper. I need to intervene deeper.

I propose Prefix-Tuning. The construction is to prepend $|P|$ "virtual token" positions and make their per-layer key/value tensors trainable, while freezing the entire LM. The reasoning for "deeper" is concrete: at each layer a position $i$ computes its activation by attending to the keys and values of the previous positions in that same layer. So rather than give the virtual positions only bottom embeddings that the frozen stack transforms however it likes, I give them their own trainable key and value vectors at every layer — exactly the attention objects the later real tokens will read. For an autoregressive LM the sequence becomes $z = [P; x; y]$ (and $[P; x; P'; y]$, a prefix on each side, for an encoder-decoder). The activation recurrence is unchanged for real tokens, while prefix positions simply read their stored activations,
$$h_i = P_\theta[i,:] \ \text{ if } i \in \text{prefix}, \qquad h_i = \mathrm{LM}_\varphi(z_i, h_{<i}) \ \text{ otherwise},$$
where $P_\theta$ is the flattened per-layer key/value prefix, equivalently tensors $\{(K_l^P, V_l^P)\}_{l=1}^{n}$ each shaped $[\text{batch}, n_\text{head}, |P|, \text{head\_dim}]$. Flattened, one prefix position carries one key and one value at every layer, so its width is $2 \cdot n_\text{layers} \cdot d_\text{model}$. The backbone parameters $\varphi$ stay frozen; the only trainable parameters are the prefix-side $\theta$, and the objective is identical to fine-tuning, $\max \sum_{i \in y} \log p_\varphi(z_i \mid h_{<i})$. Every real activation still depends on $P_\theta$, because the prefix keys and values sit in the left context of every real token at every layer — so through ordinary attention the prefix steers both the encoding of the input $x$ and the generation of the output $y$, with no weight touched and a far shorter dependency path than the embedding-only scheme.

Several design choices make it work. Placement is prefixing, not infixing: at the front, the prefix is in the left context of both $x$ and $y$, so it can shape how $x$ is encoded and how $y$ is generated; placed between them as $[x; P; y]$, $x$ is already encoded before the trainable positions appear, so they could only affect $y$. Prefixing strictly dominates in reach, so the front it is. Implementation rides on the cached-attention mechanism that already exists: "supply the prefix's per-layer key/value activations as extra left context" is exactly what a Hugging Face-style `past_key_values` cache does, so the served object is just a tuple of one $(K, V)$ pair per layer — a learned cache prepended at every layer, not a new hidden layer inside the LM. Optimizing $P_\theta$ directly turns out to be the bottleneck, because that raw cache spans layers, heads, keys and values, all living in the LM's activation space with its own scales and correlations, which makes the optimization learning-rate and initialization sensitive. So I reparameterize during training: optimize a smaller matrix $P'_\theta \in \mathbb{R}^{|P| \times k}$ — same number of rows, much smaller column dimension $k$ ($k=512$ for table-to-text, $k=800$ for summarization) — and expand each row through a shared MLP,
$$P_\theta[i,:] = \mathrm{MLP}_\theta(P'_\theta[i,:]),$$
which maps the low-dimensional rows into the cache-shaped space and shares parameters across prefix positions; that shared smooth map is the stability reason. The MLP and $P'_\theta$ are needed only during training: once trained I evaluate $P_\theta = \mathrm{MLP}_\theta(P'_\theta)$ once, store the resulting per-layer key/value tensors, and throw the reparameterization machinery away, so the stored per-task object is exactly the prefix cache and nothing else. Prefix length is a per-task hyperparameter trading expressiveness against overfitting — short for table-to-text, longer for summarization where the input is longer and the operation less templatic — and a longer prefix barely costs anything at inference because attention over the whole prefix is parallelized on the GPU rather than added as sequential depth. When data are scarce, I initialize the prefix from the per-layer key/value activations of real words run through the frozen LM itself, which seeds the cache in a region of activation space the model already "speaks," concordant with disturbing the pre-trained model as little as possible. A bonus falls out for free: because the task lives in the context rather than the weights, one batch can carry different prefixes against the same frozen backbone, enabling per-task and per-user batching that a per-task fine-tuned checkpoint cannot do. Training uses AdamW with a linear LR schedule, by default 10 epochs, batch size 5, lr 5e-5.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixTuning(nn.Module):
    """Trainable per-layer key/value prefix steering a frozen LM, MLP-reparametrized."""
    def __init__(self, lm_config, prefix_len=10, reparam_dim=512, mlp_hidden_dim=512):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layer = lm_config.n_layer
        self.n_head = lm_config.n_head
        self.d_model = lm_config.n_embd
        self.head_dim = self.d_model // self.n_head
        self.register_buffer("prefix_positions", torch.arange(prefix_len), persistent=False)
        self.prefix_basis = nn.Embedding(prefix_len, reparam_dim)  # P'_theta, width k
        self.mlp = nn.Sequential(                                   # expand k -> flattened K/V cache
            nn.Linear(reparam_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, self.n_layer * 2 * self.d_model),  # 2 = key + value
        )

    def materialize(self, batch_size, device):
        idx = self.prefix_positions.to(device)
        h = self.mlp(self.prefix_basis(idx))                       # [prefix_len, n_layer*2*d_model]
        h = h.unsqueeze(0).expand(batch_size, -1, -1)
        h = h.reshape(batch_size, self.prefix_len, self.n_layer * 2, self.n_head, self.head_dim)
        h = h.permute(2, 0, 3, 1, 4)                               # [n_layer*2, B, n_head, len, hd]
        return tuple((h[2 * l], h[2 * l + 1]) for l in range(self.n_layer))   # (key, value) per layer


def loss_fn(frozen_lm, prefix, x_ids, y_ids):
    z = torch.cat([x_ids, y_ids], dim=1)
    past = prefix.materialize(z.size(0), z.device)
    out = frozen_lm(input_ids=z, past_key_values=past)
    logits = out.logits[:, x_ids.size(1) - 1:-1, :]                # score the y positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))


def train(frozen_lm, prefix, loader, opt):
    for p in frozen_lm.parameters():
        p.requires_grad = False
    optim = torch.optim.AdamW(prefix.parameters(), lr=opt.lr)      # only prefix-side params train
    for batch in loader:
        loss = loss_fn(frozen_lm, prefix, batch["x_ids"], batch["y_ids"])
        loss.backward(); optim.step(); optim.zero_grad()

# After training: cache the materialized per-layer key/value tensors and discard the MLP/P'_theta.
```
