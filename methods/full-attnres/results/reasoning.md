Let me start with the residual stream, because the name hides two different jobs. In a pre-LN stack I usually write a layer as `h_l = h_{l-1} + f_{l-1}(h_{l-1})`, with the normalization folded into `f`. The backward story is the familiar one: if I differentiate from the loss at the top back to an intermediate state, I get `dL/dh_l = (dL/dh_L) * prod_{j=l}^{L-1}(I + df_j/dh_j)`, using row-vector gradients. When I expand that product, there is always a term made only of the identity matrices. That is the clean gradient route that makes depth trainable.

But the forward story is just as important. The same recurrence expands to `h_l = h_1 + sum_{i=1}^{l-1} f_i(h_i)`. So the residual stream is also a depth aggregation rule: it says layer `l` reads the embedding plus every earlier transformation output, each with coefficient exactly one. That is a surprisingly rigid rule. The sequence axis gets self-attention, experts get routers, gates can depend on the token, but the depth axis in the standard residual stream gets one unweighted accumulated state.

The pre-LN magnitude problem makes this more than an aesthetic complaint. Each sublayer reads a normalized input and then writes its raw output into an unnormalized running stream. As depth grows, the stream norm grows too, so a new normalized sublayer output becomes a smaller fraction of the total. If a later sublayer wants to matter, it has to push against an accumulated vector that is already large. That explains why depth can look busy in the architecture but weak in effect: hidden norms climb, early information gets buried in a sum, and many deeper blocks can become surprisingly dispensable. I want a different depth-flow rule, but the rule cannot throw away the reason residuals worked in the first place. It still needs direct gradient routes, fixed hidden width, and a small amount of extra machinery.

The easiest knob is a scalar on the update. ReZero and LayerScale say `h_l = h_{l-1} + alpha_l f_{l-1}(h_{l-1})`; now a sublayer can be quiet or loud, but after training `alpha_l` is still a fixed coefficient, and the next layer still receives only the one compressed predecessor state. Highway networks make the gate input-dependent: `h_l = (1 - g_l) * h_{l-1} + g_l * f_{l-1}(h_{l-1})`. That is better, but gating the predecessor does not recover the individual earlier outputs that have already been added together. Once layer 3 and layer 9 have both been folded into `h_{16}`, a gate on `h_{16}` cannot ask for layer 3 by itself.

So cross-layer access is the missing ingredient. DenseNet gives every layer the preceding feature maps by concatenating them, then projects the large channel stack back down. That is real access, but it grows width with depth and then relies on a projection to squeeze the result. DenseFormer keeps the Transformer width fixed by learning scalar averages over past block representations. That is cheap and direct, yet the coefficients are static: every token receives the same depth mix. Hyper-Connections widen the residual stream into `m` channels of depth state and update them through `H_l = H_{l-1} A_l + f(H_{l-1} alpha_{l-1}) beta_{l-1}^T`; unrolling gives more depth capacity, but the mechanism is still a recurrence through an `m`-wide predecessor. MUDDFormer generates cross-layer weights from the current hidden state and splits query/key/value/residual streams, which moves in the direction I care about, but the routing machinery is heavy. MRLA-like layer attention also gives cross-layer gates, but its separable score form still looks like a recurrent state update.

I should put all of these into one algebraic picture. Let the sources be `v_0 = h_1` for the embedding and `v_i = f_i(h_i)` for later transformation outputs. Any depth-flow rule can be written as `h_l = sum_{i=0}^{l-1} M_{i->l} v_i`. For the plain residual, every valid causal entry is one: `M_{i->l} = 1`, an all-ones depth kernel with semiseparable rank one because the value factors as `1 * 1`. Highway has `M_{0->l}` as a carry product and `M_{i->l} = g_{i+1} prod_{j=i+2}^{l}(1 - g_j)` for `i >= 1`; those scalar products still factor through one depth state, so it is also semiseparable rank one. Hyper-Connections give `M_{i->l} = beta_i^T A^x_{i+1->l} alpha_l`, which is semiseparable rank `m`. These are not the same parameterization, but they share the same limitation: the effective depth interaction is constrained by a recurrent state of small rank.

That makes the attention analogy precise. In `beta_i^T A^x_{i+1->l} alpha_l`, `alpha_l` is a query issued by the destination layer, `beta_i` is a key attached to a source, and the product of `A` matrices is a depth-relative positional operator. It is attention over depth, but with a separable bilinear kernel, the same kind of structure that lets linear attention collapse into a recurrence. If the sequence axis escaped recurrent compression by moving from linear/separable attention to softmax attention over the whole prefix, the depth axis can try the same jump. Depth is small enough that the all-prefix operation is not scary: per token, scanning all earlier depth sources costs `O(L^2 d)` arithmetic and keeping the sources costs `O(Ld)` memory, and ordinary training already retains those activations for backpropagation.

So I want layer `l` to form its input from the actual earlier outputs:

```text
h_l = sum_{i=0}^{l-1} alpha_{i->l} v_i,     sum_i alpha_{i->l} = 1.
```

If I transform the values into another learned state, I am back to summaries of summaries, so the values have to be the stored representations themselves. For scores, the same vectors can serve as keys: the source content is what should make the weight vary across tokens and examples. The remaining choice is the query. A query projected from the current hidden state is expressive, but it couples the score computation to the sequential forward state and adds a `d x d` projection at every layer. A single learned pseudo-query `w_l in R^d` per destination layer is much cheaper. It is fixed for the layer, but the keys are content-dependent, so the weights still vary by token and example. It also means the queries for a group of destination layers are known before those layers execute, which leaves room to batch score computation.

The score cannot use raw source magnitudes. The whole problem started with depth-dependent growth in the residual stream; if I compute `exp(w_l^T v_i)`, a large-norm source can win just by scale. RMSNorm gives me the source direction at comparable scale without recentering it. The kernel becomes

```text
phi(w_l, v_i) = exp(w_l^T RMSNorm(v_i)),
alpha_{i->l} = phi(w_l, v_i) / sum_{j=0}^{l-1} phi(w_l, v_j).
```

The normalization belongs in the score, not in the value path. I want large vectors to stop dominating the softmax merely because they are large, but once a source is selected I still want to mix the raw representation it produced.

Softmax is doing a specific job here. Independent gates can turn many sources up at once; they do not force a choice. A softmax gives the destination a fixed probability budget over sources, so emphasizing one source takes mass away from the others. That is the retrieval behavior I want along depth. I also do not see a good reason to split the depth decision into many heads. Across sequence positions, different heads can seek different tokens; across depth, a source layer's output is usually useful as a whole representation. A single `d`-dimensional pseudo-query per destination layer is the simpler default.

The initialization has to be exact. If `w_l` starts random, every destination layer begins with an arbitrary preference over source depths. That injects a random routing bias before training has learned what the sources mean. If I set every `w_l = 0`, then for every source

```text
phi(0, v_i) = exp(0^T RMSNorm(v_i)) = exp(0) = 1.
```

All logits are equal, so the softmax is uniform: `alpha_{i->l} = 1/l` when there are `l` sources. The model starts from an equal-weight average over the available sources and learns deviations from that symmetric state. That is the clean reason for zero initialization; it is the difference between a neutral prior and random depth routing.

The embedding should be source zero. It is the one representation every later layer may need to recover, and in the additive residual it is only present after being mixed into the stream. Keeping `v_0 = h_1` in the source list gives every destination a direct route back to token identity.

Now the matrix picture says what changed. Plain residual gives the all-ones causal depth kernel, so its valid entries factor as `1 * 1`, and Highway still factors through one scalar carry state. Hyper-Connections raise the semiseparable rank to `m`, which is why they are more expressive but still recurrence-like. With softmax scores `alpha_{i->l} = softmax_i(w_l^T RMSNorm(v_i))`, the causal entries of `M` are dense and input-dependent; generically the resulting `L x L` depth-mixing matrix has rank `L` rather than a forced low semiseparable rank. This is the full-rank depth-mixing regime rather than another small-state recurrence.

The gradient story changes, and I need to keep it exact. The old additive recurrence has a unit-coefficient identity term in the product `prod(I + df_j/dh_j)`. A normalized mixture does not preserve that same unit `I` term. It gives direct differentiable paths from the loss to every source with nonzero attention weight, plus the score-gradient path through the keys. At zero initialization every available source has nonzero weight, so gradients are spread across all earlier outputs instead of being forced only through the immediate predecessor. The guarantee changes from "unit identity coefficient through each residual step" to "direct weighted access to all stored sources."

The tensor operation is small. Given a list of source tensors, each `(B, T, D)`, stack them along a source axis, normalize the stacked keys, score with the destination query, softmax over sources, and mix the raw stacked values:

```python
def attn_res(sources, query):
    stacked = torch.stack(sources, dim=0)                         # (S, B, T, D)
    keys = F.rms_norm(stacked, (stacked.size(-1),))               # keys only
    logits = torch.einsum('d, s b t d -> s b t', query, keys)
    weights = logits.softmax(dim=0)
    return torch.einsum('s b t, s b t d -> b t d', weights, stacked)
```

This operation is exactly `torch.stack`, `F.rms_norm`, the `'d, s b t d -> s b t'` score einsum, `softmax(dim=0)`, then the `'s b t, s b t d -> b t d'` value einsum. The source axis is the depth axis.

To wire it into a GPT block faithfully, I should treat the attention sublayer and the MLP sublayer as the depth-producing transformations. A standard block hides two residual additions inside `Block.forward`; here I do not call that forward method. I ask the depth rule for the hidden state feeding attention, compute the attention output as a new source, ask the depth rule again for the hidden state feeding the MLP, compute the MLP output as a new source, and continue. With `n_layer` Transformer blocks there are `2 * n_layer` destination sublayers, hence `2 * n_layer` pseudo-queries, plus a final readout query.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class FullAttentionResiduals(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(2 * config.n_layer, config.n_embd))
        self.query_out = nn.Parameter(torch.zeros(config.n_embd))

    def attend(self, sources, query):
        stacked = torch.stack(sources, dim=0)                         # (S, B, T, D)
        keys = F.rms_norm(stacked, (stacked.size(-1),))               # score with normalized keys
        logits = torch.einsum('d, s b t d -> s b t', query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum('s b t, s b t d -> b t d', weights, stacked)

    def start(self, x):
        return [x]                                                    # source 0 = token embedding

    def before_sublayer(self, sources, q_index):
        return self.attend(sources, self.queries[q_index])

    def after_sublayer(self, sources, output):
        sources.append(output)
        return sources

    def readout(self, sources):
        return self.attend(sources, self.query_out)

    def query_parameters(self):
        return [self.queries, self.query_out]

    def extra_parameter_ids(self):
        return {id(p) for p in self.query_parameters()}

    def optimizer_groups(self, learning_rate):
        return [{'params': self.query_parameters(), 'lr': learning_rate * 0.1, 'weight_decay': 0.0}]


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.depth_flow = FullAttentionResiduals(config)
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.transformer.wpe(pos)

        sources = self.depth_flow.start(x)
        q = 0
        for block in self.transformer.h:
            h = self.depth_flow.before_sublayer(sources, q)
            q += 1
            sources = self.depth_flow.after_sublayer(sources, block.attn(block.ln_1(h)))

            h = self.depth_flow.before_sublayer(sources, q)
            q += 1
            sources = self.depth_flow.after_sublayer(sources, block.mlp(block.ln_2(h)))

        x = self.depth_flow.readout(sources)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        query_ids = self.depth_flow.extra_parameter_ids()
        pd = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in pd.items() if p.dim() >= 2 and id(p) not in query_ids]
        nodecay = [p for n, p in pd.items() if p.dim() < 2 and id(p) not in query_ids]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        optim_groups.extend(self.depth_flow.optimizer_groups(learning_rate))
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```

If activation recomputation or pipeline parallelism makes all stored layer outputs expensive to keep and communicate, the same operation can be approximated by grouping consecutive sources into summaries and attending over those summaries. That is a scaling variant, not the core derivation. The core method is the dense version: store the embedding and every earlier sublayer output as sources, score them with zero-initialized learned pseudo-queries against RMS-normalized keys, use a softmax over the depth axis, and mix the raw values. The residual stream stops being a fixed unit-weight accumulator and becomes a learned full-rank depth mixing matrix while keeping the implementation to one stack, one RMSNorm, two einsums, and one source-axis softmax.
