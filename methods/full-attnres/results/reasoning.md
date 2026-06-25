Let me start with the residual stream, because the name hides two different jobs. In a pre-LN stack I usually write a layer as `h_l = h_{l-1} + f_{l-1}(h_{l-1})`, with the normalization folded into `f`. The backward story is the familiar one: if I differentiate from the loss at the top back to an intermediate state, I get `dL/dh_l = (dL/dh_L) * prod_{j=l}^{L-1}(I + df_j/dh_j)`, using row-vector gradients. When I expand that product, there is always a term made only of the identity matrices. That is the clean gradient route that makes depth trainable.

But the forward story is just as important. The same recurrence expands to `h_l = h_1 + sum_{i=1}^{l-1} f_i(h_i)`. So the residual stream is also a depth aggregation rule: it says layer `l` reads the embedding plus every earlier transformation output, each with coefficient exactly one. That is a surprisingly rigid rule. The sequence axis gets self-attention, experts get routers, gates can depend on the token, but the depth axis in the standard residual stream gets one unweighted accumulated state.

The pre-LN magnitude problem makes this more than an aesthetic complaint. Each sublayer reads a normalized input and then writes its raw output into an unnormalized running stream. As depth grows, the stream norm grows too, so a new normalized sublayer output becomes a smaller fraction of the total. If a later sublayer wants to matter, it has to push against an accumulated vector that is already large. That is consistent with what depth diagnostics report: hidden norms climb, early information gets buried in a sum, and many deeper blocks turn out to be surprisingly dispensable. So I want a different depth-flow rule, but the rule cannot throw away the reason residuals worked in the first place. It still needs direct gradient routes, fixed hidden width, and a small amount of extra machinery.

The easiest knob is a scalar on the update. ReZero and LayerScale say `h_l = h_{l-1} + alpha_l f_{l-1}(h_{l-1})`; now a sublayer can be quiet or loud, but after training `alpha_l` is still a fixed coefficient, and the next layer still receives only the one compressed predecessor state. Highway networks make the gate input-dependent: `h_l = (1 - g_l) * h_{l-1} + g_l * f_{l-1}(h_{l-1})`. That is better, but gating the predecessor does not recover the individual earlier outputs that have already been added together. Once layer 3 and layer 9 have both been folded into `h_{16}`, a gate on `h_{16}` cannot ask for layer 3 by itself.

So cross-layer access is the missing ingredient. DenseNet gives every layer the preceding feature maps by concatenating them, then projects the large channel stack back down. That is real access, but it grows width with depth and then relies on a projection to squeeze the result. DenseFormer keeps the Transformer width fixed by learning scalar averages over past block representations. That is cheap and direct, yet the coefficients are static: every token receives the same depth mix. Hyper-Connections widen the residual stream into `m` channels of depth state and update them through `H_l = H_{l-1} A_l + f(H_{l-1} alpha_{l-1}) beta_{l-1}^T`; unrolling gives more depth capacity, but the mechanism is still a recurrence through an `m`-wide predecessor. MUDDFormer generates cross-layer weights from the current hidden state and splits query/key/value/residual streams, which moves in the direction I care about, but the routing machinery is heavy. MRLA-like layer attention also gives cross-layer gates, but its separable score form still looks like a recurrent state update.

I should put all of these into one algebraic picture so I can compare them by the same yardstick. Let the sources be `v_0 = h_1` for the embedding and `v_i = f_i(h_i)` for later transformation outputs. Any depth-flow rule can be written as `h_l = sum_{i=0}^{l-1} M_{i->l} v_i`, an `L x L` lower-triangular depth-mixing matrix `M` acting on the source list. For the plain residual, every valid causal entry is one: `M_{i->l} = 1`. For Highway, `M_{0->l}` is a carry product and `M_{i->l} = g_{i+1} prod_{j=i+2}^{l}(1 - g_j)` for `i >= 1`. For Hyper-Connections, unrolling gives `M_{i->l} = beta_i^T A^x_{i+1->l} alpha_l`. These look different on the page, and I want to know whether they are different in the way that matters, or whether the same constraint hides inside all three.

The natural invariant is the rank of the off-diagonal blocks of `M`. A kernel is called `r`-semiseparable when every block of `M` taken with all its rows below all its columns has rank at most `r`; that block rank is exactly what a recurrent state of width `r` can carry across a depth cut. So let me actually compute it on a small case rather than eyeball the formulas. Take `L = 12` and cut at `t = 6`, so the block is destinations `6..11` against sources `0..5`, a `6 x 6` block whose rank could in principle be as large as 6.

Plain residual: the block is all ones, so its rank is 1. Highway: I fill `M_{i->l}` from the carry products with random gates, take the same `6 x 6` block, and its rank comes out 1 as well — the scalar carry is a width-1 state, exactly as the formula suggests.

Hyper-Connections should be the interesting one, since `m` is supposed to set the rank. I build `M_{i->l} = beta_i^T (prod A_j) alpha_l` with width `m = 2`, random `A_j`, `alpha_l`, `beta_i`, and measure the block rank. It comes out 6, not 2. That is wrong for the `m`-semiseparable story, so either the claim is weaker than advertised or I built it wrong. Looking again at what I wrote: I formed the product `prod_j A_j` by multiplying the `A_j` in an arbitrary order for each `(i,l)` pair. That is not a transition operator — a genuine depth transition has to compose as `A_{l-1} A_{l-2} ... A_{i+1}`, the same factors in the same descending order, so that the segment from a source to a destination is the product of the segments through any intermediate layer. Once I fix the ordering to that proper descending product, the block rank for `m = 2` drops to 2, for `m = 4` it is 4, and for `m = 1` it is 1. So the `m`-semiseparable claim does hold, but only because the transition genuinely composes; the rank-`m` bound is a property of carrying an `m`-wide state across the cut, and my first version was silently carrying a full-rank junk operator instead. Good — now the three baselines line up: residual and Highway sit at block rank 1, Hyper-Connections at block rank `m`. They all cap the cross-depth interaction at a state width chosen in advance.

So if I want to escape that cap, the score between a destination and a source cannot factor through a fixed-width state. The `beta_i^T A^x_{i+1->l} alpha_l` form is exactly a separable bilinear kernel — `alpha_l` a query issued by the destination, `beta_i` a key on the source, the `A` product a depth-relative positional operator — and that separability is the same structure that lets linear attention collapse into a recurrence. The sequence axis already has a way out of recurrent compression: replace the separable/linear score by a softmax over the whole prefix. The depth axis can try the same jump. Depth is small enough that the all-prefix operation is affordable: per token, scoring against all earlier depth sources is `L` dot products of width `d` at each of `L` destinations, so `O(L^2 d)` arithmetic, and the sources themselves are `L` vectors of width `d`, so `O(Ld)` memory. For `L = 24` blocks (so roughly `2L` sublayer sources) and `d = 1024`, that source memory is the same order as activations that backpropagation already keeps, so the storage is not a new cost so much as a reuse of what is retained anyway.

So let me have layer `l` form its input from the actual earlier outputs with a normalized mixture:

```text
h_l = sum_{i=0}^{l-1} alpha_{i->l} v_i,     sum_i alpha_{i->l} = 1.
```

If I transform the values into another learned state, I am back to summaries of summaries, so the values have to be the stored representations themselves. For scores, the same vectors can serve as keys: the source content is what should make the weight vary across tokens and examples. The remaining choice is the query. A query projected from the current hidden state is expressive, but it couples the score computation to the sequential forward state and adds a `d x d` projection at every layer. A single learned pseudo-query `w_l in R^d` per destination layer is much cheaper. It is fixed for the layer, but the keys are content-dependent, so the weights still vary by token and example. It also means the queries for a group of destination layers are known before those layers execute, which leaves room to batch score computation.

The score cannot use raw source magnitudes. The whole problem started with depth-dependent growth in the residual stream; if I compute `exp(w_l^T v_i)`, a large-norm source can win just by scale. RMSNorm gives me the source direction at comparable scale without recentering it. Let me check that this actually neutralizes scale and not just in principle. Take two sources pointing the same direction, one of unit norm and one fifty times larger, and a query aligned with that direction. With RMSNorm applied to the keys, their logits come out 2.828 and 2.828 — equal to four digits, because RMSNorm has stripped the magnitude and left only the shared direction. Without it, the logits are 1 and 50, and the softmax hands almost everything to the larger source purely because it is larger. That is the failure mode I was worried about, and normalizing the keys removes it. So the kernel is

```text
phi(w_l, v_i) = exp(w_l^T RMSNorm(v_i)),
alpha_{i->l} = phi(w_l, v_i) / sum_{j=0}^{l-1} phi(w_l, v_j).
```

The normalization belongs in the score, not in the value path. The check above is exactly why: large vectors should stop dominating the softmax merely because they are large, but once a source is selected I still want to mix the raw representation it produced, magnitude and all.

Why softmax rather than independent gates. Independent gates can turn many sources up at once; they do not force a choice. A softmax gives the destination a fixed probability budget over sources, so emphasizing one source takes mass away from the others. Along depth, where a single source layer is usually useful as a whole representation, that competition is what I want, and it is also the reason I do not split the depth decision into heads: across sequence positions different heads can chase different tokens, but across depth a single `d`-dimensional pseudo-query per destination is the simpler default and I have no evidence yet that depth wants the extra heads.

The initialization has to be exact. If `w_l` starts random, every destination layer begins with an arbitrary preference over source depths, which injects a random routing bias before training has learned what the sources mean. If instead I set every `w_l = 0`, then for every source

```text
phi(0, v_i) = exp(0^T RMSNorm(v_i)) = exp(0) = 1.
```

All logits are equal, so the softmax is uniform. Let me read off what that gives for the first few destinations: at one available source the weight is `1/1`, at three sources it is `1/3` each, at six sources it is `1/6` each — the model starts from an equal-weight average over whatever sources exist and learns deviations from that symmetric state. That is the difference between a neutral prior and random depth routing, and it costs nothing to arrange.

The embedding should be source zero. It is the one representation every later layer may need to recover, and in the additive residual it is only present after being mixed into the stream. Keeping `v_0 = h_1` in the source list gives every destination a direct route back to token identity.

Now I can go back to the matrix and see what the softmax kernel does to the block rank that pinned the baselines at a fixed width. With `alpha_{i->l} = softmax_i(w_l^T RMSNorm(v_i))`, the entries no longer factor through a width-`m` state. On the same `L = 12`, `t = 6` cut, with random normalized keys and random queries, the `6 x 6` off-diagonal block comes out rank 6 — the full structural maximum for a block that size. So the contrast is concrete: residual 1, Highway 1, Hyper-Connections `m`, and this kernel saturates the block. Repeating at larger `L` with a larger key width, the block rank tracks the block size rather than flattening at a constant, which is the behavior I wanted: the cross-depth interaction is not capped by a state width fixed in advance.

The gradient story changes, and I want to be careful about what survives. The additive recurrence has a unit-coefficient identity term in the product `prod(I + df_j/dh_j)`; a normalized mixture does not preserve that same unit `I` term. What it does give is a direct differentiable path from the loss to every source with nonzero attention weight, plus the score-gradient path through the keys. At zero initialization every available source has weight `1/l > 0`, so at the start of training gradients are spread across all earlier outputs rather than forced through the immediate predecessor — the uniform-weight check above is also the statement that no source is gradient-starved at init. So the property I keep is not "unit identity coefficient through each residual step" but "weighted, nonzero direct access from the loss to every stored source."

The tensor operation is small. Given a list of source tensors, each `(B, T, D)`, stack them along a source axis, normalize the stacked keys, score with the destination query, softmax over sources, and mix the raw stacked values:

```python
def attn_res(sources, query):
    stacked = torch.stack(sources, dim=0)                         # (S, B, T, D)
    keys = F.rms_norm(stacked, (stacked.size(-1),))               # keys only
    logits = torch.einsum('d, s b t d -> s b t', query, keys)
    weights = logits.softmax(dim=0)
    return torch.einsum('s b t, s b t d -> b t d', weights, stacked)
```

The operation is one `torch.stack`, one `F.rms_norm`, the `'d, s b t d -> s b t'` score einsum, `softmax(dim=0)`, then the `'s b t, s b t d -> b t d'` value einsum. The source axis is the depth axis, and the softmax over `dim=0` is the depth competition.

To wire it into a GPT block faithfully, I treat the attention sublayer and the MLP sublayer as the depth-producing transformations. A standard block hides two residual additions inside `Block.forward`; here I do not call that forward method. I ask the depth rule for the hidden state feeding attention, compute the attention output as a new source, ask the depth rule again for the hidden state feeding the MLP, compute the MLP output as a new source, and continue. With `n_layer` Transformer blocks there are `2 * n_layer` destination sublayers, hence `2 * n_layer` pseudo-queries, plus a final readout query. Let me trace the bookkeeping on a tiny `n_layer = 2` to be sure the counts are right: the four sublayer queries see 1, 2, 3, and 4 sources in turn (embedding only, then each new sublayer output appended before the next query reads), and the readout sees 5 sources, which is `2 * n_layer + 1`. The four pseudo-queries plus one readout query match `self.queries` of shape `(2 * n_layer, d)` plus `self.query_out`. And feeding zero queries through this two-layer trace, the readout weights come out `[0.2, 0.2, 0.2, 0.2, 0.2]` over the five sources — the uniform start, confirmed end to end through the actual forward path rather than just at the kernel.

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

If activation recomputation or pipeline parallelism makes all stored layer outputs expensive to keep and communicate, the same operation can be approximated by grouping consecutive sources into summaries and attending over those summaries. That is a scaling variant, not the core derivation. The core method is the dense version: store the embedding and every earlier sublayer output as sources, score them with zero-initialized learned pseudo-queries against RMS-normalized keys, use a softmax over the depth axis, and mix the raw values. The residual stream stops being a fixed unit-weight accumulator — which the block-rank check pinned at 1 — and becomes a learned depth-mixing matrix whose off-diagonal blocks fill out their full rank, while the implementation stays one stack, one RMSNorm, two einsums, and one source-axis softmax.
