The standard pre-LN decoder treats depth as a fixed unit-weight accumulator. Each block reads the current stream, adds its transformed output with coefficient one, and passes the result upward. Unrolled, the state at layer l is the token embedding plus every earlier sublayer output, all weighted equally. This gives the classic residual gradient highway, but it also means the stream norm grows with depth, newer outputs are buried in an already-large sum, and many deep layers become surprisingly dispensable. The real limitation is not trainability but expressiveness along the depth axis: every destination layer sees one compressed predecessor state instead of being able to select which earlier representations matter for the current computation.

Existing fixes keep the bottleneck in different forms. ReZero and LayerScale attach learned scalar gains to sublayer outputs, yet after training those gains are static and the next layer still receives a single accumulated predecessor. Highway networks make the gate input-dependent, but unrolling shows the effective kernel still factors through one scalar carry state. DenseNet grants cross-layer access by concatenation, but it grows channel width and relies on a projection, so it does not fit the fixed-width residual stream of a GPT. DenseFormer learns scalar averages over past block outputs, but those averages are the same for every token. Hyper-Connections widen the stream into m parallel states and obtain an m-semiseparable depth kernel, which is more expressive than a single stream but still recurrence-like. Dynamic dense connections and layer-attention variants move toward content-dependent routing, yet their separable score forms or heavy routing machinery still constrain the depth mixing matrix to a low effective rank. What is missing is a full-rank, content-dependent depth mixing operation that keeps the hidden width fixed and slots into ordinary GPT training.

I propose Full Attention Residuals, or Full AttnRes. The idea is to replace the fixed additive residual update with softmax attention over earlier depth sources. Concretely, let v0 be the token embedding and vi for i greater than or equal to one be the raw output of the i-th sublayer. The input to destination layer l is formed as a weighted mixture h_l = sum_i alpha_{i->l} v_i, where the coefficients are a softmax over depth: alpha_{i->l} = exp(w_l^T RMSNorm(v_i)) / sum_j exp(w_l^T RMSNorm(v_j)). Each destination layer owns a single learned pseudo-query w_l in R^d, initialized to zero. The keys are RMS-normalized earlier outputs, so a large-norm source cannot dominate the score merely because its magnitude is large; the values that are mixed are the raw, unnormalized outputs, so once a source is selected it contributes its full representation. Because the keys are content-dependent, the routing weights vary across tokens and examples even though the query vector for a layer is fixed.

Zero initialization is deliberate. When every w_l is zero, every score becomes exp(0) = 1, so the softmax is uniform. The model therefore starts from an equal-weight average over all available depth sources and learns deviations from that neutral prior. This avoids injecting a random depth preference before training has learned what the different sources mean. The embedding is kept as source zero, giving every later layer a direct route back to token identity. Compared with the standard residual, whose causal depth kernel is rank one because every valid entry equals one, Full AttnRes yields a dense, input-dependent L-by-L depth-mixing matrix that is generically full rank. The trainability story shifts accordingly: instead of a unit identity coefficient through each residual step, the architecture provides direct weighted gradients from the loss to every stored source with nonzero attention weight. At initialization that weight is spread uniformly, so gradients reach all earlier layers rather than being forced through the immediate predecessor alone.

The implementation stores the embedding and every subsequent sublayer output, scores them with the destination pseudo-queries against RMS-normalized keys, applies a softmax over the depth axis, and mixes the raw values. Per token the cost is O(L^2 d) arithmetic and O(Ld) activation memory, which in ordinary training overlaps with the activations already retained for backpropagation. The query parameters are grouped separately in the optimizer at 0.1 times the base learning rate and no weight decay, because they are highly leveraged: a small change in a query reshapes the entire mixture entering a sublayer.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class FullAttentionResiduals(nn.Module):
    def __init__(self, config):
        super().__init__()
        # One query per sublayer destination (attention + MLP per layer)
        # plus one query for the final readout before the head.
        self.queries = nn.Parameter(torch.zeros(2 * config.n_layer, config.n_embd))
        self.query_out = nn.Parameter(torch.zeros(config.n_embd))

    def attend(self, sources, query):
        # sources: list of (B, T, D) tensors; query: (D,)
        stacked = torch.stack(sources, dim=0)                 # (S, B, T, D)
        keys = F.rms_norm(stacked, (stacked.size(-1),))       # score keys only
        logits = torch.einsum('d, s b t d -> s b t', query, keys)
        weights = logits.softmax(dim=0)                       # (S, B, T)
        return torch.einsum('s b t, s b t d -> b t d', weights, stacked)

    def start(self, x):
        return [x]                                            # source 0 = embedding

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
        return [{'params': self.query_parameters(),
                 'lr': learning_rate * 0.1, 'weight_decay': 0.0}]


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
            sources = self.depth_flow.after_sublayer(
                sources, block.attn(block.ln_1(h)))

            h = self.depth_flow.before_sublayer(sources, q)
            q += 1
            sources = self.depth_flow.after_sublayer(
                sources, block.mlp(block.ln_2(h)))

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
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in query_ids]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in query_ids]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        optim_groups.extend(self.depth_flow.optimizer_groups(learning_rate))
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```
