The modernized block reaches the bar in 5100 steps at 178.89 ms a step — about 15.2 minutes, and at a better val loss (3.2741) — so the architecture both fits faster and runs cheaper. With the body modernized and Muon learning it fast, the limiting factor is now the one piece I have not touched since the baseline: the input embedding and the output head, and specifically the fact that they are the *same* matrix. Weight tying — `wte.weight = lm_head.weight` — has been standard since GPT-2 because it saves ~39M parameters and was reported to help small models generalize, but the two ends of the network do genuinely different jobs. The input embedding is a *lookup table*: it maps a token id to a vector that seeds the residual stream, and what matters is that semantically related tokens start nearby at a sensible scale for the first block. The output head is a *classifier*: it maps the final residual vector to 50304 logits, and what matters is that its rows are arranged so the dot product with the final hidden state ranks the correct next token highest. Tying forces one matrix to be good at both — the embedding wants a representation easy to *read into* the stream at the input, the head wants rows easy to *score against* after twelve blocks — and with Muon now learning the body fast, that overloaded matrix caps how well either role is learned.

I propose to **untie the embedding and head, zero-initialize the head, and RMS-norm the embedding right after lookup** — three small, coupled changes. Untying splits the shared matrix into two independent ones so the optimizer can specialize each. The cost is paying back the ~39M parameters tying was saving, but this is legitimate in a wallclock race: it does *not* change the number of active parameters per token (head and embedding are each used exactly once per token whether or not they share storage) and it does not touch the body or inference throughput. The extra parameters are training *memory*, not compute, so if two specialized matrices reach the bar in meaningfully fewer steps the larger footprint is a fine trade.

The second change is how to initialize the now-separate head. With tying it inherited the embedding; now it is a fresh $50304\times768$ matrix and I follow the same principle that made the modern block train gently — start its contribution at *zero*. I **zero-initialize `lm_head.weight`**, so at step zero every logit is exactly $0$, the softmax is uniform over the vocabulary, and the loss starts at the maximum-entropy value $\log(\text{vocab})$. No random head injects noise into the first gradients; the head has to *learn* to separate tokens from a clean uniform start, the same "earn your output from zero" pattern as the zero-init residual projections — nothing in the network asserts a confident random opinion at initialization.

The third change is exposed by the untie. With tying, the matrix that seeded the residual stream was the same one RMS-normed all over the body, but the *raw* embedding output has whatever scale the table happens to have, and now that the head no longer shares it the embedding is free to drift in scale during training in a way that changes the effective input scale to the first block. The clean fix is to **RMS-norm the embedding immediately after lookup**, `x = norm(self.transformer.wte(idx))`, before the first block sees it. This pins the residual stream to unit RMS at the input regardless of how the embedding table's scale evolves, so the first block always receives a well-scaled input and the embedding's learned *direction* is decoupled from its learned *magnitude* — the same normalization discipline the rest of the network already follows, now extended to the very first activation. Two specialized, gently-started matrices, with the embedding's scale pinned, should reach 3.28 in fewer steps than one overloaded one; the zero-init head and the input norm keep the early dynamics clean despite the larger footprint.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)  # untied: own matrix
        self.lm_head.weight.data.zero_()                               # uniform logits at init

    def forward(self, idx, target):
        x = self.transformer.wte(idx[None])
        x = norm(x)                          # RMS-norm embedding right after lookup
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss
```
