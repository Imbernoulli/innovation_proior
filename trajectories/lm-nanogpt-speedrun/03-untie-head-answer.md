**Problem (from step 2).** With the body modernized and Muon learning it fast (15.2 min, 5100 steps), the
input embedding and output head — still the *same* tied matrix since GPT-2 — are now the limiting factor.
Tying forces one matrix to be both a good input lookup table (seed the residual stream) and a good output
classifier (rank the next token from the final hidden state), two genuinely different jobs whose ideal
representations pull apart.

**Key idea (untie + zero-init head + post-embed norm).** Split the tied matrix into two independent ones
so the optimizer can specialize each. Initialize the now-separate `lm_head.weight` to **zero**, so at step
zero every logit is 0, the softmax is uniform, and the loss starts at the maximum-entropy value log(vocab)
— no random head injecting noise into the first gradients (the same "earn your output from zero" pattern as
the zero-init residual projections). RMS-**norm the embedding immediately after lookup** so the residual
stream is pinned to unit RMS at the input, decoupling the embedding's learned direction from its drifting
magnitude now that the head no longer shares the matrix.

**Why it works.** Two specialized, gently-initialized matrices reach a given loss in fewer steps than one
overloaded one. Untying costs ~39M parameters of *training memory* but does **not** change active
parameters per token or inference throughput (each of head/embedding is used once per token regardless of
shared storage), so in a wallclock-to-3.28 race the trade is favorable. The zero-init head and post-embed
norm keep the early dynamics clean despite the larger footprint.

**Change / code.** `wte` and `lm_head` are separate; `lm_head.weight.data.zero_()`; `x = norm(wte(idx))`
right after lookup.

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
