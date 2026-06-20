Fifteen minutes now, 5100 steps. The body is modernized; the optimizers are good. So I go back to the one
piece I haven't touched since the baseline: the input embedding and the output head, and specifically the
fact that they are the *same* matrix. Weight tying — `wte.weight = lm_head.weight` — has been standard
since GPT-2 because it saves ~39M parameters (a 50304×768 matrix is large) and was reported to help small
models generalize. But let me question whether tying still earns its keep in *this* regime, where I'm
racing wallclock to a fixed loss and have just zero-initialized everything else for gentle dynamics.

The two ends of the network do genuinely different jobs. The input embedding is a *lookup table*: it maps
a token id to a vector that seeds the residual stream, and what matters is that semantically related
tokens start nearby and that the vector is at a sensible scale for the first block. The output head is a
*classifier*: it maps the final residual vector to 50304 logits, and what matters is that its rows are
arranged so the dot product with the final hidden state ranks the correct next token highest. Tying forces
one matrix to be good at both. There's tension: the embedding wants a representation that's easy to *read
into* the residual stream at the input; the head wants rows that are easy to *score against* at the output,
after twelve blocks of transformation. With Muon now learning the body fast, the embedding/head are
increasingly the limiting factor, and making one matrix serve both roles caps how well either is learned.
Untying gives the optimizer two matrices to specialize — at the cost of the 39M parameters tying was
saving.

Is paying 39M parameters legitimate in a speedrun? It does *not* change the number of active parameters
per token or the inference cost — the head and embedding are each used exactly once per token regardless
of whether they share storage — and it doesn't change the body. So the extra params are a memory cost
during training, not a compute or throughput cost. In a race measured by wallclock-to-3.28, if untying
lets the model reach the bar in meaningfully fewer steps, the slightly larger parameter footprint is a
fine trade. (It is "any%" in the sense that total parameter count goes up, but active params and
throughput don't.) Worth trying.

Now, how do I initialize the *untied* head? With tying, the head inherited whatever the embedding was. Now
it's a fresh 50304×768 matrix and I get to choose. Follow the same principle that made the modern block
train gently: start the head's contribution at *zero*. Zero-initialize `lm_head.weight`. At step zero,
every logit is exactly 0, so the softmax is uniform over the vocabulary and the loss is exactly
log(vocab) — the maximum-entropy starting point, no random head injecting noise into the first gradients.
The head then has to *learn* to separate tokens from a clean uniform start, which matches the zero-init
residual projections in the body: nothing in the network asserts a confident, random opinion at
initialization. This is the same "earn your output from zero" pattern, now applied to the classifier.
(This zero-init-the-head trick echoes scaling-guide advice that it's a cheap, real win.)

One more thing the untie exposes. With tying, the embedding vector that seeded the residual stream was the
same matrix later RMS-normed all over the body — but the *raw* embedding output, straight out of the
lookup, has whatever scale the embedding table happens to have, and now that the head no longer shares
that matrix, the embedding is free to drift in scale during training in a way that changes the effective
input scale to the first block. The clean fix is to RMS-norm the embedding *immediately* after the lookup,
before the first block sees it: `x = norm(self.transformer.wte(idx))`. That pins the residual stream to
unit RMS at the input regardless of how the embedding table's scale evolves, so the first block always
receives a well-scaled input and the embedding's learned *direction* is decoupled from its learned
*magnitude*. It's the same normalization discipline the rest of the network already follows, now extended
to the very first activation.

So three coupled changes, all small: untie the embedding and head into two separate matrices, zero-init
the head so training starts from the uniform-logit maximum-entropy point, and RMS-norm the embedding right
after lookup so the input scale is pinned. The risk is the 39M extra parameters slowing convergence or
overfitting at this small data budget, but the zero-init head and the input norm should keep the early
dynamics clean, and the whole point is that two specialized matrices reach a given loss faster than one
overloaded one. I expect this to cut the step count again at the 3.28 bar.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        # head is now its own matrix, no longer tied to wte
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()   # start from uniform logits / max-entropy loss

    def forward(self, idx, target):
        x = self.transformer.wte(idx[None])     # token embeddings
        x = norm(x)                              # RMS-norm the embedding right after lookup
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss
```

The chain: a tied matrix is forced to be simultaneously a good input lookup table and a good output
classifier, two genuinely different jobs whose representations pull apart; untying lets the optimizer
specialize each, paying only training memory (not active params or throughput); the fresh head is
zero-initialized so loss starts at log(vocab) with no random head noise; and the embedding is RMS-normed
immediately after lookup so the first block always sees a unit-scale input. Two specialized, gently-started
matrices should reach 3.28 in fewer steps than one overloaded one.
