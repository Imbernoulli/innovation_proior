The four shortcuts, added incrementally on top of the 10.8-minute record, cut the step count to 3200 and the wallclock to ~8.2 minutes. One of them nags at me as under-exploited: the embed shortcut paid off, and the *reason* it paid off is general — giving a deep block a direct path back to an *earlier representation* helps. But I framed it as a single source, $x_0$, routed forward to every consumer, and $x_0$ is just the earliest representation. Every block produces a representation, and the deep blocks might benefit from direct paths back to the richer, partially-processed activations from the middle of the network, not only the raw embedding. The embed shortcut is the degenerate, one-source case of a much more general skip-connection pattern over depth; the question is which earlier activations to route to which later blocks, and how to pair them, without ending up in a mess of connections and parameters.

I propose **U-net encoder/decoder skip connections over the transformer's depth** (the design is by @brendanh0gan), generalizing the embed shortcut from "one source → every block" to a full symmetric skip structure. There is a clean, proven structure for exactly this — the U-net from vision, whose insight is *symmetric long skips*: the first encoder layer, holding the freshest detail, connects to the *last* decoder layer reconstructing the finest output; the deepest encoder layer connects to the first decoder layer right after the bottleneck. Early-to-late, symmetrically around the middle. I map that straight onto depth. Treat the first half of the twelve blocks as an **encoder** and the second half as a **decoder**: run the encoder normally but *save* each encoder block's output on a stack, then in the decoder add the saved encoder activations back in, each decoder block getting a long skip from its symmetric partner.

What makes the pairing fall out for free is the stack discipline. I push encoder layer 0, then 1, …, up to layer 5 (with `num_encoder_layers = 6`), and in the decoder I `pop()` before each block. Last-in-first-out means the first decoder block pops the *last* pushed encoder activation — layer 5, the deepest, closest to the bottleneck — and the last decoder block pops the *first* — layer 0, the freshest. That is exactly the symmetric U-net pairing, encoder layer $i$ with decoder layer $(\text{num\_decoder\_layers} - 1 - i)$, with no index arithmetic and no bookkeeping: the LIFO stack *is* the symmetry. The decoder, doing the final shaping of the representation before the head, gets to reuse the encoder's cleaner intermediate features directly rather than reconstructing them from the residual stream alone. How much of each encoder activation each decoder block takes is, as everywhere in this network, not hardcoded but learned: a per-decoder-layer **learnable skip weight** `self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))`, initialized to ones so I start by adding the full encoder activation and let training scale each connection down or up. These are 1-D scalar params, so they ride along under Adam with the other scalars, not Muon. The decoder loop is just `x = x + self.skip_weights[i] * skip_connections.pop()` then run the block; the encoder loop is the plain block sequence appending each output; and the head keeps the tanh softcap from the prior rung, `logits = 30 * torch.tanh(self.lm_head(x).float() / 30)`.

There is a second-order benefit I want to cash in. These long skips do not just shuttle features forward — they *shorten the gradient path*. A gradient at the head can now reach an encoder layer directly through its skip connection instead of only by backpropagating through every intervening decoder block. That is the same property that made residual connections trainable at depth in the first place, now operating across the whole network rather than within a block, and shorter gradient paths mean a better-conditioned landscape. A better-conditioned landscape tolerates a *larger* learning rate without diverging, so I pair the U-net with a **doubled learning rate**. Doubling the LR normally risks instability, but with the skips damping the gradient pathologies I expect it to hold, and a $2\times$ LR on a well-conditioned problem is the cheapest way there is to cover the same loss drop in fewer steps. The U-net buys the conditioning; the doubled LR cashes it in as wallclock.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2   # first half = encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        # learnable skip-connection weights for the decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, target):
        x = norm(self.transformer.wte(idx[None]))
        x0 = x
        # Encoder pass — first half of the blocks
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, x0)  # block also threads the value-residual signal (rung 4)
            skip_connections.append(x)
        # Decoder pass — remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, x0)
        x = norm(x)
        logits = 30 * torch.tanh(self.lm_head(x).float() / 30)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

# the skip weights are 1D params optimized as scalar_params by Adam
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
```
