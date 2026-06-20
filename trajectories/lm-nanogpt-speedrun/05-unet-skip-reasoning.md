Eight point two minutes now, 3200 steps. The bundle of shortcuts worked, and one of them is nagging at me as
under-exploited. The embed shortcut — feeding the original normalized embedding x0 into every block through
two learnable scalars — paid off, and the reason it paid off is general: giving a deep block a direct path
back to an *earlier representation* helps. I framed it as "input → every block," a single source (x0) routed
forward to every consumer. But x0 is just the earliest representation. Every block produces a representation,
and the deep blocks might benefit from direct paths back to representations that aren't x0 — the richer,
partially-processed activations from the middle of the network, not just the raw embedding. The embed
shortcut is the degenerate, one-source case of a much more general skip-connection pattern over depth. Let me
generalize it.

The question is which earlier activations to route to which later blocks, and how to pair them. If I just
dumped every encoder activation into every decoder block I'd have a mess of connections and a lot of
parameters. There's a clean, proven structure for exactly this — the U-net from vision (the
encoder/decoder shape that saved spatial detail in segmentation by connecting matching-resolution layers
across the bottleneck). The U-net insight is *symmetric long skips*: the first encoder layer, which holds the
freshest detail, connects to the *last* decoder layer, which is reconstructing the finest output; the
deepest encoder layer connects to the first decoder layer right after the bottleneck. Early-to-late,
symmetrically around the middle. I can map that straight onto the depth of the transformer. Treat the first
half of my twelve blocks as an "encoder" and the second half as a "decoder." Run the encoder normally, but
*save* each encoder block's output. Then, running the decoder, add the saved encoder activations back in —
each decoder block gets a long skip connection from its symmetric partner in the encoder. This is the U-net
pattern over transformer depth, and the design is by @brendanh0gan.

The pairing falls out naturally if I store the encoder outputs on a stack and pop them in the decoder. I push
encoder layer 0, then 1, …, up to layer 5 (with num_encoder_layers = 6). In the decoder I pop before each
block: the first decoder block pops the *last* pushed encoder activation (layer 5, the deepest, closest to
the bottleneck), and the last decoder block pops the *first* pushed one (layer 0, the freshest). Last-in-
first-out gives me exactly the symmetric U-net pairing — encoder layer i pairs with decoder layer
(num_decoder_layers − 1 − i) — for free, just from `skip_connections.pop()`. No bookkeeping, no index
arithmetic; the stack discipline *is* the symmetry. The decoder, which is doing the final shaping of the
representation before the head, gets to reuse the encoder's cleaner intermediate features directly rather
than reconstructing them from the residual stream alone.

How much of each encoder activation should each decoder block take? Same answer as everywhere else in this
network: don't hardcode it, let the model choose. I give the decoder a learnable skip weight per decoder
layer, `self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))`, initialized to ones so I
start by adding the full encoder activation and let training scale each connection down (or up) as it likes.
These are 1-D scalar parameters, so they ride along with the other scalar params under Adam, not Muon —
`scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]`. In the forward pass the
decoder loop is just: pop the partner, add `self.skip_weights[i]` times it, then run the block —
`x = x + self.skip_weights[i] * skip_connections.pop()` then `x = self.transformer.h[layer](x, x0)`. The
encoder loop is the plain block sequence, appending each output. And the head stays as it was, with the tanh
softcap I added last time: `logits = 30 * torch.tanh(self.lm_head(x).float() / 30)`.

There's a second-order benefit I want to exploit. These long skips don't just shuttle features forward —
they shorten the gradient path. A gradient at the head can now reach an encoder layer directly through its
skip connection instead of only by backpropagating through every intervening decoder block. That's the same
property that made residual connections trainable at depth in the first place, now operating across the whole
network rather than within a block. Cleaner, shorter gradient paths mean the optimization landscape is better
conditioned, and a better-conditioned landscape can tolerate a *larger* learning rate without diverging. So I
should pair the U-net with a learning-rate increase — doubling the LR. Normally doubling the LR risks
instability, but with the skip connections damping the gradient pathologies I expect it to hold, and a 2× LR
on a well-conditioned problem is the cheapest way there is to cover the same loss drop in fewer steps. The
U-net buys the conditioning; the doubled LR cashes it in as wallclock.

So: split the twelve blocks into a six-block encoder and a six-block decoder; save each encoder output on a
stack; in the decoder, pop in LIFO order to recover the symmetric U-net pairing and add the partner back with
a learnable per-decoder-layer skip weight initialized to one; and double the learning rate because the
shortened gradient paths let me. It's the embed shortcut's principle — direct paths to earlier
representations — generalized from one source to a full symmetric skip structure over depth. I expect the
richer skips plus the doubled LR to cut the step count again at the 3.28 bar.

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

The chain: the embed shortcut showed that direct paths to earlier representations pay off, so I generalize it
from one source (x0) to a full U-net over depth — first six blocks encode and push their outputs on a stack,
last six decode and pop them in LIFO order, which gives the symmetric early-to-late U-net pairing for free,
each long skip scaled by a learnable per-decoder-layer weight initialized to one; and because those skips
shorten the gradient paths and better-condition the landscape, I double the learning rate to cash the
conditioning out as fewer steps to 3.28.
