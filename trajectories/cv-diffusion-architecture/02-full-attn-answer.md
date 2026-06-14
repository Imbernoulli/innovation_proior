**Problem (from rung 1).** No-attn (FID 21.39/11.25/9.55 small/medium/large) is a long-range-coherence
failure, not a capacity failure: even at ~140M params a purely-convolutional denoiser has no direct
way to couple distant image positions, so every long-range dependency is smeared across a fragile
chain of local conv layers. The omitted operation is per-resolution self-attention.

**Key idea.** Self-attention couples any two spatial positions in one layer with a content-dependent
weight — the every-to-every operation convolution cannot do. As a feature-map block: 1×1
query/key/value projections, a `√d`-scaled softmax over all positions, a value average, wrapped in an
identity-at-init residual. The binding O(N²) cost in positions N = H·W makes attention *placement*
across the four resolution levels (32/16/8/4) the design space. This rung takes the maximal placement
— self-attention at *every* resolution (`AttnDownBlock2D`/`AttnUpBlock2D` everywhere) — to bracket the
cure from above and test whether unrationed global mixing recovers the coherence no-attn lost.

**Why it works.** Direct content-addressed coupling is strictly more than the indirect multi-layer
coupling no-attn relied on; multi-head attention (per `attention_head_dim`) lets distinct heads
specialize to distinct relations at near the cost of one head.

**Scaffold edit / hyperparameters.** Swap the block tuples to `("AttnDownBlock2D",)×4` and
`("AttnUpBlock2D",)×4`; everything else is the shared DDPM config, unchanged from rung 1, so the only
delta is attention placement.

**What to watch.** Expect the biggest *relative* win at the small scale (coherence, not capacity, was
its bottleneck). The risk: attention at 32×32 is the most expensive block on the ladder, and if
CIFAR's long-range structure lives mostly at coarser feature maps the fine-resolution attention is
idle — which would name a cheaper, targeted placement as the next move.

```python
def build_model(device):
    """Full-attention: self-attention at every resolution."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        layers_per_block=layers,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
```
