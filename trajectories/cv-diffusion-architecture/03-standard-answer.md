**Problem (from rung 2).** Full-attn (FID 17.44/8.82/7.31) beat no-attn at every scale, confirming
attention is the coherence cure — but it placed attention at all four resolutions, including the
32×32 map where O(N²) is maximal and the shallow local features are too local to carry the long-range
structure attention is for. The suspicion: the fine-resolution attention is over-provisioned.

**Key idea.** Placement trades cost against structure along the resolution axis: cost falls toward
coarse maps, while meaningful global structure first rises (features grow abstract enough to hold
layout) then falls (the grid grows too coarse to hold layout — 4×4 is only 16 positions). The
interior optimum is **16×16**: modest quadratic cost (N=256), features abstract enough to carry
global layout and symmetry, grid still fine enough to hold real spatial structure. So place a single
attention-bearing block at 16×16 only, leaving 32×32 texture to cheap convolution. This is the
standard DDPM placement (`google/ddpm-cifar10-32`), here derived from the cost-vs-structure trade-off
rather than inherited.

**Why it works.** Attention at 16×16 buys the most coherence per unit of quadratic cost; concentrating
it there removes full-attn's wasted fine-resolution computation without removing the coordination.

**Scaffold edit / hyperparameters.** `down_block_types=("DownBlock2D","AttnDownBlock2D","DownBlock2D",
"DownBlock2D")` and the mirror `up_block_types=("UpBlock2D","UpBlock2D","AttnUpBlock2D","UpBlock2D")` —
attention only at the 16×16 level. Everything else is the shared DDPM config (group norm 32 groups,
`norm_eps=1e-6`, SiLU, positional time embedding, `flip_sin_to_cos=False`, `freq_shift=1`,
`downsample_padding=0`), channels from `BLOCK_OUT_CHANNELS`, `layers_per_block` from the env.

**What to watch.** Expect standard ≥ full-attn at medium and large (the targeted placement matches the
everywhere placement at a fraction of the cost — confirming over-provisioning), with the small scale
the one place full-attn's extra attention might still substitute for missing conv capacity.

```python
def build_model(device):
    """Standard DDPM architecture: attention at 16x16 only."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
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
