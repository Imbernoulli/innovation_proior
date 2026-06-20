I have a fixed pretrained LLM and a single instruction: make it smaller and faster to serve without
retraining it. The weights are FP16; the obvious thing to do is store them in fewer bits. Before I
reach for anything clever I want to know exactly how far the simplest possible scheme gets me, because
whatever it leaves on the table is the problem the rest of the work has to solve, and I do not want to
build machinery to fix a problem that rounding alone already handles.

The simplest scheme is uniform integer quantization by round-to-nearest. A weight matrix W is a grid
of floats; I pick a step size Δ, divide every entry by Δ, round to the nearest integer, clamp into the
representable range, and store the integers plus Δ. At inference I multiply back by Δ to recover an
approximation Ŵ ≈ W. The only real decision is the step size, and the only honest way to set it from
the data alone is to look at the magnitudes: take Δ = max(|W|) / (2^(N−1) − 1) for an N-bit symmetric
grid, so the largest-magnitude weight lands at the edge of the range and everything else falls inside.
There is no calibration here, no Hessian, no search — Δ comes straight from the tensor's own min/max.
That is the whole appeal: it costs nothing, it is exact in the sense that it never overflows the grid,
and it is trivially parallel.

The first question is *granularity*. One Δ for an entire weight matrix (per-tensor) is crude, because
different output channels of a linear layer have very different scales, and a single Δ forces the
quiet channels to round against the loud channels' range. The cheap fix that costs almost nothing is
**per-channel** scaling: one Δ per output channel (per row of the weight). That already helps a lot.
The finer-still option is **group-wise**: split each row into contiguous groups of, say, 128 weights
and give each group its own Δ (and, for an affine grid, its own zero-point). Group-128 — written g128
— stores a few extra scales but lets Δ track the local magnitude, so the rounding error per weight
falls. I should keep both settings in mind, because they are not comparable: a g128 number is always
going to look better than a per-channel number at the same nominal bit-width, and if I am not careful
I will fool myself by comparing across them.

Now the real question: how low can the bit-width go before this falls apart? At 8 bits the rounding
error is tiny — the grid has 256 levels and the relative error per weight is well under a percent;
the model is essentially unchanged. So 8-bit weight-only is basically free with RTN, and there is
nothing to discover there. The interesting regime is 4 bits and below, because that is where the
memory savings get large (4× over FP16) and where a batch-1 generation workload, which is bound by how
fast it can read weights from DRAM, actually speeds up.

Let me think about why low bits should hurt. With an N-bit grid the step Δ is fixed by the *maximum*
magnitude in the channel (or group). But the distribution of weights in a trained Transformer is
heavy in the middle and light in the tails — most weights are small, a few are large. Because Δ is set
by the largest weight, the small weights — the bulk of the matrix — are being rounded against a grid
that is far too coarse for them: a weight that is 1% of the max gets rounded to one of only a couple of
levels near zero. At 4 bits with a fine group size this is survivable, because the group's local max
keeps Δ from being dominated by some far-away outlier, and most groups are reasonably uniform. But at
**3 bits per-channel** the grid has only eight levels for an entire channel's worth of weights, the
step is enormous relative to the typical weight, and the accumulated rounding error across a deep stack
of layers compounds into a large shift in the output distribution. I expect perplexity to degrade
sharply — not gracefully — somewhere between 4 and 3 bits, and per-channel to degrade well before g128.

There is a second, worse failure I should name now even though I am not attacking it yet, because it
sets up everything that comes later. Everything above is *weight-only*: I quantize W and leave the
activations X in FP16. If I also try to quantize the activations to low bits — which is what I would
need to run the matmul itself on integer tensor cores rather than just save weight memory — RTN does
not merely degrade, it detonates. Transformer activations are not heavy-tailed-but-bounded the way
weights are; they have a few *persistent channels* whose values are on the order of 100× everything
else. A single per-tensor activation Δ set by those outlier channels makes the step so coarse that
every ordinary activation rounds to almost nothing. So I fully expect that the moment I push
activations down to 4 bits with plain RTN, perplexity goes to the thousands and the model is
effectively destroyed. That is not a bug in my arithmetic; it is the structural fact that the rest of
this ladder exists to defeat.

So here is the bet I am placing on RTN as the floor. It is the right *baseline* — free, calibration-
free, and genuinely fine at 8 bits — and it tells me precisely where the cliff is. The code is exactly
the magnitude-based step size and a clamp-and-round, applied per-channel (and optionally per-group):

```python
def find_params(self, x):                       # per-(out)channel symmetric grid
    x = x.flatten(1) if self.perchannel else x.flatten().unsqueeze(0)
    xmax = torch.maximum(x.abs().max(1)[0], torch.zeros(x.shape[0])).clamp(min=1e-5)
    self.maxq = torch.tensor(2 ** (self.bits - 1) - 1)   # symmetric N-bit grid
    self.scale = xmax / self.maxq                         # Δ from the channel's own max
    self.zero  = torch.zeros_like(self.scale)

def sym_quant_dequant(x, scale, maxq):          # round-to-nearest, clamp, dequantize
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return scale * q
```

The prediction I am setting the bar at: this should be excellent at 8 bits and acceptable at 4-bit
g128 (I would guess a Llama-2-7B WikiText perplexity within a few tenths of FP16, on the order of 5.7
against an FP16 ≈ 5.5), but it should fall off a cliff at 3-bit *per-channel*, where I expect LLaMA-7B
perplexity to climb to something like the mid-twenties against an FP16 of ≈ 5.68 — a more than 4× blow-
up that no one would deploy. If that is what the numbers say, then the question the next rung must
answer is sharp and well-posed: the rounding errors at low bit-width are not random noise I have to
live with, they are a structured loss I should be able to *compensate* — and the lever to compensate
them is sitting in the calibration activations I have so far refused to use.
