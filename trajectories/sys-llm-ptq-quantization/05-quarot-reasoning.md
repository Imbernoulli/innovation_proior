SmoothQuant got me across the line into activation quantization: OPT-175B W8A8 back to 66.8% zero-shot,
a tenth of a point off the 66.9% FP16, by migrating the activation outliers into the weights offline.
The linear layers run as plain INT8 GEMMs. But INT8 is the comfortable case. The real prize is W4A4 —
4-bit weights *and* 4-bit activations and 4-bit KV-cache — because that is what halves the memory and
doubles the integer throughput again over INT8. So I try SmoothQuant's recipe at 4-bit activations, and
it does not hold up. On Llama-2-7B, SmoothQuant-style W4A4 lands at a WikiText perplexity around 83 — far
from the FP16 5.47, an essentially unusable model. The migration helped, but the migrated outliers are
*smaller*, not *gone*: at INT8 the residual spikes fit inside the grid; at INT4, with only sixteen
levels, even the tamed outliers blow the per-tensor activation scale, and the bulk of the signal rounds
to mush again. Offline rescaling is the wrong instrument for 4-bit activations.

Let me look harder at *why* the outliers are so destructive, because every method so far has been
fighting the outliers' magnitude — clipping them, rescaling them, splitting the difficulty — and at
4-bit that is not enough. The deeper fact is geometric. The activation outliers live in a few specific,
fixed coordinate directions of the hidden state — the same handful of feature channels, over and over.
The quantization grid is axis-aligned: it quantizes each coordinate independently. So the problem is a
collision between a signal whose energy is concentrated along a few coordinate axes and a quantizer that
treats every axis identically. What if I could *rotate the coordinate frame* so that the energy that was
piled into a few axis-aligned spikes is spread evenly across all the coordinates? A random rotation of a
vector spreads its mass: a spike in one coordinate, after multiplying by a random orthogonal matrix,
becomes a little bit of magnitude in every coordinate. The outlier would be *dissolved* into the bulk —
not clipped, not migrated, but smeared out so that no single coordinate is special and the per-tensor
grid sees a nice, near-Gaussian distribution with no spikes at all.

The objection is obvious and it is the whole reason this hasn't been the standard move: I cannot just
rotate the activations, because rotating the activations changes the function the network computes,
unless I undo the rotation somewhere. So I need rotations that are *free* — that change the internal
representation but leave the model's output exactly unchanged. This is where I get to exploit a
structural property of the Transformer: **computational invariance**. The residual stream is acted on by
linear maps (the projections in attention and MLP), and an orthogonal matrix Q can be pushed through
them. If I rotate the hidden state by Q, I can absorb Qᵀ into the weights that read from the stream and
Q into the weights that write to it, and — because RMSNorm is invariant to orthogonal rotation of its
input (it only rescales by the norm, which Q preserves) — the rotation commutes through the normalization
too, provided I first fuse the per-channel LayerNorm/RMSNorm scale into the adjacent linear so the norm
is a pure normalization. The result is a network that is *functionally identical* to the original in
FP16, but whose hidden states have been rotated into a frame where the outliers are gone.

Concretely, I fuse the norms, then pick an orthogonal Q and bake it into the weights: every matrix that
reads the residual stream gets its input rotated (W ← W Q), every matrix that writes to it gets its
output counter-rotated (W ← Qᵀ W), the embeddings get rotated and the head un-rotated. Nothing about the
output changes; everything about the *distribution* of the activations changes. The activations entering
each quantized linear are now in the rotated frame, outlier-free, and quantize cleanly to 4 bits.

Which orthogonal matrix should Q be? A random orthogonal matrix (Gaussian, then QR) already works — it
spreads the spikes. But a **Hadamard** matrix is better: its entries are all ±1/√n, so applying it is a
sequence of additions and subtractions — a Walsh–Hadamard transform — rather than a dense matmul, which
means I can do some of these rotations *online* at inference for almost no cost. And a random Hadamard
(a Hadamard with a random ±1 diagonal) keeps the outlier-spreading property while making the transform
cheap. So Q is a randomized Hadamard matrix on the hidden dimension. Some rotations can be fused into
weights once, offline (the residual-stream rotation); others — inside the attention value/output path,
and on the keys and queries for the KV-cache — must be applied *online*, but because they are Hadamard
they are just fast WHT kernels with no learned parameters.

```python
def random_hadamard_matrix(size, device):                  # ±1/√n entries, with a random sign diagonal
    Q = (torch.randint(0, 2, (size,)).double() * 2 - 1)
    return matmul_hadU(torch.diag(Q)).to(device)

@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)   # 'hadamard'
    rotate_embeddings(model, Q); rotate_head(model, Q)                       # rotate in / un-rotate out
    for layer in model_utils.get_transformer_layers(model):
        rotate_attention_inputs(layer, Q, model_type)      # W ← W Q  (reads the stream)
        rotate_attention_output(layer, Q, model_type)      # W ← Qᵀ W (writes the stream)
        rotate_mlp_input(layer, Q, model_type)
        rotate_mlp_output(layer, Q, model_type)            # + exact Hadamard on down_proj input
        rotate_ov_proj(layer, model_type, num_heads, head_dim)   # online Hadamard in the V/O path
```

With the outliers dissolved, the activations quantize to 4 bits on ordinary per-tensor (or per-token)
scales — no special outlier path, no channels kept in higher precision, every matmul genuinely in 4-bit.
And I do not throw away the weight-quantization machinery I already trust: after rotating, I still run
**GPTQ** on the (now-rotated) weights to compensate the 4-bit weight rounding error optimally. The two
ideas compose cleanly — the rotation makes the *activations* quantizable, GPTQ makes the *weights*
quantizable, and the residual KV-cache rotation makes the cache quantizable. This is the full W4A4KV4
pipeline. I will call it rotation-based quantization.

The bar and the bet. The reference is SmoothQuant pushed to W4A4 on Llama-2-7B: ≈83 perplexity, broken.
My claim is that *removing* the outliers by rotating them out of existence — rather than rescaling them —
finally makes 4-bit activations quantize as cleanly as 4-bit weights, and that with GPTQ on the rotated
weights the whole W4A4KV4 model stays within a small margin of FP16. Concretely I am betting Llama-2-7B
W4A4 comes down from ~83 to about 6.1 against an FP16 of 5.47 — and, because the rotation argument is a
property of the residual stream that gets *cleaner* as the model gets wider, that the gap shrinks with
scale: 13B landing near 5.40 (FP16 4.88) and 70B near 3.79 (FP16 3.32), within roughly half a perplexity
point of full precision at the largest size. The risk is the online Hadamard cost — I am wagering the WHT
is cheap enough that the rotations are essentially free — and the assumption that a *random* rotation is
good enough to spread every outlier. That last assumption is exactly the loose thread: a random Hadamard
spreads the mass on average, but it is not *chosen* to minimize the post-rotation quantization error for
this particular model. If rotating away the outliers is the right idea, the natural next question is
whether the rotation should be *learned* rather than left to chance.
