SmoothQuant got me across the line into activation quantization: OPT-175B W8A8 back to 66.8% zero-shot, a tenth of a point off the 66.9% FP16, by migrating the activation outliers into the weights offline so the linear layers run as plain INT8 GEMMs. But INT8 is the comfortable case. The real prize is W4A4KV4 — 4-bit weights, 4-bit activations, and 4-bit KV-cache — because that halves the memory and doubles integer throughput again over INT8. Apply SmoothQuant's recipe at 4-bit activations, though, and it does not hold up: on Llama-2-7B, SmoothQuant-style W4A4 lands near 83 perplexity against an FP16 of 5.47, an essentially unusable model. The migrated outliers are *smaller*, not *gone*; at INT8 the residual spikes fit inside the grid, but at INT4, with only sixteen levels, even the tamed outliers blow the per-tensor activation scale and the bulk of the signal rounds to mush again. Offline rescaling is the wrong instrument for 4-bit activations.

The deeper fact about why outliers are so destructive is geometric, and every method so far has been fighting their *magnitude* — clipping, rescaling, splitting the difficulty — which at 4 bits is not enough. The activation outliers live in a few specific, fixed coordinate directions of the hidden state, the same handful of feature channels over and over, while the quantization grid is axis-aligned and quantizes each coordinate independently. The problem is a collision between a signal whose energy is concentrated along a few coordinate axes and a quantizer that treats every axis identically. So instead of fighting the magnitude, *rotate the coordinate frame* so the energy piled into a few axis-aligned spikes is spread evenly across all coordinates. A random rotation spreads a vector's mass: a spike in one coordinate, after multiplying by a random orthogonal matrix, becomes a little magnitude in every coordinate. The outlier is *dissolved* into the bulk — not clipped, not migrated, but smeared so that no single coordinate is special and the per-tensor grid sees a near-Gaussian distribution with no spikes at all.

The objection — the whole reason this has not been the standard move — is that rotating the activations changes the function the network computes, unless the rotation is undone somewhere. The method, **QuaRot**, gets the rotations *for free* by exploiting **computational invariance** of the Transformer. The residual stream is acted on by linear maps (the projections in attention and MLP), and an orthogonal matrix $Q$ can be pushed through them: rotate the hidden state by $Q$, absorb $Q^\top$ into the weights that read from the stream and $Q$ into the weights that write to it. Because RMSNorm is invariant to orthogonal rotation of its input — it only rescales by the norm, which $Q$ preserves — the rotation commutes through the normalization too, provided I first fuse the per-channel LayerNorm/RMSNorm scale into the adjacent linear so the norm is a pure normalization. The result is a network *functionally identical* to the original in FP16 whose hidden states have been rotated into a frame where the outliers are gone. Concretely I fuse the norms, pick an orthogonal $Q$, and bake it in: every matrix that reads the residual stream gets its input rotated ($W \leftarrow WQ$), every matrix that writes to it gets its output counter-rotated ($W \leftarrow Q^\top W$), the embeddings are rotated and the head un-rotated. Nothing about the output changes; everything about the activation *distribution* does.

Which orthogonal matrix should $Q$ be? A random orthogonal matrix (Gaussian, then QR) already works — it spreads the spikes. But a **Hadamard** matrix is better: its entries are all $\pm 1/\sqrt{n}$, so applying it is a sequence of additions and subtractions — a Walsh–Hadamard transform — rather than a dense matmul, which means some of these rotations can run *online* at inference for almost no cost. A *random* Hadamard (a Hadamard with a random $\pm 1$ sign diagonal) keeps the outlier-spreading property while staying cheap. So $Q$ is a randomized Hadamard matrix on the hidden dimension. The residual-stream rotation is fused into weights once, offline; others — inside the attention value/output path, and on the keys and queries for the KV-cache — must run *online*, but because they are Hadamard they are just fast WHT kernels with no learned parameters. With the outliers dissolved, the activations quantize to 4 bits on ordinary per-tensor (or per-token) scales — no special outlier path, no channels kept in higher precision, every matmul genuinely in 4-bit.

I do not throw away the weight-quantization machinery I already trust: after rotating, I run **GPTQ** on the now-rotated weights to compensate the 4-bit weight rounding error optimally. The two ideas compose cleanly — the rotation makes the *activations* quantizable, GPTQ makes the *weights* quantizable, and the residual rotation makes the KV-cache quantizable — giving the full W4A4KV4 pipeline.

The reference is SmoothQuant pushed to W4A4 on Llama-2-7B, $\approx 83$ perplexity, broken. The bet is that *removing* the outliers by rotating them out of existence, rather than rescaling them, finally makes 4-bit activations quantize as cleanly as 4-bit weights, and that GPTQ on the rotated weights keeps the whole model within a small margin of FP16 — Llama-2-7B W4A4 coming down from $\sim 83$ to about 6.1 against FP16 5.47. Because the rotation argument is a property of the residual stream that gets *cleaner* as the model widens, the gap should shrink with scale: 13B near 5.40 (FP16 4.88) and 70B near 3.79 (FP16 3.32), within roughly half a perplexity point of full precision at the largest size. The one loose thread is that a random Hadamard spreads the mass *on average* but is not *chosen* to minimize the post-rotation quantization error for this particular model — which raises the natural next question of whether the rotation should be *learned* rather than left to chance.

```python
import torch
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear

def get_orthogonal_matrix(size, mode, device):
    if mode == 'hadamard':
        return random_hadamard_matrix(size, device)        # ±1/√n entries, random sign diagonal
    elif mode == 'random':
        m = torch.randn(size, size, dtype=torch.float64, device=device)
        q, r = torch.linalg.qr(m)
        return q * torch.sign(torch.diag(r)).unsqueeze(0)

def rotate_attention_inputs(layer, Q):                     # W ← W Q  (reads the residual stream)
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        W.weight.data = torch.matmul(W.weight.double(), Q).to(W.weight.dtype)

def rotate_attention_output(layer, Q):                     # W ← Qᵀ W (writes the residual stream)
    W = layer.self_attn.o_proj
    W.weight.data = torch.matmul(Q.T, W.weight.double()).to(W.weight.dtype)
    if W.bias is not None:
        W.bias.data = torch.matmul(Q.T, W.bias.double()).to(W.bias.dtype)

def rotate_ov_proj(layer, head_num, head_dim):             # online Hadamard in the V/O path
    apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(layer.self_attn.o_proj, had_dim=-1, output=False)

@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode, "cuda")
    rotate_embeddings(model, Q); rotate_head(model, Q)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer in model_utils.get_transformer_layers(model):
        rotate_attention_inputs(layer, Q)
        rotate_attention_output(layer, Q)
        rotate_mlp_input(layer, Q)
        rotate_mlp_output(layer, Q)                        # + exact (inverse) Hadamard on down_proj
        rotate_ov_proj(layer, model.config.num_attention_heads, head_dim)
# K/Q are Hadamard-rotated online before KV-cache quantization (QKRotationWrapper).
```
