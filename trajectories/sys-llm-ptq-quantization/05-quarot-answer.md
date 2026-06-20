**Problem (from step 4).** SmoothQuant's offline outlier migration works at W8A8 but breaks at W4A4:
on Llama-2-7B, SmoothQuant-style W4A4 gives ~83 perplexity (FP16 5.47), an unusable model. With only 16
levels, even the *migrated* (smaller, not gone) outliers blow the per-tensor activation scale. The
activation outliers live in a few fixed, axis-aligned coordinate directions, and the grid quantizes each
axis independently — so rescaling magnitudes is the wrong instrument for 4-bit activations.

**Key idea.** **QuaRot** (Ashkboos 2024). *Rotate the coordinate frame* so the energy piled into a few
axis-aligned spikes is spread evenly across all coordinates — a random/Hadamard orthogonal Q dissolves
each spike into a near-Gaussian, outlier-free distribution the per-tensor grid handles. Make the rotation
*free* via **computational invariance**: the residual stream is acted on by linear maps and RMSNorm is
invariant to orthogonal rotation, so (after fusing the norm scale into the adjacent linear) Q can be baked
into the weights — every matrix reading the stream gets W←WQ, every matrix writing it gets W←QᵀW,
embeddings rotated and head un-rotated — leaving the FP16 function *exactly unchanged* while the
activations move into an outlier-free frame. Use a **randomized Hadamard** Q (±1/√n entries) so some
rotations run online as cheap Walsh–Hadamard transforms (residual rotation offline; V/O path and K/Q for
the KV-cache online). Then run **GPTQ** on the rotated weights to compensate 4-bit weight rounding. The
result: true W4A4KV4 with every matmul in 4 bits, no higher-precision channels.

**Why it works.** Rotation *removes* outliers (spreads them across coordinates) rather than rescaling
them, so 4-bit activations quantize as cleanly as 4-bit weights; computational invariance makes it
zero-cost in FP16; GPTQ handles the weights and the residual rotation handles the KV-cache. The argument
is a property of the residual stream that gets cleaner with width, so the gap to FP16 shrinks with scale.

**Change / code.** The randomized-Hadamard rotation and its invariant application across the block.

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

**Target.** Bring Llama-2-7B W4A4 from SmoothQuant's ~83 down to ~6.1 (FP16 5.47), with the gap shrinking
at scale — 13B ~5.40 (FP16 4.88) and 70B ~3.79 (FP16 3.32) — making true 4-bit weights+activations+KV
viable, and raising the question of whether the rotation should be *learned* rather than random.
