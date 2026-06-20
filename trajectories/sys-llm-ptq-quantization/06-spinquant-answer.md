**Problem (from step 5).** QuaRot makes W4A4 work by rotating outliers away, but its rotation Q is
*random* — a Hadamard draw spreads the outlier mass *on average*, not optimally for this model. The
realized quantization error is a random variable around that mean, and the gap between an average
rotation and the *best* rotation is exactly the perplexity still left to recover. Can the rotation be
*learned*? The obstacle: Q must stay orthogonal (computational invariance needs QᵀQ=I), so plain
gradient descent on Q immediately leaves the manifold and breaks the construction.

**Key idea.** **SpinQuant** (Liu 2024). *Learn* the rotation on the **Stiefel manifold** of orthogonal
matrices via **Cayley SGD**: take the Euclidean gradient of the *quantized* model's loss w.r.t. Q,
project it to the tangent space (a skew-symmetric W), and retract back onto the manifold with the
**Cayley transform** (I−½W)⁻¹(I+½W) — solved by a short fixed-point loop of cheap matmuls so every update
lands on another exact rotation. Optimize Q against the loss of the *quantized* network (rotate →
fake-quantize to 4-bit → forward on a few hundred WikiText sequences → backprop with straight-through
rounding), since the FP16 output is invariant to Q and carries no signal. Learn the global residual
rotation R1 and a per-block head-dim rotation R2, initialized at the QuaRot random Hadamard and refined.
Only the rotation matrices are trainable — the model weights stay frozen — so this is a tiny
optimization, not retraining. Once learned, R1/R2 are still orthogonal, so they fold into the weights
exactly as QuaRot did: identical W4A4KV4 inference cost, a better frame.

**Why it works.** A learned orthogonal frame finds the low-variance *bottom* of the rotation
distribution — the frame that minimizes this model's 4-bit quantization loss — rather than the average
random frame, recovering most of the remaining gap to FP16. The orthogonality constraint is a strong
regularizer and the parameter count is tiny, so it generalizes from a small calibration set; folding the
learned rotations away gives the quality upgrade at zero inference overhead.

**Change / code.** Cayley SGD on the Stiefel manifold (the rotation optimizer) and the learned-rotation
training setup.

```python
import torch
from torch.optim.optimizer import Optimizer

def Cayley_loop(X, W, tan_vec, t):              # fixed-point approx of the Cayley retraction
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()

class SGDG(Optimizer):                          # Stiefel-manifold (Cayley) SGD
    def step(self):
        for group in self.param_groups:
            momentum, stiefel = group["momentum"], group["stiefel"]
            for p in group["params"]:
                if p.grad is None: continue
                unity, _ = unit(p.data.view(p.size(0), -1))
                if stiefel and unity.size(0) <= unity.size(1):
                    g = p.grad.data.view(p.size(0), -1)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros(g.t().size()).to(g)
                    V = state["momentum_buffer"]
                    V = momentum * V - g.t()                  # momentum on the Euclidean gradient
                    MX  = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()                     # skew-symmetric tangent direction
                    t = 0.5 * 2 / (matrix_norm_one(W) + 1e-8)
                    alpha = min(t, group["lr"])
                    p_new = Cayley_loop(unity.t(), W, V, alpha)   # retract onto the manifold
                    p.data.copy_(p_new.view(p.size()))            # p stays orthogonal
                    V.copy_(torch.mm(W, unity.t()))

class RotateModule(torch.nn.Module):
    def __init__(self, R_init):
        super().__init__()
        self.weight = torch.nn.Parameter(R_init.to(torch.float32).cuda())
    def forward(self, x, transpose=False):
        return x @ self.weight if transpose else self.weight @ x

# learn R1 (residual) and per-layer R2 (head-dim), init at the QuaRot random Hadamard, weights frozen:
R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")
model.R1 = RotateModule(R1)
for i in range(model.config.num_hidden_layers):
    R2 = random_hadamard_matrix(model.config.hidden_size // model.config.num_attention_heads, "cuda")
    model.model.layers[i].self_attn.R2 = RotateModule(R2)
for param in model.parameters():
    param.requires_grad = False                  # only the rotations train
trainable = [model.R1.weight] + [l.self_attn.R2.weight for l in model.model.layers]
optimizer = SGDG(trainable, lr=training_args.learning_rate, stiefel=True)   # Cayley SGD
trainer.train()                                  # minimize the *quantized* model's loss
```

**Target (finale).** Beat the standing W4A4KV4 record (QuaRot, Llama-2-7B ~6.4 ppl): learned rotations
bring Llama-2-7B to **~5.9 perplexity** (FP16 5.47) and **~64.0% zero-shot** average, the smallest
W4A4KV4 gap to full precision reported — the strongest post-training 4-bit quantization of a fixed LLM,
where the ladder ends.
