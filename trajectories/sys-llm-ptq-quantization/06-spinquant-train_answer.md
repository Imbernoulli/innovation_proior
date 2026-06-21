QuaRot cracked the W4A4 problem: rotating the residual stream by a randomized Hadamard matrix dissolves the activation outliers for free, and with GPTQ on the rotated weights Llama-2-7B lands at 6.10 perplexity at W4A4 (about 6.4 with the KV-cache also at 4 bits), against an FP16 of 5.47 — every matmul in integer arithmetic, within striking distance of full precision. But one thing about QuaRot keeps pulling at me: the rotation $Q$ is *random*. A random Hadamard spreads the outlier mass across coordinates *on average*, which gets most of the way, but it is not *chosen* for this particular model. Different random Hadamard draws give measurably different post-quantization error; the method leaves the choice of frame to chance. After I rotate by $Q$ and quantize, the quantity I actually pay for is the quantization error of the rotated activations and weights — a random rotation makes the *expected* error small, but the *realized* error for the specific $Q$ I drew is a random variable around that mean. The gap between an average random rotation and the *best* rotation is exactly the perplexity left to recover, so the question is sharp: can the rotation be *learned*?

The obstacle is that $Q$ is not a free matrix — it must stay orthogonal. The entire QuaRot construction relies on $Q^\top Q = I$: that is what makes the rotation invertible and foldable into the weights without changing the FP16 function. Parametrize $Q$ as an arbitrary matrix and run ordinary gradient descent, and the first step takes me off the orthogonal manifold; $Q$ stops being a rotation, computational invariance breaks, and I am no longer computing the same network. So I cannot optimize $Q$ in plain Euclidean space — I must optimize it *on the manifold of orthogonal matrices*, the Stiefel manifold, so every update lands on another valid rotation.

The method is **SpinQuant**: *learn* the rotation on the Stiefel manifold via **Cayley SGD**. The clean way to do constrained optimization on a smooth manifold is Riemannian gradient descent — compute the ordinary Euclidean gradient of the loss with respect to $Q$, *project* it onto the tangent space of the manifold at the current point (the allowed infinitesimal moves that keep $Q$ orthogonal), step along that tangent direction, and *retract* back onto the manifold so the new point is exactly orthogonal. For the Stiefel manifold the tangent direction's natural form is a skew-symmetric matrix $W$, and the retraction that preserves orthogonality exactly is the **Cayley transform**: for any skew-symmetric $W$,
$$\big(I - \tfrac{1}{2}W\big)^{-1}\big(I + \tfrac{1}{2}W\big)$$
is exactly orthogonal. So the recipe is: form the momentum-smoothed Euclidean gradient, build from it the skew-symmetric $W$ that is its tangent-space projection, and apply the Cayley transform to move along the manifold. The Cayley inverse is expensive, so rather than inverting I solve it with a short fixed-point iteration — a Cayley loop of a handful of cheap matmuls — which converges to the same orthogonal update. That is Cayley SGD: momentum on the gradient, skew-symmetric tangent projection, Cayley retraction, step.

The other half is what I optimize the rotation *for*. Not the FP16 output — that is invariant to $Q$ by construction and carries no gradient signal about the rotation. I optimize $Q$ for the *quantized* network's quality: build the rotated-and-quantized model end to end — rotate by the current $Q$, fake-quantize weights and activations to 4 bits, run a forward pass on a small calibration set from WikiText-2 — measure that quantized network's loss, and backpropagate it to $Q$. The gradient flows through the quantization (with a straight-through estimator on the rounding) back to the rotation matrices, and Cayley SGD steps $Q$ to reduce the quantized loss while keeping it orthogonal. I do this for the global residual-stream rotation $R_1$ and a per-block rotation $R_2$ inside the attention head dimension — the same rotation slots QuaRot used, but now *learned parameters* initialized at a random Hadamard and refined, rather than frozen at the random draw.

Two things keep this honest about cost. First, it is cheap: the only trainable parameters are the rotation matrices — one hidden-size $\times$ hidden-size $R_1$ and one small head-dim $R_2$ per layer — while the model weights stay frozen, so this is a tiny optimization (a few hundred WikiText sequences, a short run), not retraining. Second, once learned, $R_1$ and $R_2$ are *still orthogonal*, so I fold them into the weights exactly as QuaRot did and ship a model with identical inference cost — the same W4A4KV4 matmuls, the same online Hadamards for the un-fused rotations — but a *better* frame. The learned rotation is a pure quality upgrade at zero inference overhead, and the orthogonality constraint doubles as a strong regularizer, so the tiny parameter count generalizes from the small calibration set rather than overfitting it.

This is the closing move. The standing record at the top of the ladder is QuaRot at W4A4KV4 — Llama-2-7B at about 6.4 perplexity with the KV-cache also at 4 bits, a zero-shot reasoning average a couple of points under full precision. The bet is that replacing the *random* rotation with one *learned* on the Stiefel manifold finds the low-variance bottom of the rotation distribution — the frame that minimizes *this* model's 4-bit quantization loss rather than the average frame — and recovers most of the remaining gap: Llama-2-7B W4A4KV4 down to about **5.9** perplexity (FP16 5.47) with a zero-shot average around **64.0%**, narrowing the gap to full precision to under three points on the reasoning suite, the smallest W4A4KV4 gap anyone has reported. The thread this whole ladder has pulled — from compensating rounding error, to protecting salient weights, to migrating activation outliers, to rotating them away — ends here, at *learning* the rotation that rotates them away, the strongest post-training 4-bit quantization of a fixed LLM.

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
