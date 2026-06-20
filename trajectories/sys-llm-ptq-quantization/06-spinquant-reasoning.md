QuaRot cracked the W4A4 problem: rotating the residual stream by a randomized Hadamard matrix dissolves
the activation outliers for free, and with GPTQ on the rotated weights Llama-2-7B lands at 6.10
perplexity at W4A4, against an FP16 of 5.47 — a gap of about 0.6, shrinking to under half a point at
70B. That is a remarkable place to be: true 4-bit weights, activations, and KV-cache, every matmul in
integer arithmetic, within striking distance of full precision. But there is one thing about QuaRot I
keep circling back to, and it is the thread I want to pull. The rotation Q is *random*. A random
Hadamard spreads the outlier mass across coordinates *on average*, and that is enough to get most of the
way — but it is not *chosen* for this particular model. Different random Hadamard draws give
measurably different post-quantization error; the method is leaving the choice of frame to chance.

Let me make the variance concrete, because it is the whole opening. After I rotate by Q and quantize,
the quantity I actually pay for is the quantization error of the rotated activations and weights. A
random rotation makes the *expected* error small by spreading the spikes, but the *realized* error for
the specific Q I drew is a random variable around that expectation. Some rotations happen to leave a
channel slightly heavier than others; some happen to align a little better with the weight structure.
QuaRot accepts whatever it draws. If I could instead find the Q that *minimizes* the realized
quantization loss for this model, I would get the low-variance bottom of that distribution rather than
its mean — strictly better, and the gap between "average random rotation" and "best rotation" is exactly
the perplexity I have left to recover. So the question is sharp: can I *learn* the rotation?

The obstacle is that Q is not a free matrix — it must stay orthogonal. The entire QuaRot construction
relies on QᵀQ = I: that is what makes the rotation invertible and foldable into the weights without
changing the FP16 function. If I just parametrize Q as an arbitrary matrix and run gradient descent on
it to minimize quantization loss, the first step takes me off the orthogonal manifold, Q stops being a
rotation, computational invariance breaks, and I am no longer computing the same network. So I cannot
optimize Q in plain Euclidean space. I need to optimize it *on the manifold of orthogonal matrices* —
the Stiefel manifold — so that every update lands on another valid rotation.

This is a constrained optimization, and the clean way to do constrained optimization on a smooth
manifold is Riemannian gradient descent: compute the ordinary (Euclidean) gradient of my loss with
respect to Q, *project* it onto the tangent space of the manifold at the current point (the space of
allowed infinitesimal moves that keep Q orthogonal), step along that tangent direction, and then
*retract* back onto the manifold so the new point is exactly orthogonal again. For the Stiefel manifold
the tangent projection has a known form, and the retraction that preserves orthogonality exactly is the
**Cayley transform**: given a skew-symmetric matrix W (skew-symmetric is the tangent direction's
natural form), the map (I − ½W)⁻¹(I + ½W) is exactly orthogonal for any W. So the recipe is: form the
momentum-smoothed Euclidean gradient, build from it the skew-symmetric W that is its tangent-space
projection, and apply the Cayley transform to move along the manifold. The Cayley inverse is expensive,
so instead of inverting I solve it with a short fixed-point iteration — a Cayley loop — which converges
to the same orthogonal update in a handful of cheap matmuls. This is **Cayley SGD** on the Stiefel
manifold: momentum, skew-symmetric tangent projection, Cayley retraction, step.

```python
def Cayley_loop(X, W, tan_vec, t):                 # fixed-point approx of the Cayley transform
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()

# inside the Stiefel branch of the optimizer step:
V = momentum * V - g.t()                           # momentum on the Euclidean gradient
MX  = torch.mm(V, unity)
XMX = torch.mm(unity, MX)
XXMX = torch.mm(unity.t(), XMX)
W_hat = MX - 0.5 * XXMX
W = W_hat - W_hat.t()                              # skew-symmetric tangent direction
t = 0.5 * 2 / (matrix_norm_one(W) + 1e-8)
alpha = min(t, lr)
p_new = Cayley_loop(unity.t(), W, V, alpha)        # retract back onto the manifold
p.data.copy_(p_new.view(p.size()))                 # p stays exactly orthogonal
```

Now the loss. What am I optimizing the rotation *for*? Not the FP16 output — that is invariant to Q by
construction, so it carries no gradient signal about the rotation. I optimize Q for the *quantized*
network's quality: I build the rotated-and-quantized model end to end (rotate by the current Q, fake-
quantize weights and activations to 4 bits, run a forward pass on a small calibration set from
WikiText-2), measure the loss of that quantized network, and backpropagate it to Q. The gradient flows
through the quantization (with a straight-through estimator on the rounding) back to the rotation
matrices, and Cayley SGD steps Q to reduce the quantized loss while keeping it orthogonal. I do this for
the global residual-stream rotation R1 and for a per-block rotation R2 inside the attention head
dimension — the same rotation slots QuaRot used, but now their matrices are *learned parameters*
initialized at a random Hadamard and refined, rather than frozen at the random draw.

```python
R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")   # initialize at the QuaRot rotation
model.R1 = RotateModule(R1)
for i in range(model.config.num_hidden_layers):
    R2 = random_hadamard_matrix(head_dim, "cuda")
    model.model.layers[i].self_attn.R2 = RotateModule(R2)
trainable = [model.R1.weight] + [l.self_attn.R2.weight for l in model.model.layers]
optimizer = SGDG(trainable, lr=lr, stiefel=True)               # Cayley SGD on the Stiefel manifold
trainer.train()                                                # minimize the *quantized* model's loss
```

Two things make this honest about cost. First, it is cheap: the only trainable parameters are the
rotation matrices — one hidden-size×hidden-size R1 and one small head-dim R2 per layer — not the model
weights, which stay frozen. So this is a tiny optimization (a few hundred WikiText sequences, a short
run), not retraining. Second, once learned, R1 and R2 are *still orthogonal*, so I fold them into the
weights exactly as QuaRot did and ship a model with identical inference cost to QuaRot — same W4A4KV4
matmuls, same online Hadamards for the un-fused rotations — but a *better* frame. The learned rotation is
a pure quality upgrade at zero inference overhead.

This is the closing move, so let me state the bar and why I believe it clears it. The standing record at
the top of this ladder is QuaRot at W4A4KV4: Llama-2-7B at about 6.4 perplexity with the KV-cache also
at 4 bits, and a zero-shot reasoning average a couple of points under full precision. The bet is that
replacing the *random* rotation with one *learned* on the Stiefel manifold finds the low-variance bottom
of the rotation distribution — the frame that actually minimizes this model's 4-bit quantization loss
rather than the average frame — and that this recovers most of the remaining gap to FP16. Concretely I am
wagering Llama-2-7B at W4A4KV4 comes down to about **5.9** perplexity (against an FP16 of 5.47) with a
zero-shot average around **64.0%**, narrowing the gap to full precision to under three points on the
reasoning suite — the smallest W4A4KV4 gap anyone has reported. The risks are real and bounded: the
Cayley retraction is approximate (a fixed-point loop, not an exact inverse), so the iterates are only
*near*-orthogonal between retractions, and the learned rotation could in principle overfit the small
calibration set — but the parameter count is tiny (a handful of rotation matrices, not the model) and
the orthogonality constraint is a strong regularizer, so I expect it to generalize. The thread this whole
ladder has been pulling — from compensating rounding error, to protecting salient weights, to migrating
activation outliers, to rotating them away — ends here, at *learning* the rotation that rotates them
away. If a learned orthogonal frame beats a random one at W4A4KV4, this is the strongest post-training
4-bit quantization of a fixed LLM known, and that is where the ladder ends.
