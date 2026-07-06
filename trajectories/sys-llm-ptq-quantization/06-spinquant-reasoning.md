QuaRot cracked the W4A4 problem: rotating the residual stream by a randomized Hadamard matrix dissolves
the activation outliers for free, and with GPTQ on the rotated weights Llama-2-7B lands at 6.10
perplexity at W4A4, against an FP16 of 5.47 — a gap of about 0.6, shrinking to under half a point at
70B. That is a remarkable place to be: true 4-bit weights, activations, and KV-cache, every matmul in
integer arithmetic, within striking distance of full precision. But there is one thing about QuaRot I
keep circling back to, and it is the thread I want to pull. The rotation Q is *random*. A random
Hadamard spreads the outlier mass across coordinates *on average* — that was the 1/√n argument, and it is
an expectation — but it is not *chosen* for this particular model. Different random Hadamard draws give
measurably different post-quantization error; the method is leaving the choice of frame to chance, and
accepting whatever the first draw happens to give.

Let me make the variance concrete, because it is the whole opening. After I rotate by Q and quantize, the
quantity I actually pay for is the quantization error of the rotated activations and weights. The 1/√n
smearing says the *expected* per-coordinate outlier contribution is M/√n, but that is a mean over random
Q; the *realized* value for the specific Q I drew is a random variable scattered around it. Some rotations
happen to leave one coordinate a little heavier than M/√n — a residual mini-outlier that costs
disproportionately at 4-bit, where every level is precious — and some happen to align the rotated weight
distribution slightly better or worse with the axis-aligned grid. QuaRot accepts whatever it draws, which
means it lands at the *mean* of the realized-error distribution. But I do not want the mean; I want the
bottom. If I could find the Q that *minimizes* the realized quantization loss for this model, I would get
the low-variance floor of that distribution rather than its average, and the gap between "average random
rotation" and "best rotation" is exactly the perplexity I have left to recover — the ~0.6 to FP16 at 7B is
not all irreducible; some fraction of it is just the cost of not having chosen Q. So the question is
sharp: can I *learn* the rotation?

The obstacle is that Q is not a free matrix — it must stay orthogonal. The entire computational-invariance
construction relies on QᵀQ = I: that is what makes the rotation invertible and foldable into the weights
without changing the FP16 function. If I just parametrize Q as an arbitrary matrix and run gradient
descent on it to minimize quantization loss, the very first step takes me off the orthogonal set — Q + ηG
is not orthogonal for a generic gradient G — so Q stops being a rotation, the reader/writer foldings no
longer compose to the identity, computational invariance breaks, and I am no longer computing the same
network in FP16. I cannot optimize Q in plain Euclidean space. I have to optimize it *on the manifold of
orthogonal matrices* — the Stiefel manifold — so that every update lands on another valid rotation.

Let me weigh the ways to enforce orthogonality, because the choice determines whether this is practical.
One option is a *soft penalty*: add λ‖QᵀQ − I‖² to the loss and hope Q stays near-orthogonal. But
"near" is fatal here — the invariance is exact only at QᵀQ = I, and a Q that is off by even a little
injects an uncontrolled error into every layer it touches, which at 4-bit is not absorbable; and I would
have a new hyperparameter λ trading orthogonality against loss with no principled setting. A second
option is to *reparametrize* through a map that is orthogonal by construction — e.g. Q = exp(A) for a
skew-symmetric A, optimizing A freely in Euclidean space. That is clean but the matrix exponential (and
its gradient) is expensive for a 4096×4096 rotation. The third option is honest Riemannian optimization:
keep Q exactly on the manifold at every step by construction. That is the one that gives exact invariance
throughout training at a cost I can control, so that is the path.

The clean way to do constrained optimization on a smooth manifold is Riemannian gradient descent: compute
the ordinary Euclidean gradient of my loss with respect to Q, *project* it onto the tangent space of the
manifold at the current point (the space of allowed infinitesimal moves that keep Q orthogonal), step
along that tangent direction, and then *retract* back onto the manifold so the new point is exactly
orthogonal again. For the orthogonal group the tangent space at Q is the set of QA with A skew-symmetric,
and the retraction that preserves orthogonality exactly is the **Cayley transform**: given a
skew-symmetric matrix W, the map (I − ½W)⁻¹(I + ½W) is orthogonal for *any* W. Let me verify that, because
the whole method rests on it. Write Y = (I − ½W)⁻¹(I + ½W) with Wᵀ = −W. Then Yᵀ = (I + ½W)ᵀ(I − ½W)⁻ᵀ =
(I − ½W)(I + ½W)⁻¹, using (I + ½W)ᵀ = I − ½W and ((I − ½W)ᵀ)⁻¹ = (I + ½W)⁻¹. Now YᵀY = (I − ½W)(I + ½W)⁻¹
(I − ½W)⁻¹(I + ½W). Every factor is a polynomial in W, so they all commute, and I can reorder to
(I − ½W)(I + ½W)·(I + ½W)⁻¹(I − ½W)⁻¹ = I. So YᵀY = I exactly — the Cayley transform of any skew-symmetric
W is exactly orthogonal, no approximation. That is precisely the retraction I need. It is worth noting
*why* skew-symmetry is the right tangent form and not an arbitrary choice: the orthogonal group's tangent
space at the identity is exactly the skew-symmetric matrices (differentiate QᵀQ = I to get Q̇ᵀQ + QᵀQ̇ = 0,
i.e. QᵀQ̇ is skew), so an infinitesimal move that keeps Q orthogonal must be QA with A skew. The Cayley
transform then maps that whole tangent space onto the group, and the verification above shows the map is
onto the orthogonal matrices for *any* skew W, so no step size can ever knock me off the manifold — the
retraction is exact by construction, which is the property a soft penalty could never give me.

So the recipe is: form the momentum-smoothed Euclidean gradient, build from it the skew-symmetric W that
is its tangent-space projection, and apply the Cayley transform to move along the manifold. There is one
cost problem: the Cayley transform needs (I − ½W)⁻¹, a dense matrix inverse, O(n³) per step for the big R1
— too expensive to do every iteration. But I do not need the exact inverse; I need the *action* of the
transform, and I can get it by solving the defining fixed-point relation Y = X + t·W·½(X + Y) with a short
iteration instead of an explicit inverse. A handful of passes (five is plenty) converges to the same
orthogonal update in a few cheap matmuls — a Cayley loop. This is **Cayley SGD** on the Stiefel manifold:
momentum on the Euclidean gradient, skew-symmetric tangent projection, Cayley retraction by fixed-point,
step.

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

The step-size clamp `t = 1/‖W‖₁` capped by the learning rate is not decoration: the fixed-point Cayley
loop converges only when the step is small enough relative to the magnitude of W, so scaling the step by
the inverse one-norm of W keeps the five-iteration loop inside its contraction regime. It is the guard
that lets me get away with five passes instead of an exact inverse. Wrapping the step in an optimizer,
the whole Stiefel branch keys off a simple shape test — it only applies the manifold machinery when the
parameter is a matrix whose rows are the orthonormal directions (`unity.size(0) <= unity.size(1)`), and
otherwise falls back to ordinary SGD — so the same optimizer can hold the rotation matrices and any
conventional parameters side by side:

```python
class SGDG(Optimizer):                          # Stiefel-manifold (Cayley) SGD
    def step(self):
        for group in self.param_groups:
            momentum, stiefel = group["momentum"], group["stiefel"]
            for p in group["params"]:
                if p.grad is None: continue
                unity, _ = unit(p.data.view(p.size(0), -1))
                if stiefel and unity.size(0) <= unity.size(1):
                    g = p.grad.data.view(p.size(0), -1)
                    V = self.state[p].setdefault("momentum_buffer", torch.zeros(g.t().size()).to(g))
                    V = momentum * V - g.t()                  # momentum on the Euclidean gradient
                    W_hat = torch.mm(V, unity) - 0.5 * torch.mm(unity.t(), torch.mm(unity, torch.mm(V, unity)))
                    W = W_hat - W_hat.t()                     # skew-symmetric tangent direction
                    alpha = min(0.5 * 2 / (matrix_norm_one(W) + 1e-8), group["lr"])
                    p.data.copy_(Cayley_loop(unity.t(), W, V, alpha).view(p.size()))   # retract, stay orthogonal
                    V.copy_(torch.mm(W, unity.t()))

class RotateModule(torch.nn.Module):            # holds a learnable rotation as a Parameter
    def __init__(self, R_init):
        super().__init__()
        self.weight = torch.nn.Parameter(R_init.to(torch.float32).cuda())
    def forward(self, x, transpose=False):
        return x @ self.weight if transpose else self.weight @ x
```

The `RotateModule` is what turns a QuaRot rotation slot into a trainable one: it stores R as an ordinary
`nn.Parameter` so autograd tracks it, and applies it (or its transpose) in the forward pass exactly where
QuaRot applied its fixed Hadamard. Registering R1 and R2 this way is the whole mechanical difference from
the previous rung — the rotations went from constants baked into weights to leaf tensors the optimizer
updates.

Now the loss — what am I optimizing the rotation *for*? This is the subtle part, and getting it wrong
would make the whole thing carry no signal. I cannot optimize Q against the FP16 output, because by
construction the FP16 output is *invariant* to Q — that was the entire point of computational invariance —
so ∂(FP16 loss)/∂Q ≡ 0 and there is nothing to learn. The gradient signal about the rotation lives
entirely in the *quantized* network: I build the rotated-and-quantized model end to end (rotate by the
current Q, fake-quantize weights and activations to 4 bits, run a forward pass on a small calibration set
from WikiText-2), measure the loss of *that* quantized network, and backpropagate it to Q. The difference
between the FP16 output and the quantized output is a function of Q — different frames give different
quantization error — so the quantized loss is exactly the quantity whose Q-dependence I want to minimize.
The gradient has to flow through the rounding operation, which has zero derivative almost everywhere, so I
use a straight-through estimator (treat round as identity in the backward pass) to pass gradient through
the quantizer to the rotation matrices. Then Cayley SGD steps Q to reduce the quantized loss while
keeping it orthogonal. I do this for the global residual-stream rotation R1 and for a per-block rotation
R2 inside the attention head dimension — the same rotation slots QuaRot used, but now their matrices are
*learned parameters* initialized at a random Hadamard and refined, rather than frozen at the random draw.
Initializing at the Hadamard matters: it starts the optimization at QuaRot's already-good frame, so
learning can only improve on it — I am descending from the mean of the random-rotation distribution
toward its floor, not searching from scratch.

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

Why these two rotation slots and not more or fewer? R1 is the *global* residual-stream rotation — one
matrix shared across the whole network, because the residual stream is a single shared basis that every
block reads and writes, so a single learned frame there Gaussianizes the activations feeding every
attention and MLP input at once. R2 is *per-block* and lives on the *head* dimension, because the
value/output path's outliers are organized per-head (that was the separate online Hadamard QuaRot needed
there), a structure that is local to each layer and not shared, so it earns its own learned rotation per
layer. Those are exactly the two places QuaRot placed fixed Hadamards, so learning R1 and R2 upgrades the
frames that already mattered without inventing new intervention points — every learned rotation sits in a
slot whose invariance I already proved at the previous rung, so folding them away afterward is guaranteed
to preserve the FP16 function. Adding rotations at more slots would mean more parameters and more online
transforms at inference; keeping exactly these two is what preserves QuaRot's inference cost while
capturing the dominant frames.

I should double-check that the straight-through estimator is not silently lying to the optimizer, because
the entire gradient signal passes through it. The rounding step round(x/Δ) has derivative zero almost
everywhere and undefined at the step boundaries, so a true gradient would be useless; STE replaces its
backward pass with the identity, i.e. it pretends d(round(u))/du = 1. This is a biased estimator, but the
bias is benign here for a specific reason: I am not using the gradient to move the *quantized values*
(which would need the true rounding derivative), I am using it to move the *rotation* that decides which
frame the values are quantized in. The quantized loss as a function of R1, R2 is, away from the
measure-zero set where a coordinate sits exactly on a step boundary, a smooth function of the rotation
composed with a piecewise-constant rounding, and STE gives the correct gradient of the smooth part while
ignoring the delta-function contributions at the boundaries — which is exactly the descent direction I
want for the frame. So the STE gradient points the rotation toward frames with lower quantization error,
which is the thing I am optimizing, and its bias does not corrupt that direction.

Two things make this honest about cost, and I want to count them because "learned" could sound like
retraining and it is nothing of the sort. First, it is cheap in parameters: the only trainable tensors
are the rotation matrices — one hidden-size × hidden-size R1 (4096² ≈ 17M entries) and one small head-dim
R2 per layer (128² ≈ 16k entries × 32 layers ≈ 0.5M) — against the model's 7×10⁹ frozen weights, so I am
training on the order of 0.25% as many parameters as the model has, and the model weights never move.
This is a tiny optimization — a few hundred WikiText sequences, a short run — not a training run over a
corpus. Second, once learned, R1 and R2 are *still exactly orthogonal* (the Cayley retraction guaranteed
it every step), so I fold them into the weights exactly as QuaRot did and ship a model with *identical*
inference cost to QuaRot — same W4A4KV4 matmuls, same online Hadamards for the un-fused rotations — but a
*better* frame. The learned rotation is a pure quality upgrade at zero inference overhead: nothing about
the served model is heavier, only better-chosen.

Let me also reason about whether learning R1 and R2 could *overfit* the small calibration set, since that
is the obvious failure mode of any learned PTQ step. The hypothesis class is extraordinarily constrained:
R1 and R2 are not free matrices but points on the Stiefel manifold, so the optimization can only rotate
the frame — it cannot, for instance, scale a coordinate, drop information, or fit calibration-specific
artifacts, because every reachable Q is norm-preserving and invertible. The orthogonality constraint is
itself a powerful regularizer, far stronger than an L2 penalty, and it means the learned rotation is
searching a compact, low-dimensional set for the frame that best Gaussianizes *this model's* weight and
activation distributions — properties that are stable across text, not idiosyncratic to a few hundred
sequences. So I expect it to generalize from the small calibration set, and the initialization at the
Hadamard bounds the downside: even if learning helped nothing, I would recover QuaRot.

This is the closing move, so let me state the bar and why I believe it clears it. The standing record at
the top of this ladder is QuaRot at W4A4KV4: Llama-2-7B at about 6.4 perplexity with the KV-cache also at
4 bits (a touch above the 6.10 it reaches when the cache is left at higher precision), and a zero-shot
reasoning average a couple of points under full precision. The bet is that replacing the *random*
rotation with one *learned* on the Stiefel manifold finds the low-variance bottom of the rotation
distribution — the frame that actually minimizes this model's 4-bit quantization loss rather than the
average frame — and that this recovers most of the remaining gap to FP16. Let me size the wager so it is falsifiable and not just hopeful. At W4A4KV4 the standing gap to FP16 is
about 6.4 − 5.47 ≈ 0.93 perplexity. Learning the rotation cannot close *all* of it — 4-bit quantization
has an irreducible error floor no frame can beat, since even a perfectly Gaussianized signal loses
resolution at 16 levels — so the recoverable part is only the slice attributable to the *random* frame
being worse than the *best* frame. My argument is that this slice is a meaningful fraction of the gap
(the realized-error variance across Hadamard draws is not tiny at 4-bit, where a small residual outlier is
expensive), so I expect to recover roughly half of the 0.93: from ~6.4 down to about 5.9, leaving a ~0.43
residual to FP16 that is mostly the irreducible 4-bit floor. If learning the rotation instead moved the
number by only a hundredth, that would falsify the premise that the random frame was leaving real
perplexity on the table. Concretely I am wagering Llama-2-7B at W4A4KV4 comes down to about **5.9**
perplexity (against an FP16 of 5.47) with a zero-shot average around **64.0%**, narrowing the gap to full
precision to under three points on the reasoning suite — the smallest W4A4KV4 gap anyone has reported. The risks are real and bounded: the Cayley
retraction is solved by a fixed-point loop rather than an exact inverse, so the iterates are only
*near*-orthogonal *between* retractions (though the step-size clamp keeps the loop converging and each
completed retraction lands exactly on the manifold), and the learned rotation could in principle overfit
the small calibration set — but the parameter count is tiny and the orthogonality constraint is a strong
regularizer, so I expect it to generalize. The thread this whole ladder has been pulling — from
compensating rounding error, to protecting salient weights, to migrating activation outliers, to rotating
them away — ends here, at *learning* the rotation that rotates them away. If a learned orthogonal frame
beats a random one at W4A4KV4, this is the strongest post-training 4-bit quantization of a fixed LLM
known, and that is where the ladder ends.
