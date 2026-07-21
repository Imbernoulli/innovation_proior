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

The ways to enforce orthogonality are not equal. A *soft penalty* λ‖QᵀQ − I‖² only keeps Q near-orthogonal,
and "near" is fatal — invariance is exact only at QᵀQ = I, so any drift injects an unabsorbable error into
every layer, and λ is an unprincipled new knob. A *reparametrization* like Q = exp(A) for skew-symmetric A
is orthogonal by construction but the matrix exponential and its gradient are expensive at 4096×4096. So I
take honest Riemannian optimization: keep Q exactly on the manifold at every step, which gives exact
invariance throughout training at a cost I can control.

The clean way to do constrained optimization on a smooth manifold is Riemannian gradient descent: compute
the ordinary Euclidean gradient of my loss with respect to Q, *project* it onto the tangent space of the
manifold at the current point (the space of allowed infinitesimal moves that keep Q orthogonal), step
along that tangent direction, and then *retract* back onto the manifold so the new point is exactly
orthogonal again. The tangent space at Q is the set QA with A skew-symmetric (differentiate QᵀQ = I: an orthogonality-
preserving infinitesimal move must be skew), and the exact retraction off it is the **Cayley transform**:
for skew-symmetric W, Y = (I − ½W)⁻¹(I + ½W) is orthogonal for *any* W. This is load-bearing, so I check
it. With Wᵀ = −W, Yᵀ = (I + ½W)ᵀ(I − ½W)⁻ᵀ = (I − ½W)(I + ½W)⁻¹, and since every factor is a polynomial in
W they commute, so YᵀY = (I − ½W)(I + ½W)⁻¹(I − ½W)⁻¹(I + ½W) = (I − ½W)(I + ½W)(I + ½W)⁻¹(I − ½W)⁻¹ = I,
exactly, no approximation. So no step size can knock me off the manifold — the exactness a soft penalty
could never give.

So the recipe is: form the momentum-smoothed Euclidean gradient, build from it the skew-symmetric W that
is its tangent-space projection, and apply the Cayley transform to move along the manifold. There is one
cost problem: the Cayley transform needs (I − ½W)⁻¹, a dense matrix inverse, O(n³) per step for the big R1
— too expensive to do every iteration. But I do not need the exact inverse; I need the *action* of the
transform, and I can get it by solving the defining fixed-point relation Y = X + t·W·½(X + Y) with a short
iteration instead of an explicit inverse. A handful of passes (five is plenty) converges to the same
orthogonal update in a few cheap matmuls — a Cayley loop. This is **Cayley SGD** on the Stiefel manifold:
momentum on the Euclidean gradient, skew-symmetric tangent projection, Cayley retraction by fixed-point,
step (the optimizer is in the answer).

The step-size clamp `t = 1/‖W‖₁` capped by the learning rate is not decoration: the fixed-point loop
converges only when the step is small relative to ‖W‖, so scaling by the inverse one-norm keeps the five
iterations inside their contraction regime — the guard that lets me skip the exact inverse. The optimizer
keys the Stiefel branch off a shape test (`unity.size(0) <= unity.size(1)`) and otherwise falls back to
plain SGD, so it can hold rotation matrices and conventional parameters side by side. The one mechanical
change from the previous rung is that each rotation slot is now a trainable `nn.Parameter` — applied (or
transposed) in the forward pass exactly where the fixed Hadamard sat — so autograd tracks it and the
rotations go from baked-in constants to leaf tensors the optimizer updates.

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
learning can only improve on it — descending from the mean of the random-rotation distribution toward its
floor, not searching from scratch.

Why these two slots? R1 is the *global* residual-stream rotation — one matrix, because the residual
stream is a single shared basis every block reads and writes, so one learned frame Gaussianizes every
attention and MLP input at once. R2 is *per-block* on the *head* dimension, because the value/output
path's outliers are organized per-head (the separate online Hadamard QuaRot needed there), a structure
local to each layer. Those are exactly the two places the previous rung placed fixed Hadamards, so
learning R1 and R2 upgrades frames whose invariance I already proved — folding them away afterward is
guaranteed to preserve the FP16 function — without inventing new intervention points or adding online
transforms, which is what preserves the earlier inference cost.

The straight-through estimator carries the entire gradient signal, so I should check it is not lying.
round(x/Δ) has zero derivative almost everywhere, so STE replaces its backward pass with the identity —
biased, but benign here for a specific reason: I am not moving the *quantized values* (which would need
the true rounding derivative), I am moving the *rotation* that decides which frame the values are
quantized in. Away from the measure-zero set where a coordinate sits on a step boundary, the quantized
loss is a smooth function of R1, R2 composed with a piecewise-constant rounding, and STE gives the correct
gradient of the smooth part — exactly the descent direction toward frames with lower quantization error.

"Learned" could sound like retraining, and it is nothing of the sort. The only trainable tensors are the
rotation matrices — one hidden-size × hidden-size R1 (4096² ≈ 17M) and a small head-dim R2 per layer
(128² × 32 ≈ 0.5M) — against 7×10⁹ frozen weights, on the order of 0.25% of the model, and the weights
never move: a tiny optimization on a few hundred WikiText sequences, not a corpus run. And once learned,
R1 and R2 are *still exactly orthogonal* (the Cayley retraction guaranteed it every step), so they fold
into the weights exactly as before and ship a model with *identical* inference cost — same W4A4KV4
matmuls, same online Hadamards — but a better-chosen frame. A pure quality upgrade at zero overhead.

Could learning R1 and R2 *overfit* the small calibration set — the obvious failure of any learned PTQ
step? The hypothesis class is extraordinarily constrained: points on the Stiefel manifold can only rotate
the frame, not scale a coordinate or drop information or fit calibration-specific artifacts, since every
reachable Q is norm-preserving and invertible. Orthogonality is a far stronger regularizer than an L2
penalty, and what it searches for — the frame that best Gaussianizes *this model's* weight and activation
distributions — is stable across text, not idiosyncratic to a few hundred sequences. And the Hadamard
initialization bounds the downside: even if learning helped nothing, I recover the previous rung.

The standing record is the previous rung at W4A4KV4: Llama-2-7B at about 6.4 perplexity with the
KV-cache also at 4 bits (above the 6.10 it reaches with a higher-precision cache), a couple of points
under full precision on zero-shot. The bet is that a rotation *learned* on the Stiefel manifold finds the
low-variance bottom of the rotation distribution rather than the average frame. I can size it a priori:
the standing gap is 6.4 − 5.47 ≈ 0.93, and learning cannot close all of it — 4-bit has an irreducible
floor even a perfectly Gaussianized signal cannot beat at 16 levels — so the recoverable part is the
slice where the random frame is worse than the best frame. That slice is a meaningful fraction (the
realized-error variance across Hadamard draws is not tiny at 4-bit, where a small residual outlier is
expensive), so I expect to recover roughly half: Llama-2-7B down to about **5.9** perplexity (FP16 5.47)
with a zero-shot average around **64.0%** — the smallest W4A4KV4 gap reported. If learning moved the
number by only a hundredth, that would falsify the premise that the random frame left real perplexity on
the table. The bounded risks are the fixed-point retraction (near-orthogonal *between* retractions, but
the step clamp keeps it converging and each completed step lands exactly on the manifold) and overfitting
(bounded by the tiny parameter count and the orthogonality regularizer). This is where the ladder ends:
if a learned orthogonal frame beats a random one at W4A4KV4, it is the strongest post-training 4-bit
quantization of a fixed LLM known.
