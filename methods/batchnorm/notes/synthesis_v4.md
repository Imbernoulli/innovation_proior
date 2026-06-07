# Synthesis (V4) — Batch Normalization

Composing-source for results_v4. Built on the V3 derivations + research-notes.md + cs231n code. This file
is written *before* the three deliverables; the deliverables are transcribed from it. Focus of V4: NEW-skill
conformance — notes-first, a genuinely *pre-method* minimal code scaffold (no method names anywhere in
context.md's code-framework), and local insight-before-method ordering.

## The questions the skill says I must be able to answer before writing

**What pain point existed?** Around 2014–2015, training deep feedforward/conv nets by SGD is fragile. Three
recurring complaints: (1) high learning rate → divergence; (2) sensitivity to initialization; (3) deep
saturating nonlinearities (sigmoid/tanh) barely train. Folk remedies (ReLU, Xavier/orthogonal init, tiny LR,
Dropout) treat each as separate.

**What tools existed and where did each fall short?** (load-bearing-ancestor write-ups, below).

**The precise first-principles object + central difficulty.** SGD minimizes Θ = argmin (1/N)Σ ℓ(x_i,Θ), each
step using a mini-batch gradient (1/m)Σ ∂ℓ/∂Θ. Write the net as a composition ℓ = F₂(F₁(u,Θ₁),Θ₂). To F₂,
x = F₁(u,Θ₁) is "the input," and the step on Θ₂ is *algebraically identical* to the step a stand-alone F₂ fed
x would take. So a textbook fact about learners — they train best when the input distribution is fixed —
applies to every internal sub-network. The central difficulty: every time Θ₁ updates, the distribution of x
moves (mean wanders, variance breathes, shape changes); upper layers spend capacity re-tracking their own
drifting inputs; the effect compounds with depth; with saturating g, drift shoves pre-activations into the
flat tails where g'→0 and gradients vanish. Name it: internal covariate shift.

**Chain of approximations theory→practice.** Whiten every layer's inputs throughout training (ideal) →
can't do it outside the gradient (b-blow-up) → must be in the model with both Jacobians → full joint whitening
is correct but O(d³), non-differentiable through the matrix root, singular for m<d → retreat to per-dimension
normalization (LeCun licence) → that shrinks representable class → restore with learnable γ,β (identity
recoverable) → can't sweep whole set each step → use mini-batch statistics (makes ∂Norm/∂X free + regularizes)
→ batch coupling breaks inference → freeze unbiased population stats, collapse to one affine → conv needs
per-feature-map joint pooling → scale-invariance explains higher LR.

**Which step makes a prior method fall out as a special case?** γ=√Var, β=E[x] recovers the *identity* — so
the un-normalized network is a representable special case; normalization changes only the default, not the
hypothesis class.

## Load-bearing ancestors (write-ups, verified against arx.tex)

- **LeCun, Bottou, Orr & Müller, "Efficient BackProp" (1998); Wiesler & Ney (2011).** Nets converge faster
  when inputs are *whitened* (zero mean, unit variance, decorrelated). Load-bearing detail: per-coordinate
  mean/variance normalization helps *even without decorrelation* — decorrelation is the expensive and least
  essential part. This is the licence for cheap per-coordinate centering-and-scaling. Gap: applied once,
  statically, to network inputs; silent about internal activations that drift during training.
- **Shimodaira (2000), covariate shift; Jiang (2008) survey.** Input distribution to a learner differs between
  train and test; fixed by domain adaptation. Conceptual lever: extend from whole system to a sub-network/layer,
  and to a shift that happens *within* training. Gap: classical version is train/test, not within-training.
- **Glorot & Bengio (2010), Xavier init.** Keep activation/gradient variances roughly constant across layers
  at the *start*. Saxe, McClelland & Ganguli (2013): orthogonal init, Jacobian singular values near 1 preserve
  gradient magnitude through depth. Gap: both pin only the *starting* distribution; it drifts as params move —
  static remedy for a dynamic problem.
- **Nair & Hinton (2010), ReLU = max(x,0).** Non-saturating positive branch = standard escape from vanishing
  gradients. Gap: treats the symptom (saturation), not the cause (drift into the saturated regime).
- **Normalization pushed inside training, done awkwardly:** mean-normalized SGD (Wiesler et al. 2014),
  reparameterizations of Raiko, Valpola & LeCun (2012), natural gradient (Povey et al. 2014), natural neural
  nets (Desjardins & Kavukcuoglu). Either modify the optimizer or normalize as a side computation interleaved
  with gradient steps. Gap = the central cautionary tale: gradient blind to how stats depend on params → the
  b-blow-up.
- **Lyu & Simoncelli (2008), divisive normalization.** Per single example / across feature maps at one location.
  Stabilizes scale but discards *absolute* scale → changes representable ability. Gap: should normalize each
  example *relative to dataset (mini-batch) statistics* so absolute scale survives.
- **Hyvärinen & Oja (2000), ICA.** An affine mixture Wu+b of many inputs tends toward a symmetric, non-sparse,
  more-Gaussian distribution than a nonlinearity's output. Argues for normalizing the *pre-activation*, not the
  layer input u (whose shape keeps drifting, so matching two moments wouldn't stabilize it).
- **Convolution property (LeCun et al. 1998a).** One filter shared across all spatial locations of a feature
  map → translation equivariance + parameter efficiency. Any inserted op must treat all locations of one map
  the same, else it breaks weight sharing.
- **Bias/variance of sample variance (Bessel).** σ̂² = (1/m)Σ(x_i−x̄)² about the *sample* mean is biased low
  by (m−1)/m (one d.o.f. spent estimating the mean); unbiased = ×m/(m−1). Matters when a small-batch statistic
  must stand in for a population quantity at inference.
- **Gülçehre & Bengio (2013), standardization layer.** Nearest neighbor in prior art: standardizes activations
  applied to the *output* of the nonlinearity, no learned scale/shift, no conv handling, no deterministic
  inference. Marks exactly the open design choices.
- Plumbing: SGD+momentum (Sutskever 2013), Adagrad, DistBelief (Dean 2012), Inception/GoogLeNet (Szegedy 2014).

## Design-decision → why (with rejected alternatives + failure modes)

| Decision | Why this | Rejected alternative & its failure |
|---|---|---|
| Stabilize each layer's *input distribution throughout training* | Sub-network equivalence: F₂'s step = stand-alone step on x; learners train best on fixed input dist | Static fixes (Xavier/orthogonal init) pin only t=0; dist drifts as params move |
| Normalization must be *inside the model* (differentiable, in the gradient) | Else gradient ignores ∂stats/∂params → b-blow-up: (u+b+Δb)−E[u+b+Δb]=u+b−E[u+b], loss flat but b→∞ | Normalize as a side step (mean-norm SGD, Raiko, natural grad) → optimizer fights normalizer |
| Carry *both* ∂Norm/∂x and ∂Norm/∂X | Dropping ∂Norm/∂X is exactly what detonates b | — |
| Per-dimension, not joint whitening | Cheap, closed-form differentiable, no covariance; LeCun: helps even undecorrelated; dissolves the m<d singularity | Full whitening: Cov^{−1/2} is O(d³) via eigendecomp, not everywhere differentiable (deriv blows up at colliding eigenvalues), re-done over whole set each step, singular when m<d |
| Learnable γ, β after normalize | Pure norm pins pre-act to mean0/var1 (e.g. sigmoid's linear regime) → shrinks representable class; γ=√Var,β=E recovers *identity* → capacity preserved | No γ,β (Gülçehre standardization layer): permanently constrains representation |
| Mini-batch statistics in training | Makes ∂Norm/∂X ordinary calculus on a sum over m (free); per-dim choice keeps it well-defined for any m; bonus regularizing noise | Whole-set sweep each step: infeasible; joint+batch: singular covariance |
| ε inside the sqrt | Near-constant feature → σ²≈0 → divide-by-zero | — |
| Three-path backprop into x_i (direct, via σ²_B, via μ_B) | x_i reaches loss by all three routes; dropping any = same error class as b-blow-up | — |
| Population stats (frozen) + unbiased Var=m/(m−1)·E_B[σ²_B] at inference | Prediction must be deterministic, batch-mate-independent; biased σ²_B underestimates pop var; collapses to one affine | Use batch stats at test: output depends on batch-mates, non-deterministic; use biased σ²_B: train/inference mismatch |
| Conv: pool per feature map over batch AND space, one (γ,β) per channel | Weight sharing → all locations produced identically → must normalize identically (equivariance); effective m′=m·p·q | Per-(channel,location) activation: different affine per position → breaks equivariance |
| Normalize pre-activation Wu+b, drop bias b | Wu+b is "more Gaussian" (ICA) so 2 moments stabilize it; mean-subtraction cancels b; β subsumes b's role | Normalize layer input u: u's shape (post-nonlinearity) keeps drifting, 2 moments insufficient |
| (Mechanism for higher LR) scale invariance BN(aWu)=BN(Wu) | ∂/∂u unchanged (backward signal scale-free) and ∂/∂(aW)=(1/a)∂/∂W (big weights→small grads, self-stabilize) → runaway cut; heuristic JJᵀ=I, singular values→1 | — |
| Batch-noise as partial Dropout replacement | Example jittered by batch-mates via μ_B,σ²_B = noise injection = regularization; Dropout noise slows convergence | — |

No holes: every knob above has a why + a rejected alternative where one exists.

## Code scaffold (PRE-METHOD) — what is genuinely known before the method exists

At context time I have a bare deep-net training harness and I do *not* yet know I'll invent a normalization
layer. The scaffold must presuppose NOTHING method-specific. Pieces that genuinely already exist:

- A generic `class Layer:` base with `forward(...)`/`backward(...)` that subclasses override — the layers I
  *already* have implement these; any *new* layer I design will be exactly one more subclass filling in these
  two methods. (This is the single empty slot the contribution will occupy — stated generically as "a layer
  whose forward/backward I have not yet written," NOT shaped around normalization.)
- Existing layers that subclass it: an `Affine` (Wu+b) layer (fully-connected pre-activation producer) and a
  `Conv` layer over (N,C,H,W) tensors with weight sharing, plus a pooling layer and a nonlinearity (ReLU).
- A `Sequential`/MLP-or-convnet container that stacks `Layer`s and runs forward then backward.
- An `SGD` (with momentum) optimizer that updates each layer's parameters from their grads.
- A training loop drawing mini-batches.
- A loss (softmax/SVM) seeding the top gradient.
- An *input-only* preprocessing/whitening utility applied once to the data at the bottom (zero-mean,
  unit-variance the dataset features) — this exists and is generic; it is NOT applied to internal activations.

Hard rules honored: no "reference implementation"/"official repo"; no BatchNorm/normalize-over-minibatch/
gamma-beta; no slot pre-shaped around normalization. The new layer in reasoning.md/answer.md is the
`forward`/`backward` of one such `Layer` subclass — it fills the generic stub.

### Scaffold ↔ final code correspondence
The final cs231n-grounded functions map onto the scaffold thus: `batchnorm_forward`/`batchnorm_backward` are
the `forward`/`backward` of a new `Layer` subclass that slots between the existing `Affine` and the
nonlinearity; `spatial_batchnorm_*` are the same for a `Layer` slotted after `Conv`; the running-stat /
mode='train'/'test' dict is the per-layer state the training loop already threads; γ,β are parameters the
existing SGD optimizer updates exactly like W,b. I wrote the final code first (grounded in cs231n) and hollowed
out the new-layer body to `# TODO` to get the scaffold.

## Evaluation settings (pre-method, settings only, no outcomes)
- MNIST (LeCun 1998a): 28×28, 10 classes; controlled probe = 3 FC hidden layers ×100 units, sigmoid, small-
  Gaussian init, 10-way + cross-entropy, ~50k steps, batch 60; compare baseline vs normalized; track a sigmoid
  input's {15,50,85} percentiles over training (direct read on drift). Metric: held-out test acc vs steps.
- ImageNet/ILSVRC 2012 (Russakovsky 2014): 1000-class, 50k val images. Host = Inception-style conv net
  (Szegedy 2014), 10M+ params, momentum SGD, batch ~32, distributed. Metrics: top-1/top-5 error; val acc@1
  vs training steps (steps-to-target is measurable). Tuning levers: init LR, LR decay, Dropout p, L2, LRN,
  shuffling, photometric distortion strength; multi-crop (144) + ensembling for final acc.

## Self-check reminders for V4
- reasoning.md: ZERO real markdown headers; continuous first-person; all derivations inline (b-blow-up,
  per-dim vs ZCA cost, γ/β identity, full 3-path backprop + simplified dx, conv joint norm, train/inference +
  unbiased var, scale-invariance/higher-LR). Local insight-before-method everywhere; revision pass.
- context.md: five headers; never name "Batch Normalization"/"BN"; no artifact; code-framework pre-method only,
  corresponds to final code; no outcomes.
- answer.md: opens with the method; no citation header; no arXiv links in code comments.
