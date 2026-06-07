# SimCLR — synthesis for V4 (notes-first; compose results FROM this file)

This is the Phase-1.5 synthesis the new skill requires *before* writing any deliverable.
It (a) states the pain point and the precise object being optimized, (b) writes up every
load-bearing ancestor with its specific gap, (c) carries the full design-decision → why
table with rejected alternatives, and (d) records the V4-specific framing rules. The three
results files are transcribed from this, not from memory.

---

## 0. V4 framing rules (what changed since V3, must obey)

- **In-frame**: never name SimCLR or treat it as a published artifact in `context.md`; no
  "Paper:" line; no authors/venue/arXiv anywhere; no arXiv links in code comments. Prior-art
  author/year citations are encouraged. `answer.md` may name the method "SimCLR".
- **context.md** = five sections (research question / background / baselines / evaluation
  settings / code framework), structured prose OK.
- **CODE-FRAMEWORK = MINIMAL PRE-METHOD SCAFFOLD.** This is the big V4 change. At context
  time we presuppose **nothing** about the method. We do NOT know we'll use a projection
  head, NT-Xent, temperature, ℓ2-normalization, two views, color/crop/blur, or global BN.
  The scaffold is a bare self-supervised representation-learning harness:
    - an existing `encoder` (standard ResNet — known prior art),
    - `def augment(x): # TODO: the augmentation we'll design` (generic, no recipe),
    - `class RepresentationObjective: # TODO: the training signal we'll design` (generic, no
      NT-Xent/temperature/negatives shape),
    - an optimizer (generic SGD/momentum + LR schedule machinery that already exists),
    - a linear-probe evaluation stub (the yardstick, which predates the method).
  NO "reference implementation" / "official repo" wording anywhere. NO method names.
  The final code in reasoning.md/answer.md FILLS IN these exact stubs (augment → the 3-aug
  pipeline; RepresentationObjective → NT-Xent + projection head + large-batch in-batch
  negatives; optimizer → LARS; eval stub stays). Write final code first, then hollow out.
- **reasoning.md** = ONE continuous first-person present-tense monologue, ZERO real markdown
  headers, ALL derivations inline, dead ends + aha, insight-before-method everywhere, a 2.4
  revision pass. V3 was Chinese and ~13k chars; V4 in English, genuinely rich, ~28k+ chars.
- **answer.md** opens with the method; faithful code; no citation header.

The required inline derivations in reasoning.md (hard completeness bar):
  1. InfoNCE optimum is the density ratio p(x|c)/p(x) — Bayes posterior derivation.
  2. MI bound I ≥ log N − L_N — the full E[r_j]=1, drop-the-+1, N−1≈N chain.
  3. NT-Xent gradient → automatic hard-negative weighting (each negative weighted by its
     own softmax prob); contrast with triplet/logistic gradients that don't.
  4. Projection-head information-discarding argument (loss makes z invariant → z throws away
     color/orientation → keep h before g; nonlinear > linear > none).
  5. Color-histogram shortcut for crop-only → composition with color distortion.
  Plus: τ as sharpness knob, ℓ2-norm so magnitude can't be gamed, global-BN leak, LARS.

---

## 1. Pain point and the object being optimized

Pain: human labels are expensive; most images are unlabeled. We want representations as good
as supervised ones, "for free" from pixels. **Yardstick = linear evaluation**: freeze the
encoder, train one linear classifier on top, read its test accuracy as a proxy for how
linearly separable (= semantically organized) the representation is. A solution must (a) push
the linear-probe number toward a supervised net of the same architecture, (b) keep scaling,
(c) not rely on a bespoke architecture or a fragile auxiliary mechanism — every such crutch
couples the learning signal to engineering and limits generality.

Two clean sub-questions a method must answer on its own terms:
  - **where does the predictive task come from** (what is the net asked to predict, built
    without labels)?
  - **where do the contrastive negatives come from** (how to supply enough informative
    "this is not the same thing" signal)?

The central object once we commit to contrastive learning: a categorical cross-entropy that
identifies, among a candidate set with one positive and many negatives, which is the positive.
Its optimum and its MI interpretation are the two load-bearing derivations (§3).

---

## 2. Load-bearing ancestors (verified vs primary sources) — write-ups for context/reasoning

### Becker & Hinton 1992 — agreement under transformation (the seed)
Make representations of two views of the same input AGREE; maximize MI between adjacent
patches' outputs. Lineage seed of "agreement under transformation". Gap: small scale, no
deep nets / modern augmentation / softmax-over-many-negatives contrastive machinery.

### Hadsell, Chopra, LeCun 2006 — DrLIM / contrastive loss
Make "agree" trainable: pull positive pairs together, push negatives apart past a margin
(spring model). Establishes positive-vs-negative as the basic contrastive shape. Gap:
pairwise margin loss; no temperature, no softmax over many negatives, no in-batch negatives
at scale.

### Dosovitskiy et al. 2014 — Exemplar CNN (parametric instance discrimination)
Treat each image (and its augmentations) as its own class; classify which exemplar. PARAMETRIC:
one weight vector w_j per instance in the softmax → classifier head grows with the dataset →
intractable at millions; per-instance weights don't generalize to unseen instances. Hits a
scaling wall.

### Wu et al. 2018 — InstDisc (non-parametric instance discrimination) [PDF in refs]
Fix: replace class weight w_j with the ℓ2-normalized FEATURE v_j itself →
  P(i|v) = exp(vᵢᵀv/τ) / Σ_j exp(vⱼᵀv/τ),  ‖v‖=1.
Introduces temperature τ and ℓ2-normalization explicitly — the exact ingredients carried
forward. Full denominator over all instances is prohibitive → NCE approximation + a MEMORY
BANK storing every instance's most recent feature, updated each step; Z by Monte Carlo. Gap:
bank features are STALE (written by an earlier encoder than the one producing the query) and
the bank is extra machinery with its own knobs.

### Oord et al. 2018 — CPC / InfoNCE [PDF in refs] — the loss ancestor
Autoregressive context c predicts a future latent; among set X with 1 positive from
p(x|c) and N−1 negatives from proposal p(x):
  L_N = −E[ log f(x_pos,c) / Σ_{x_j∈X} f(x_j,c) ].
Categorical cross-entropy of "identify the positive". Optimal f ∝ density ratio
p(x|c)/p(x); minimizing L_N maximizes the MI lower bound I ≥ log N − L_N (tighter as N
grows). NT-Xent IS this, specialized: positive = other augmentation of same image,
similarity = cosine/τ, negatives = rest of batch, symmetric over both views. Gap: CPC ties
the loss to a pipeline — deterministically patch the image into a grid, run a PixelCNN
context-aggregation net, encoder only sees small patches. Architecture-heavy.

### Sohn 2016 — N-pair loss
Generalizes triplet from 1 to N−1 negatives via softmax / log-sum-exp; points at reusing
the OTHER in-batch examples as negatives (also chen2017sampling; Ye 2019; Ji 2019). Gap:
framed for deep metric learning / retrieval, not a full SSL framework; often paired with
mining.

### Schroff et al. 2015 — FaceNet / triplet + semi-hard mining
Margin triplet max(0, d(a,p) − d(a,n) + m). Gradient = v⁺ − v⁻ if margin violated, else 0 —
treats all margin-violating negatives alike, no weighting by hardness → REQUIRES explicit
semi-hard mining (pick negatives within the margin, farther than the positive). The baseline
that motivates choosing the softmax form (which weights hard negatives automatically).

### He et al. 2019 — MoCo (momentum contrast) [PDF in refs]
Decouple #negatives from batch size with a QUEUE of keys from previous minibatches (enqueue
current, dequeue oldest). Can't backprop the whole queue and an evolving encoder makes queued
keys inconsistent → momentum key encoder θ_k ← m·θ_k+(1−m)·θ_q, m≈0.999, keeps queued keys
mutually consistent. Gap: still a separate mechanism (queue + momentum encoder), keys not
produced by the CURRENT query encoder; consistency is approximate.

### Hjelm 2018 (DIM) / Bachman 2019 (AMDIM)
Maximize MI global-to-local / across views by CONSTRAINING the receptive field in the
architecture (many 1×1 convs), tanh-clipped + regularized critic, learned aug policy. Task
lives in the network design → can't use a standard powerful backbone. (AMDIM also hints at a
nonlinear transform before the loss.)

### Supporting machinery
ResNet (he2016) encoder; BatchNorm (ioffe2015) → its per-device statistics under data
parallelism create a leak we must fix (global BN). LARS (you2017) for large-batch stability;
linear warmup + cosine decay (goyal2017; loshchilov2016). Augmentation toolbox: Inception
random resized crop (szegedy2015), color jitter/dropping (howard2013; szegedy2015), cutout
(devries2017), Gaussian blur, Sobel, AutoAugment (cubuk2019). Word2Vec (mikolov2013)
logistic/NCE = the "NT-Logistic" comparison point.

---

## 3. The two load-bearing derivations (worked, for reasoning.md)

### 3a. InfoNCE optimum = density ratio (Bayes)
Set X = {x_1..x_N}, exactly one positive at unknown index, drawn from p(x|c); the rest from
p(x). Event "index i is the positive" has likelihood ∝ p(x_i|c) Π_{j≠i} p(x_j). Posterior:
  P(pos=i|X,c) = [p(x_i|c) Π_{j≠i}p(x_j)] / Σ_k [p(x_k|c) Π_{j≠k}p(x_j)].
Divide numerator and denominator by Π_all p(x): Π_{j≠i}p(x_j) = (Π_all p)/p(x_i), so each
term → p(x_i|c)/p(x_i), Π_all p cancels:
  P(pos=i|X,c) = [p(x_i|c)/p(x_i)] / Σ_k [p(x_k|c)/p(x_k)].
Match to the loss's softmax shape f(x_i,c)/Σ_k f(x_k,c): optimal f(x,c) ∝ p(x|c)/p(x). ⇒ the
correct loss shape is the log-sum-exp softmax pushing similarity toward the density ratio —
NOT a margin. (First hard reason to pick the softmax form.)

### 3b. MI bound I ≥ log N − L_N
Plug optimal f into L_N. Let r = p(x|c)/p(x) for each candidate:
  L_N = −E log[ r_pos / (r_pos + Σ_neg r_j) ] = E log[ 1 + (1/r_pos) Σ_neg r_j ].
Each negative x_j ~ p: E[r_j] = ∫ p(x_j) · p(x_j|c)/p(x_j) dx_j = ∫ p(x_j|c) dx_j = 1. N−1
negatives ⇒ Σ_neg r_j ≈ N−1. So
  L_N ≈ E log[1 + (N−1)/r_pos] ≥ E log[(N−1)/r_pos] ≈ E log[N/r_pos]
      = log N − E log r_pos = log N − I(x;c).
⇒ I(x;c) ≥ log N − L_N. The bound's ceiling is log N: more negatives → looser-but-higher
ceiling → can track the true MI. "More negatives is better" is the inequality, not a
preference. (Note tschannen2019: success may be the loss form rather than MI per se — either
way, conclusion is softmax form + many negatives.)

### 3c. NT-Xent gradient → automatic hard-negative weighting (from Table tab:loss)
For ℓ2-normalized anchor u, positive v⁺, negatives {v⁻}, Z(u)=Σ_v exp(uᵀv/τ):
  ℓ = −uᵀv⁺/τ + log Σ_v exp(uᵀv/τ).
  ∂ℓ/∂u = (1/τ)[ Σ_v (exp(uᵀv/τ)/Z) v − v⁺ ]
        = −(1/τ)(1 − p⁺) v⁺ + (1/τ) Σ_{v⁻} p⁻ v⁻,  p = softmax prob.
Each negative is pushed away with weight p⁻ = exp(uᵀv⁻/τ)/Z(u): the more similar (larger
uᵀv⁻), the larger p⁻, the harder it's pushed. AUTOMATIC hard-negative mining, inside the
softmax denominator, zero mining code.
  - Margin triplet gradient: v⁺ − v⁻ if violated, else 0 — all violating negatives equal, no
    hardness weighting ⇒ needs semi-hard mining.
  - NT-Logistic gradient: σ(−uᵀv⁺/τ)/τ v⁺ − σ(uᵀv⁻/τ)/τ v⁻ — per-pair sigmoid, no relative
    hardness weighting either.
τ = sharpness knob: small τ → sharp p⁻, gradient concentrated on hardest negatives (extreme
mining) but too small over-penalizes / unstable / gradient hijacked by 1–2 hardest;
large τ → uniform p⁻, hard-negative signal washed out, weak. Intermediate τ (≈0.1 ImageNet,
≈0.5 CIFAR). ℓ2-norm: without it uᵀv is unbounded ⇒ model games the loss by GROWING vector
magnitude rather than aligning direction (contrastive accuracy can rise while representation
worsens); normalizing to the unit sphere makes uᵀv∈[−1,1], gives τ a clean meaning, and puts
the hardness weighting on direction-similarity (the thing we care about).

---

## 4. DESIGN-DECISION → WHY table (with rejected alternatives + failure modes)

| Decision | Why this | Rejected alternative → failure mode |
|---|---|---|
| Contrastive / agreement-under-augmentation pretext | Doesn't bake a fixed invariance in by fiat; the invariance set is chosen by the augmentation family. Already the leading line. | Hand-crafted pretext (rotation/jigsaw/colorize): each forces a specific cue (rotation-prediction makes the net rotation-SENSITIVE — opposite of a general rep); heuristic → limited generality. Generative (VAE/GAN): models full pixel distribution — heavy, overkill for a representation. |
| Predictive task FROM data augmentation (random crop subsumes global→local + adjacent-view) | Decouples task from architecture ⇒ can use any standard powerful ResNet unchanged. Two crops of one image = one big/one small (global→local) or two adjacent (adjacent-view) automatically. | CPC: deterministic patch grid + PixelCNN context net + encoder sees only small patches → architecture-heavy. DIM/AMDIM: constrain receptive field (1×1 convs) → can't use a standard backbone. |
| COMPOSITION of augmentations; crop + color distortion the essential pair | Crop-only lets the net match the two views by their shared COLOR HISTOGRAM — a shortcut solving the contrastive task without learning semantics; color distortion destroys it; composition makes the task hard & meaningful. Corollary: contrastive learning needs STRONGER augmentation than supervised (no labels to fall back on; aug defines the task & blocks shortcuts). | Single augmentation: task near-perfectly solvable yet representation poor. AutoAugment (supervised-tuned): optimizes a different objective; not necessarily best for contrastive. |
| Add Gaussian blur | Blocks a high-frequency shortcut; small extra gain. | Omit → slightly worse. |
| Negatives FROM a large batch (in-batch negatives, no bank/queue) | Batch of thousands ⇒ 2(N−1) negatives, all FRESH (current encoder), all back-proppable; no staleness, no NCE denominator approximation; denominator = the batch. The N-pair / in-batch-negative idea pushed to ImageNet scale as a full framework. | Memory bank (InstDisc): stale features, extra knobs. Queue + momentum encoder (MoCo): keys from old encoder; momentum only makes staleness mild, not gone; extra mechanism. Trade-off: bank/queue buy quantity at the cost of freshness; large batch gives both, moving the cost to engineering — accepted. |
| NT-Xent (softmax) loss | (3a) optimal critic ∝ density ratio ⇒ softmax shape, not margin; (3c) gradient auto-weights hard negatives by softmax prob ⇒ no mining. | Margin triplet / NT-Logistic: gradients don't weight negatives by hardness ⇒ require semi-hard mining to train at all. |
| Temperature τ (intermediate) | Sharpness knob for the hard-negative weighting; intermediate balances concentration vs signal strength. | Too small → over-penalize / unstable / hijacked by hardest few. Too large → uniform → weak hard-negative signal. |
| ℓ2-normalization (cosine similarity) | Bounds uᵀv∈[−1,1] so the net can't game the loss via magnitude; gives τ a clean scaling; puts hardness weighting on direction. | No normalization → unbounded dot products, model grows magnitude, contrastive acc up but representation worse; τ loses meaning. |
| Symmetric loss over both directions (i,j) and (j,i) | Otherwise the gradient flows from only one view direction. | One-directional → half the signal. |
| Nonlinear projection head g; train on z=g(h), KEEP h (before g) downstream | The loss trains z to be INVARIANT to augmentation → z throws away exactly color/orientation/position/high-freq, which are downstream-useful; g is a buffer that absorbs the invariance so h upstream keeps the info. Counter-intuitive: use the layer BEFORE the optimized one. Probe: an MLP recovers the applied transform far better from h than from z. | No head (loss on h): forces h itself to be invariant → discards downstream-useful info. Use z downstream: z is the most-compressed/invariant layer → worst. Linear head: limited capacity to absorb invariance nonlinearly; nonlinear > linear > none. |
| Low-dim z (128) | z is a temporary "loss interface", discarded after training → no need to be wide. | Wide z: wasted; discarded anyway. |
| Global BN (aggregate BN stats across devices) | A positive pair's two views often land on the same device; per-device BN stats then ENCODE "these samples are together" — a local-statistics leak the model reads to cheat the contrastive task without improving the rep (same class of shortcut as color histogram). Global aggregation removes the leak. Minimal change to a standard ResNet. | Local per-device BN → leak. MoCo: shuffle BN across devices. CPC v2: layer norm instead of BN. Global BN chosen for least architecture change. |
| LARS + linear warmup + cosine decay | At batch 4096–8192 plain SGD with linear LR scaling is unstable (per-layer gradient-norm disparity); LARS scales each layer's update by ‖w‖/‖grad‖ (layer-wise adaptive). Warmup avoids blowing up the random init with a large LR; cosine anneals smoothly. | Plain momentum SGD + linear scaling → unstable at large batch. |
| Weight decay excludes BN & bias params | Decaying normalization/bias params has no regularization meaning and disturbs them. | Include them → harmful. |
| ResNet encoder, unconstrained | The whole point of decoupling task from architecture: use the strongest standard backbone; unsupervised benefits MORE from bigger models. | Constrained backbone (DIM/AMDIM) → weaker, bespoke. |

---

## 5. Code grounding (canonical TF impl in code/) — final code fills the scaffold

- `objective.add_contrastive_loss(hidden, hidden_norm, temperature, tpu_context)`: ℓ2-norm →
  split 2N into two views → cross-replica concat to expose all in-batch negatives →
  four similarity blocks /τ → mask self-similarity with −LARGE_NUM (1e9) → two
  softmax_cross_entropy terms (a→b and b→a). labels/masks one-hot place the positive at the
  counterpart index after concatenation. (objective.py.)
- `model_util.projection_head`: modes none/linear/nonlinear; nonlinear = num_proj_layers of
  linear_layer→BN→ReLU, last layer no bias/ReLU; pretrain returns hiddens_list[-1] (=z),
  finetune returns hiddens_list[ft_proj_selector] (can pick h). (model_util.py.)
- `resnet.BatchNormalization._moments`: override to cross_replica_average mean & variance
  (global BN), variance corrected by + cross_replica_average((group_mean−shard_mean)²).
  (resnet.py.)
- `model_util.get_optimizer`: LARSOptimizer(lr, momentum, weight_decay,
  exclude_from_weight_decay=['batch_normalization','bias','head_supervised']);
  `learning_rate_schedule`: scaled_lr = base·BatchSize/256, linear warmup, cosine decay.
- `data_util.preprocess_for_train`: random_crop_with_resize → random_flip → random_color_jitter;
  blur applied batched (batch_random_blur). color_distortion(s): brightness 0.8s, contrast
  0.8s, saturation 0.8s, hue 0.2s, p=0.8 jitter + p=0.2 grayscale. blur: 50%, σ∈[0.1,2.0],
  kernel = 10% of side.
- `model.build_model_fn`: pretrain path = encoder → projection_head → add_contrastive_loss;
  finetune/eval path = encoder → (selected head layer) → supervised_head + softmax CE.

### Scaffold (context.md) = hollowed final code, pre-method vocabulary only
- `encoder = ResNet(...)`  (prior art, known)
- `def augment(x): # TODO: the augmentation we'll design`  (NO crop/color/blur named)
- `class RepresentationObjective: def __call__(self, encoder, batch): # TODO: the training
  signal we'll design`  (NO projection head / NT-Xent / temperature / negatives / two-views)
- `optimizer = make_optimizer(...)` + LR schedule machinery (exists)
- `def linear_probe_eval(encoder, ...): # freeze encoder, train one linear layer, report acc`
  (the yardstick, predates the method)
- training loop: for batch → views = augment(batch) → loss = objective(encoder, views) →
  optimizer step. ONE big empty slot (the objective) + one (the augmentation). No method-shaped
  sub-stubs.
The final code's pieces map: augment ← preprocess_for_train(3 augs); RepresentationObjective ←
projection_head + add_contrastive_loss (+ global BN inside encoder, LARS as the optimizer);
linear_probe_eval ← unchanged.

---

## 6. Evaluation settings (pre-method facts, no outcomes)
Pretrain unlabeled on ImageNet ILSVRC-2012 (russakovsky2015); small-scale confirmation on
CIFAR-10 (krizhevsky2009). Read-outs: (a) linear evaluation (freeze encoder, train linear
classifier, top-1/top-5); (b) semi-supervised fine-tuning on class-balanced 1%/10% label
subsets; (c) transfer (linear + fine-tune) on Food-101, CIFAR-10/100, Birdsnap, SUN397, Cars,
Aircraft, VOC2007, DTD, Pets, Caltech-101, Flowers (kornblith2019 protocol). No numbers.
