# MoCo — Synthesis Notes (Phase 1.5)

## The pain point at the time (2018–2019)

Unsupervised pretraining had transformed NLP (word2vec, GPT, BERT) but supervised
ImageNet pretraining still dominated vision. Hypothesized reason: NLP signal is
**discrete** (words/sub-words) → a finite tokenized **dictionary** exists, and
unsupervised objectives (predict masked/next token over a vocabulary softmax) are
natural. Vision signal is **continuous, high-dimensional, unstructured** → no
ready-made dictionary. So self-supervised vision had to *build* its dictionary.

A unifying lens (the key reframing): many recent contrastive methods are doing
**dictionary look-up**. There's an encoded query `q`, a set of encoded keys
`{k_0,k_1,...}`, one positive key `k_+` (a different view of the same source), and
the rest are negatives. Train the encoder so `q` matches `k_+` and is dissimilar
to the negatives → InfoNCE. Under this lens the question becomes: **what makes a
good dictionary for contrastive learning?**

Hypothesis = two desiderata, in tension under existing mechanisms:
- **(i) Large.** The negatives are a sample of the continuous visual space; more
  negatives sample it better. (InfoNCE's own MI bound `I >= log N - L` makes "more
  negatives" literally a tighter bound — strong derivation-time motivation.)
- **(ii) Consistent.** All keys should be encoded by the *same or very similar*
  encoder, so the query-key dot products are comparable. If keys come from wildly
  different encoder states, the comparison is apples-to-oranges.

## Load-bearing ancestors (verified against sources)

### Hadsell, Chopra, LeCun 2006 — contrastive loss (DrLIM)
- Origin of "contrastive": loss on **pairs**. Pull similar pairs together in
  embedding space; push dissimilar pairs apart but only up to a **margin** `m`:
  `L = (1-Y) * (1/2) D^2 + Y * (1/2) max(0, m - D)^2`, D = embedding distance.
- Core idea MoCo inherits: the target is not a fixed label but **the data's own
  representation**, computed on the fly by a network → the target can *vary*
  during training. Limitation: pairwise/margin form; modern variants moved to
  softmax-over-many-negatives (InfoNCE) which is empirically much stronger.

### Gutmann & Hyvärinen 2010 — NCE
- Estimate an unnormalized model by **binary classification**: data vs. a known
  noise distribution. Turns an intractable partition-function normalization into a
  logistic-regression problem. This is the trick that makes instance-level softmax
  (over N≈1.28M classes) tractable.

### Wu, Xiong, Yu, Lin 2018 — Instance Discrimination + Memory Bank  (Fig 3b)
- Pretext task MoCo adopts: **every image is its own class**. Non-parametric
  softmax over instances:
  `P(i|v) = exp(v_i^T v / τ) / Σ_j exp(v_j^T v / τ)`.
- Full softmax over all n images is intractable → approximate with **NCE**:
  noise `P_n = 1/n` uniform, posterior `h(i,v) = P(i|v) / (P(i|v) + m P_n(i))`,
  objective `J = -E_{Pd}[log h] - m E_{Pn}[log(1-h)]`.
- **Memory bank**: store one feature vector per dataset image (an N×128 table).
  Each step, draw negatives by *sampling rows from the bank* — no extra forward
  pass, so the effective dictionary = whole dataset (huge). After computing a
  feature, write it back: `v_i ← (1-λ) v_i + λ f_θ(x_i)`, plus a **proximal**
  regularizer `λ‖v_i^(t) − v_i^(t-1)‖²` to damp feature drift.
- **The fatal inconsistency (MoCo's opening)**: a bank entry was written the last
  time *that specific image* was sampled — potentially a whole epoch ago, by a very
  different encoder state. So the negatives in any one step come from encoders
  scattered across the **entire past epoch** → highly *inconsistent* dictionary.
  The bank's momentum is on the *stored features of the same sample*, NOT on the
  encoder — irrelevant to fixing cross-key consistency.
- Also: the bank stores all N samples → does not scale to billion-image data.

### van den Oord, Li, Vinyals 2018 — CPC / InfoNCE
- The loss MoCo uses:
  `L_N = -E[ log f(x_+,c) / Σ_{x_j∈X} f(x_j,c) ]`, X = {1 positive + (N−1) neg}.
- Optimal classifier: `p(d=i|X,c) = (p(x_i|c)/p(x_i)) / Σ_j (p(x_j|c)/p(x_j))`
  ⇒ optimal `f ∝ p(x_+|c)/p(x_+)` (a **density ratio**, not a calibrated prob).
- MI bound (derivation I must re-live inline):
  `L_N^opt = E[ log(1 + (p(x)/p(x|c))(N−1)) ] >= E[ log( (p(x)/p(x|c)) N ) ]
           = −I(x,c) + log N`  ⇒  `I(x,c) >= log N − L_N`.
  **This is the rigorous "bigger dictionary = tighter MI bound" argument.**
- Limitation MoCo reacts to: in CPC/end-to-end, the N negatives live in the
  **current minibatch** (or current forward pass), so N is capped by GPU memory /
  batch size. Pretext tasks that inflate N via many spatial positions need
  patchifying / custom receptive fields → hard to transfer to downstream tasks.

### Goyal et al. 2017 — "ImageNet in 1 hour" (large-batch SGD)
- Linear LR scaling + warmup let large batches train. But large-batch optimization
  is still an open problem; scaling the dictionary by scaling the batch (end-to-end
  route) inherits all of large-batch's difficulty (needs the scaling rule, accuracy
  drops ~2% at batch 1024 without it, and it's questionable past that). → motivates
  *decoupling* dictionary size from batch size.

### Ye 2019 / Bachman 2019 — two augmented views as the positive pair
- The positive pair = two random augmentations of the same image. MoCo adopts this
  (rather than image-vs-its-bank-entry).

## The three mechanisms (Fig 2) — the design fork

| mechanism | how keys made | dict size | consistency |
|---|---|---|---|
| (a) end-to-end | both encoders backprop, keys = current batch | = batch (small, GPU-bound) | **high** (same encoder) |
| (b) memory bank | sample stored features | huge (=dataset) | **low** (encoders span an epoch) |
| (c) MoCo | momentum encoder + queue | large (decoupled K) | **high** (slow encoder) |

End-to-end maxes consistency, loses size. Memory bank maxes size, loses
consistency. **MoCo's whole contribution = get both at once.**

## The two MoCo ideas (derived, with the *why*)

### Idea 1 — Dictionary as a QUEUE (decouple size from batch)
- Want many keys but can't backprop through all of them and can't fit them in one
  batch. Observation: keys from the *immediately preceding* minibatches are still
  usable as negatives — just reuse them. Maintain a FIFO **queue** of encoded keys:
  enqueue current minibatch's keys, dequeue the oldest. Queue length `K` is a free
  hyperparameter, **independent of batch size N** (K=65536 ≫ N=256).
- Why a *queue* specifically (not a random buffer)? The oldest keys were made by
  the most-outdated encoder → least consistent with current keys → evicting
  oldest-first is exactly the right staleness policy. The queue gives "large" *and*
  keeps the set fresh.

### Idea 2 — MOMENTUM key encoder (restore consistency)
- Queue keys come from past minibatches, encoded by past encoder states. If `f_k`
  changes fast (e.g. `f_k = f_q` copied each step, gradient ignored), the queue
  mixes wildly different encoders → inconsistent → **training fails** (empirically:
  m=0 oscillates/diverges).
- Can't backprop into the queued keys (gradient would have to flow to all K samples
  across many past steps — intractable). So `f_k` must be updated *without*
  gradient, but *smoothly* so the past keys stay nearly-consistent with now.
- Resolution = exponential moving average of the query encoder:
  `θ_k ← m θ_k + (1−m) θ_q`, only `θ_q` gets gradients.
- Why this keeps keys consistent: `θ_k` is a running average, it moves a factor
  `(1−m)` slower than `θ_q`. With m=0.999, over the ~K/N≈256 steps it takes to
  refill the queue, `θ_k` barely moves → all keys in the queue were encoded by
  nearly the same `θ_k` → consistent. **Large m is essential**: ablation
  m∈{0,0.9,0.99,0.999,0.9999} → {fail,55.2,57.8,59.0,58.9}; 0.999 default. m=0.9
  too fast; m=0 fails entirely. This is the empirical heart of "consistency matters".

## Other design choices → why
- **L2-normalize q,k; dot product = cosine; τ=0.07** (from Wu2018a). Normalization
  puts features on a unit sphere so dot product is bounded cosine sim; τ sharpens
  the softmax (small τ ⇒ harder concentration). τ=0.07 inherited.
- **dim=128** output (from Wu2018a).
- **Loss = (K+1)-way CrossEntropy, label 0** (positive is logit index 0). This is
  literally InfoNCE: `logits = [q·k_+ ; q·queue]/τ`, softmax-CE with target 0.
- **K=65536, m=0.999** for the main results.
- **Standard ResNet-50 encoder, no architectural surgery** — deliberately, so
  features transfer cleanly to detection/segmentation (unlike CPC's patchify /
  AMDIM's receptive-field surgery).
- **Init `θ_k ← θ_q`** at start (copy params once).
- **Queue init**: random + L2-normalized; `K % batch == 0` for clean enqueue.
- **Shuffling BN** (the subtle bug-fix): both encoders use BN. Plain BN lets the
  model "cheat" — intra-batch statistics leak which sample is the positive (the
  query and its key share a batch and thus share BN stats → a signature). Fix:
  for `f_k`, shuffle sample order across GPUs before encoding (each GPU computes BN
  on a different sub-batch), then unshuffle; `f_q`'s order untouched. Ensures the
  query and its positive key see *different* BN statistics → no leak. Without it,
  pretext train-acc shoots to >99.9% while kNN val-acc collapses (overfitting/
  cheating). Memory bank doesn't have this issue (positive key came from a past
  batch). Implemented via all_gather across DDP ranks + random permutation
  broadcast from rank 0.
- **`k = k.detach()` / no_grad on key path** — keys never receive gradient;
  consistent with "only θ_q is learned".
- **Optimizer**: SGD, lr=0.03 (IN-1M, batch 256, 8 GPUs), momentum 0.9, wd 1e-4,
  200 epochs, ×0.1 at 120/160. (Pretraining setting — pre-method-ish recipe.)

## Code grounding (canonical impl, KaimingHe-patch branch of facebookresearch/moco)
- `MoCo(nn.Module)`: builds `encoder_q`, `encoder_k` (copies params, freezes k),
  registers `queue` (dim×K, normalized) and `queue_ptr` buffers.
- `_momentum_update_key_encoder`: the EMA loop `param_k = m*param_k + (1-m)*param_q`.
- `_dequeue_and_enqueue`: all_gather keys, write at ptr, advance ptr mod K.
- `_batch_shuffle_ddp` / `_batch_unshuffle_ddp`: shuffling-BN via DDP all_gather +
  randperm broadcast + argsort restore.
- `forward(im_q, im_k)`: q=normalize(encoder_q(im_q)); under no_grad: momentum
  update, shuffle, k=normalize(encoder_k), unshuffle; `l_pos = einsum nc,nc->n`;
  `l_neg = einsum nc,ck->nk`; `logits = cat([l_pos,l_neg])/T`; `labels = 0`;
  enqueue k; return logits, labels. Outer loop: `CrossEntropyLoss`, SGD step.

## Scaffold ↔ final-code correspondence (for context.md)
Pre-method scaffold = a generic contrastive-learning harness: a `base_encoder`
(ResNet), an L2-normalize, an InfoNCE/CrossEntropy loss, a data pipeline giving two
augmented views, an SGD loop — with ONE empty slot: `class ContrastiveModel:` whose
`__init__` (how to maintain the key encoder + the negative set) and `forward` (how
to produce logits) are `# TODO`. The reasoning fills that slot with queue + momentum
encoder. Must NOT pre-name queue/momentum in the scaffold.
