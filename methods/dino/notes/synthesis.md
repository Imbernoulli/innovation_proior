# DINO — synthesis notes (Phase 1.5)

## The pain point / research question
- ViT in vision (2020-21): competitive with convnets only with huge supervised pretraining (JFT-300M) or distillation from a convnet teacher (DeiT). Plain supervised ImageNet ViT underwhelms; features show "no unique properties".
- In NLP, the win came from *self-supervised* pretraining (BERT masked-token, GPT LM): richer signal than one label/sentence. Image-level supervision collapses a rich image to one of ~1000 categories.
- Question: does SSL pretraining unlock something in ViT features that supervision does not? And can we design a *simple* SSL objective (no negatives, no clustering machinery) that works on both ViT and convnets?

## Load-bearing ancestors and exactly where each falls short

### Instance discrimination + contrastive (NCE/InfoNCE) — Wu 2018, SimCLR (Chen 2020), MoCo (He 2020)
- Treat each image as its own class; pull augmentations of the same image together, push different images apart. NCE/InfoNCE compares a positive against many negatives.
- Collapse is avoided *by construction*: the negatives in the denominator stop everything mapping to one point.
- Cost: needs many negatives → large batches (SimCLR) or a memory bank / momentum queue (MoCo). The momentum encoder in MoCo exists to keep the queue's keys consistent as the encoder drifts.
- Gap: dependence on negatives / large batch; the contrastive denominator is the load-bearing anti-collapse device.

### Momentum encoder — MoCo (He 2020)
- A second network whose weights are an EMA of the online net: θ_k ← λ θ_k + (1−λ) θ_q. Originally a *consistency* trick so the queued keys don't go stale. Stop-gradient through it.

### Mean teacher — Tarvainen 2017 (semi-supervised)
- Teacher = EMA (Polyak-Ruppert) of student weights; student trained to match teacher's predictions on perturbed inputs (consistency). Weight-averaging gives a better model than any single iterate → teacher is a free ensemble that leads the student. This is the *self-training/co-distillation* framing, not the contrastive framing.

### Knowledge distillation — Hinton 2015; self-training soft labels — Xie 2020 (Noisy Student)
- KD: train student g_s to match a *fixed pretrained* teacher g_t's softened (temperature) output distribution via cross-entropy on K classes. Self-training: teacher produces soft pseudo-labels on unlabeled data. Both assume a teacher given a priori.
- Gap to bridge: no labels, no pretrained teacher. Need to *build* the teacher on the fly.

### BYOL — Grill 2020 (the closest ancestor and the central puzzle)
- No negatives at all. Online net (encoder f + projector + *predictor* q) regresses (MSE on ℓ2-normalized outputs) the target net's projection of another view. Target net = EMA of online. Stop-gradient on target.
- The puzzle: with no negatives, why doesn't it collapse to a constant? The accepted answer at the time: the *predictor* asymmetry + the EMA target together prevent the trivial solution (predictor can't instantly track a constant target; EMA makes the target a slow-moving function of the past student). Also BYOL was reported to lean on batch normalization in the heads (the BN-removal debate, Richemond 2020 "BYOL works even without BN" with group norm + weight standardization replacing it).
- Gap: relies on predictor + (debated) BN; asymmetric architecture; MSE-in-feature-space objective; no probabilistic interpretation.

### SimSiam — Chen 2020 (exploring)
- BYOL without the momentum encoder: stop-gradient + predictor alone prevent collapse. Confirms predictor + stop-grad is the anti-collapse mechanism in that family; momentum only helps performance.

### SwAV — Caron 2020 (the source of two components)
- Online clustering: assign features to K prototypes, enforce *equipartition* across the batch via the Sinkhorn-Knopp optimal-transport algorithm (balanced assignments → no dimension can dominate → no collapse), then swap-predict one view's code from the other view. No negatives.
- Introduces **multi-crop**: 2 global high-res crops + several low-res local crops; "swap" matching across views; cheap because local crops are small. Big free accuracy boost.
- Gap: Sinkhorn is a batch-coupled iterative normalization (couples samples in a batch); prototype/cluster machinery is heavier than necessary.

### Codistillation — Anil 2018
- Several students with same architecture distill from each other during training. DINO's teacher is *not* a peer distilling back; it is a weight-average of the student → one-directional.

## The method, derived

Frame SSL as **knowledge distillation with no labels and no a-priori teacher**.
- Student g_θs, teacher g_θt, *same architecture* g = h∘f (backbone f + projection head h). No predictor (unlike BYOL) → student and teacher are architecturally identical.
- Both map an image to a K-dim distribution via softmax with temperatures τ_s, τ_t:
  P_s(x)^(i) = softmax(g_θs(x)/τ_s)_i ; P_t with τ_t.
- Loss: cross-entropy H(P_t, P_s) = −Σ_k P_t^(k) log P_s^(k), matched across **different** views.
- Multi-crop: set V = {2 global x_1^g, x_2^g} ∪ {local crops}. All crops → student; only global crops → teacher. Sum over teacher-global × student-other-view pairs:
  min_θs  Σ_{x∈{x1g,x2g}} Σ_{x'∈V, x'≠x} H(P_t(x), P_s(x')).
  "local-to-global": student must predict the global/teacher distribution from a small local crop.
- Stop-gradient on teacher; gradients flow only to student (θ_s by SGD/AdamW).

### Teacher construction
- No teacher given. Build from past students. Ablations:
  - student copy / previous iteration → does NOT converge (needs more normalization).
  - previous epoch → works, ~66.6 k-NN (competitive but not best).
  - **EMA / momentum**: θ_t ← λ θ_t + (1−λ) θ_s, λ cosine 0.996→1. Best. Acts as Polyak-Ruppert averaging = a running ensemble; teacher is *consistently better* than student throughout training and so provides higher-quality targets — a property not seen in MoCo/BYOL (because here teacher and student share architecture and objective symmetrically, it reads as mean-teacher self-distillation).

### Avoiding collapse — the centering + sharpening argument (DERIVED)
With no negatives, no predictor, no Sinkhorn — what stops collapse? Two trivial solutions:
1. **One dimension dominates** P_t → near one-hot, same for every input (constant output).
2. **Uniform** P_t = 1/K for every input (constant output).
Both are collapse: P_t independent of input ⇒ nothing to learn.

Two cheap teacher-only operations, pushing in *opposite* directions:
- **Centering**: subtract an EMA center c from teacher logits before softmax: g_t(x) ← g_t(x) − c, with c ← m c + (1−m) (1/B) Σ_i g_θt(x_i). Equivalent to a bias on the teacher. Centering prevents *any single dimension from dominating* (it recenters the mean logit), but on its own it pushes the output toward **uniform** (flattens differences) → collapse mode 2.
- **Sharpening**: low teacher temperature τ_t (e.g. 0.04–0.07). Sharpening concentrates mass → prevents the *uniform* collapse, but on its own encourages **one dimension to dominate** → collapse mode 1.
- They are complementary: centering counters mode 1 but invites mode 2; sharpening counters mode 2 but invites mode 1. Apply both → effects balance → stable, in the presence of a momentum teacher.

**Decomposition proof of complementarity** (decompose the loss):
H(P_t, P_s) = h(P_t) + D_KL(P_t ‖ P_s), where h is entropy.
- KL → 0 ⇒ teacher output is constant ⇒ collapse. So "KL→0" is the collapse detector.
- When one operation is missing, KL → 0 (collapse), and the *entropy h of the teacher* converges to a tell-tale value:
  - no centering → h → 0 (one-hot / one dim dominates), and
  - no sharpening → h → −log(1/K) = log K (uniform).
  Two different collapse entropies ⇒ the two operations induce *opposite* failure modes. Both together hold KL away from 0.

Empirical hyperparameter facts (diagnostic, pre-method-style): centering EMA m robust over 0..0.99, collapses at m=0.999 (too slow). τ_t must be ≤ ~0.06 to avoid collapse (τ_t=0.08 → loss converges to ln K = uniform collapse); warm up τ_t 0.04→0.07 over first 30 epochs to avoid early collapse. τ_t→0 = argmax = hard one-hot. τ_s = 0.1.

### Why centering (vs Sinkhorn / softmax-over-batch)
- Sinkhorn (SwAV) and softmax-over-batch couple samples within a batch and iterate. Centering depends only on *first-order* batch statistics (a mean) and is EMA'd over batches → works across batch sizes (down to bs=8), low batch dependence. Trades a little stability for batch-independence. With a momentum teacher, centering+sharpening alone suffice (Sinkhorn adds little).

## Design-decision → why table

| Decision | Why this, why not the alternative |
|---|---|
| Cross-entropy on softmax K-dim outputs (not MSE) | Casts SSL as KD; gives the entropy/KL decomposition that explains collapse; CE clearly beats MSE empirically (76.1 vs 62.4 linear). MSE (BYOL-style) "works" but much worse and has no probabilistic reading. |
| No predictor (student=teacher architecture) | BYOL/SimSiam need the predictor to break collapse; here collapse is handled by centering+sharpening, so predictor is redundant — adding it changes ~0.5%. Symmetric architecture reinforces the mean-teacher reading. |
| EMA momentum teacher (λ 0.996→1) | Polyak-Ruppert ensemble: teacher consistently better than student → higher-quality targets that lead training. Student-copy/prev-iter teachers don't converge; prev-epoch works but worse. |
| Stop-gradient on teacher | Teacher is a *target*; gradients only update the student. Teacher updated by weight EMA, not backprop. |
| Centering (EMA bias on teacher logits) | Kills the dominant-dimension collapse with only first-order batch stats → batch-size robust. Sinkhorn/softmax-batch also work but couple the batch and iterate; with momentum they add nothing. |
| Sharpening (low τ_t) | Kills the uniform collapse. Complementary to centering. Must stay ≤~0.06; warm up to avoid early-training collapse. |
| τ_s = 0.1, τ_t < τ_s | Teacher sharper than student is the distillation signal — student is pulled toward a peakier target. |
| Multi-crop: 2 global (224²) + several local (96²), local→student only | "Local-to-global" forces small crops to predict the global view's distribution; many cheap view comparisons. Biggest single accuracy lever for DINO (+3.4% linear). Global-only to teacher keeps targets high-quality. |
| Projection head: 3-layer MLP (GELU, hidden 2048) → ℓ2-normalize → weight-normalized last linear, output K=65536, bottleneck d=256 | SwAV-style. ℓ2 bottleneck *stabilizes* deep heads (without it, 3-4 layer heads collapse, k-NN 0.1). Large K helps. Weight-norm last layer with g fixed to 1 (and frozen first epoch) stabilizes early training. |
| BN-free (no BN in head for ViT) | ViT has no BN; removing BN from head has ~0 effect (69.7 vs 68.6) → makes the whole system BN-free, avoiding cross-GPU BN sync cost and the BYOL "needs BN" dependence. |
| AdamW, lr=0.0005·bs/256, 10-ep warmup, cosine decay; WD cosine 0.04→0.4; clip grad 3.0; freeze last layer 1st epoch | ViT training stability recipe (DeiT-style). |
| k-NN evaluation (k=20, τ=0.07) | Hyperparameter-free probe of feature quality; DINO ViT features are unusually k-NN friendly. |

## Code grounding (facebookresearch/dino)
- `DINOLoss.forward`: student_out/τ_s chunked over ncrops; teacher_out = softmax((teacher − center)/τ_t) detached, chunked over 2 globals; double loop over (teacher global iq, student view v), skip v==iq, CE = sum(−q·log_softmax(student_v)); average over terms; then update_center.
- `update_center`: batch mean of teacher_output (all-reduced), EMA with center_momentum=0.9.
- EMA teacher in train loop: param_k = m·param_k + (1−m)·param_q, m from cosine schedule 0.996→1.
- `DINOHead`: 3-layer MLP+GELU → ℓ2 normalize → weight_norm linear to out_dim, weight_g=1 (frozen if norm_last_layer).
- `MultiCropWrapper`: groups crops by resolution, one backbone forward per resolution, head on concatenated features.
- teacher forward only on images[:2] (globals); student on all crops.
- cancel_gradients_last_layer (freeze_last_layer epochs); clip_gradients.
