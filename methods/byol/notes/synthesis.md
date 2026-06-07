# BYOL synthesis notes (Phase 1.5)

## The pain point at the time (2020)
- State-of-the-art self-supervised image representation = contrastive learning (SimCLR, MoCo, CPC, CMC, InfoMin).
- Core mechanism: pull together representations of two augmented views of the SAME image (positive pair), push apart representations of views of DIFFERENT images (negative pairs). Discrimination, not raw prediction.
- WHY negatives at all: predicting one view's representation directly from another's, in representation space, admits trivial collapse — a constant representation is perfectly predictive of itself. Contrastive methods reframe prediction as discrimination so the constant solution is no longer optimal (you can't discriminate with constant features). Negatives are the anti-collapse term.
- Cost of negatives:
  - Need MANY negatives close enough to be hard. SimCLR -> large batches (batch 4096-8192 => up to ~16k negatives/positive). MoCo -> memory-bank queue + momentum encoder to keep a large consistent dictionary. Others -> hard-negative mining.
  - Performance critically depends on the augmentation set (SimCLR collapses to "color histogram" shortcut if color distortion removed, because crops of the same image share color histograms; needs strong color aug to break the shortcut).
- Research question: are negative pairs INDISPENSABLE to prevent collapse while keeping high performance? Can we learn directly by prediction in representation space, no negatives, no large-batch dependence?

## Load-bearing ancestors (verified against biblio + web)
1. **SimCLR (Chen et al. 2020)** — NT-Xent: l2-normalized embeddings, cosine sim / temperature τ, softmax over (1 positive + 2(N-1) negatives in-batch). Adds a PROJECTION HEAD (MLP) g on top of encoder f; loss applied on projection z=g(y), but the representation kept for downstream is y (pre-projection) — projection improves linear-eval performance. Strong augmentations crucial. Big-batch dependent. This is the direct baseline BYOL reformulates.
2. **MoCo (He et al. 2019)** — contrastive with a QUEUE of negatives + a MOMENTUM ENCODER (key encoder = EMA of query encoder, θ_k ← m θ_k + (1-m) θ_q). The EMA keeps the dictionary keys consistent across steps despite the encoder changing. KEY DONOR IDEA TO BYOL: a slow EMA "target" network produces stable targets. But in MoCo the EMA still serves the contrastive objective (consistent negatives). BYOL repurposes it as the prediction-target producer, dropping negatives entirely.
3. **Mean Teacher (Tarvainen & Valpola 2017)** — semi-supervised. Student network + teacher = EMA of student weights. Consistency loss = l2 between teacher and student predictions (softmax logits), ADDED TO a supervised classification loss on a few labels. The classification loss is what grounds it / prevents collapse. BYOL = "unsupervised mean teacher" but WITHOUT the classification loss AND with an extra predictor; paper shows removing predictor + removing classif loss => collapse (mean-teacher-without-labels collapses). So the predictor is the new anti-collapse ingredient.
4. **PBL (Guo et al. 2020, RL)** — predictions of bootstrapped latents: two networks, each provides targets for the other. BYOL simplifies: one network + its own slow EMA, no second network.
5. **Deep RL target networks (DQN 2015, DDPG 2016)** — bootstrapping the Bellman target off a slow/frozen copy of the network stabilizes the moving-target problem. DDPG specifically uses a soft (EMA) target update τθ_target+(1-τ)θ. BYOL borrows the soft target update to stabilize the representation bootstrap.
6. **DeepCluster (Caron 2018)** — bootstraps prior representation into pseudo-labels (cluster indices) to train the next; avoids negatives but needs costly clustering + anti-collapse precautions. Motivates "bootstrap representations directly".
7. **Cross-view prediction framework (Becker & Hinton 1992)** — predict one view from another as the SSL principle.

## The core discovery / motivating experiment (THE seed)
- Predict a FIXED RANDOMLY INITIALIZED network's projection as the target. Cannot collapse (target is fixed, not trained). Empirically: predicting the random target gives 18.8% top-1 linear eval — MUCH better than the random net's own representation (1.4%). So "train a new (online) representation to predict a fixed (target) representation" yields a strictly BETTER representation than the target.
- => Bootstrap idea: take the improved online net as the NEW target, train again, iterate => sequence of improving representations. Generalize discrete checkpoints into a continuous slow EMA of the online net as the target. This is BYOL.

## The method (final landing)
- Two nets, same architecture, different weights:
  - online θ: encoder f_θ -> projector g_θ -> predictor q_θ
  - target ξ: encoder f_ξ -> projector g_ξ (NO predictor — asymmetry!)
- From image x, two augs t~T, t'~T': v=t(x), v'=t'(x).
- Online on v: y_θ=f_θ(v), z_θ=g_θ(y_θ), prediction q_θ(z_θ).
- Target on v': y'_ξ=f_ξ(v'), z'_ξ=g_ξ(y'_ξ).  stop-gradient on target.
- l2-normalize both, MSE loss:
  L = ||  qbar_θ(z_θ) - zbar'_ξ ||^2 = 2 - 2 * <q_θ(z_θ), z'_ξ> / (||q_θ(z_θ)|| ||z'_ξ||).
- Symmetrize: also feed v' to online, v to target; L^BYOL = L + Ltilde.
- Updates:
  θ ← optimizer(θ, ∇_θ L^BYOL, η)   [only θ; stop-grad on ξ]
  ξ ← τ ξ + (1-τ) θ                  [EMA / soft target update, from DDPG]
- At end keep only f_θ as the representation. Everything else discarded.

## WHY each design choice (design-decision -> reason -> rejected alternative)
- **No negatives / pure prediction objective**: negatives are the established collapse-preventer; the whole question is whether they're necessary. The fixed-random-target experiment shows prediction alone can improve a representation, so try to keep only prediction.
- **Slow EMA target instead of online-as-its-own-target**: if you regress the online net onto ITSELF (target=online, gradient flowing), there is no resistance to collapse — both sides can race to a constant; ablation τ=0 (instant copy, β=0) => 0.2-0.3% (collapse). A FIXED target avoids collapse but can't improve past one round. EMA = the "iterate the bootstrap" generalization: a delayed, stable version of the online net; τ between 0.9 and 0.999 all work (>68%); τ=1 (never update) = stuck at random target quality; τ=0 (instant) = destabilizes. The EMA also keeps targets STABLE & STALE — ablation shows stability/staleness, not the stop-gradient per se, is the main source of gain over plain SimCLR-style self-target.
- **Predictor q on online branch only (asymmetry)**: without the predictor, BYOL = unsupervised mean teacher; ablation (predictor removed, target net kept, β=0) => 0.2% (collapse). The predictor breaks the symmetry between the two branches so the online net is NOT simply asked to equal the target; it's asked to predict the target's EXPECTED value E[z'|z_θ]. This is the crux of the collapse argument.
- **WHY predictor avoids collapse (the conditional-variance argument)**: with optimal predictor q* = E[z'_ξ | z_θ], BYOL's online update follows (in expectation, via envelope theorem, taking gradient only through q's input) the gradient of E[Σ_i Var(z'_{ξ,i} | z_θ)] — the expected conditional variance of the target given the online projection. Key facts:
  - Var(X|Y,Z) ≤ Var(X|Y): adding info never increases conditional variance => the online net cannot reduce the loss by DISCARDING information; collapsing z_θ to a constant gives Var(z'|const) ≥ Var(z'|z_θ), i.e. a constant online projection is the WORST, not the best => constant equilibrium is unstable.
  - If one instead minimized that conditional variance w.r.t. ξ (the target), the minimizer IS a constant z'_ξ (collapse). BYOL does NOT take a gradient step on ξ toward minimizing the loss; ξ only moves toward θ via EMA. So the collapse direction (move ξ to constant) is never taken. This is why "there is no joint loss L(θ,ξ) that BYOL descends" — like GANs. ξ-update is NOT -∇_ξ L.
  - Envelope theorem detail: d/dθ E[L(q*(θ,ξ,z_θ), z'_ξ)] = E[∂L/∂q · ∂q*/∂θ] + E[∂L/∂q · ∂q*/∂z · ∂z_θ/∂θ]. First term = 0 by optimality of q* (∂L/∂q=0 at optimum). Second term = the direction BYOL actually follows (grads only through predictor's input z). So with optimal predictor, BYOL ≈ minimizing conditional variance over θ.
- **Why EMA (not hard copy) given the variance argument**: a hard copy θ→ξ each step WOULD propagate new variability too, BUT sudden target changes break the "predictor is optimal" assumption (the loss is then no longer ≈ conditional variance). EMA keeps targets changing slowly so the predictor stays near-optimal. Confirmed: you CAN remove EMA (τ=0 hard copy) without collapse IF you keep the predictor near-optimal — by giving the predictor a higher learning rate (λ=10 => 66.5%) or solving the optimal linear predictor in closed form per batch (52.5%); but raising BOTH projector and predictor LR fails (~25%). So the predictor near-optimality is the real anti-collapse mechanism; the target net's role is to keep it near-optimal.
- **Project before predicting (loss on z=g(y), keep y)**: inherited from SimCLR — applying loss on a lower-dim projection and keeping the pre-projection y for downstream improves linear-eval. Predictor/projector depth 2 (one hidden layer) is best in ablation; depth 1 (linear) worse, depth 3 no better. Projection dim ~256 (plateau from 64-512).
- **l2-normalization + MSE = cosine loss**: L = ||x̄-ȳ||² = 2-2 cos. l2-norm performs best (72.5) vs no-norm (67.4, projection norm explodes to 3e6 but still ok) vs batchnorm-in-loss (65.3). Normalization stabilizes scale.
- **Projector MLP**: Linear(4096) -> BN -> ReLU -> Linear(256). BN inside the MLP (unlike output, which is NOT batch-normed, contrary to SimCLR). Predictor q same architecture as g.
- **Symmetrized loss**: use both (v->online, v'->target) and (v'->online, v->target); doubles signal, uses both views fully.
- **Augmentations**: same as SimCLR (random crop+resize, flip, color jitter, grayscale, Gaussian blur, solarization). T and T' asymmetric (blur prob 1.0 vs 0.1; solarize 0 vs 0.2). BYOL much more robust to removing color (−9.1 vs SimCLR −22.2) and to crop-only (59.4 vs 40.3): because BYOL is incentivized to retain ANY info the target captured (to predict it better), not just info that discriminates between images; so it doesn't fall into the color-histogram shortcut.
- **Robust to batch size**: only batch-dependence is BatchNorm in encoder (no negatives => no need for many in-batch comparisons). Stable 256-4096; SimCLR degrades faster.
- **Optimizer**: LARS (layer-wise adaptive rate scaling) for large batch, cosine decay LR over 1000 epochs, 10-epoch warmup. base LR 0.2 scaled by batch/256. weight decay 1.5e-6, excluding biases & BN params from both LARS adaptation and WD. Removing WD => divergence.
- **τ schedule**: τ starts 0.996 (τ_base) and is annealed to 1 over training via τ = 1-(1-τ_base)(cos(πk/K)+1)/2. Early on target moves faster (online is improving fast); late, target nearly frozen (online converged).

## Contrastive recasting (unifying frame, from ablation + appendix)
- General InfoNCE^{α,β}: positive term (2/B)Σ S(v_i,v'_i) minus β·(2α/B)Σ ln Σ_{neg} exp(S/α).
  - S(u1,u2) = <φ(u1),ψ(u2)>/(||·||·||·|). 
  - SimCLR: φ=ψ=z_θ (no predictor, no target), β=1.
  - BYOL: φ=q_θ(z_θ) (predictor), ψ=z_ξ (target net), β=0 (no negative term).
- Derivation from standard InfoNCE: take factored InfoNCE = (1/B)Σ ln[ f(v_i,v'_i) / ((1/B)Σ_j exp f(v_i,v'_j)) ]; add in-batch negatives (v_i,v_j); expand log of sum => ln B + (1/B)Σ f(v_i,v'_i) - (1/B)Σ ln(Σ_{j≠i} exp f(v_i,v_j) + Σ_j exp f(v_i,v'_j)). Drop ln B (const in θ), multiply by 2α, set f=S/α, multiply negative term by β. β=1, /2α recovers SimCLR.
- Ablation intermediate variants (Table simclr_to_pbl, 300ep): BYOL(pred+target,β=0)=72.5; +negatives β=1 =>70.9 (hurts, no retune); target+no pred β=1=70.7; SimCLR β=1=69.4; pred only β=1=69.1; pred only β=0=0.3 (collapse); target only β=0=0.2 (collapse); nothing β=0=0.1. Only pred+target with β=0 avoids collapse. Adding a target net to SimCLR alone gives +1.6 (stabilization, recasts MoCo's momentum encoder benefit as stabilization not just more negatives).

## Canonical code (grounded)
- JAX/Haiku official pseudocode (appendix J) + deepmind-research/byol repo (byol_experiment.py, networks.py, optimizers.py in code/).
- regression_loss(x,y) = -2 * mean( sum(x*y,-1) / (norm_x*norm_y) )  [= cosine loss up to constant].
- update_fn: grad of loss_fn wrt online_params only (argnums=0); LARS step; target_params = tree_map(lambda x,y: x+(1-tau)*(y-x), target, online); tau from cosine schedule.
- stop_gradient on target projection (explicitly noted "not strictly necessary since we only diff wrt online_params, kept to indicate gradients don't flow into target").

## Evaluation settings (pre-method, no outcomes)
- ImageNet ILSVRC-2012 pretrain. Linear eval protocol (freeze rep, train linear classifier; SGD+Nesterov, 80 epochs, sweep LR). Semi-supervised (fine-tune on 1%/10% splits). Transfer: Food101, CIFAR10/100, Birdsnap, SUN397, Cars, Aircraft, VOC2007, DTD, Pets, Caltech101, Flowers (linear + fine-tune). VOC seg (FCN), VOC det (Faster R-CNN), NYUv2 depth. Metrics: top-1/top-5, mAP, mIoU, AP50, depth rel/rms/pct<1.25^n. ResNet-50 (and wider/deeper) encoders.
</content>
</invoke>
