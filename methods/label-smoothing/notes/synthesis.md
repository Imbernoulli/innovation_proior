# Label smoothing synthesis (Phase 1.5)

Target arXiv 1906.02629 "When Does Label Smoothing Help?" Müller, Kornblith, Hinton (NeurIPS 2019) — an ANALYSIS paper. The *technique* label smoothing is PRIOR ART (Szegedy et al. 2016, Inception-v3 / "Rethinking the Inception Architecture", CVPR 2016, arXiv 1512.00567). So the in-frame discovery is: the technique itself (recovered from the overconfidence pain, citing Szegedy 2016 as ancestor) PLUS the understanding of WHEN/WHY it helps (calibration, penultimate-layer geometry) and when it HURTS (distillation). Never cite Müller/the target as artifact. May name "label smoothing" the technique.

## Prior art: label smoothing origin (Szegedy 2016, verified against Inception PDF §7)
- Softmax p(k) = exp(z_k)/Σ exp(z_i). Hard-target loss = cross entropy ℓ = −Σ_k q(k) log p(k), q(k)=δ_{k,y}.
- Gradient ∂ℓ/∂z_k = p(k) − q(k), bounded in [−1, 1].
- With hard target, max log-likelihood is approached only as z_y ≫ z_k ∀k≠y, i.e. the true logit must run away to +∞. Two problems: (1) overfitting (full prob to train label, no generalization guarantee); (2) the gap between largest logit and the rest grows large, and combined with the bounded gradient this REDUCES the model's ability to adapt — too confident.
- Fix: replace q with q'(k) = (1−ε)δ_{k,y} + ε u(k), u = prior over labels; uniform u(k)=1/K -> q'(k) = (1−ε)δ_{k,y} + ε/K. "marginalized label-dropout."
- Decomposition: H(q',p) = (1−ε)H(q,p) + ε H(u,p). Second term = ε·[D_KL(u‖p)+H(u)], penalizes deviation of p from uniform with relative weight ε/(1−ε). Prevents the largest logit running away because all q'(k) have a positive lower bound ε/K, so an infinite logit gap gives infinite CE.
- ImageNet: K=1000, u=1/1000, ε=0.1; consistent ~0.2% absolute improvement top-1 AND top-5.
- (Related prior art: Pereyra 2017 confidence penalty = penalize low-entropy/overconfident outputs = −β H(p); equivalent to LSR with the KL direction reversed. DisturbLabel (Xie 2016) = label dropout; LS is its marginalized version. Bishop 1995 training-with-noise lineage.)

## The analysis contribution (target paper) — the "when/why"
This is the discovery to reconstruct in-frame as the narrator's own investigation.

### 1. Penultimate-layer geometry (§2, verified main.tex)
- Logit z_k = x^T w_k where x = penultimate activations (+1 bias), w_k = class template.
- KEY identity: ‖x − w_k‖² = x^T x − 2 x^T w_k + w_k^T w_k. The x^T x term is common to all k (factors out of softmax), and w_k^T w_k is ~constant across classes. So the softmax over logits is monotone in MINUS the squared Euclidean distance ‖x−w_k‖² to each template. The logit ≈ negative squared distance to template.
- Therefore minimizing CE pushes x toward the correct template w_y. With HARD targets, only the gap z_y − z_k matters and it can be arbitrarily large -> broad clusters, incorrect-class logits free to differ wildly from one another, over-confident large-magnitude activations.
- With LABEL SMOOTHING: the soft target wants a SPECIFIC finite gap between correct and incorrect logits (controlled by ε), AND wants all incorrect logits EQUAL (since q'(k)=ε/K is the same for every wrong class). Equal incorrect logits ⇔ x equidistant from all incorrect templates. Result: activations cluster TIGHTLY around the correct template and equally distant from the others -> tight, equally-separated clusters; on 3 classes the projections form regular triangles; magnitudes bounded (no over-confident runaway).
- Visualization scheme: pick 3 classes, find orthonormal basis of plane through their 3 templates, project penultimate activations onto it.
- For semantically similar classes (two poodles + a tench): hard targets -> similar classes cluster close with isotropic spread, and there's a CONTINUOUS gradient of "how much a poodle resembles this tench"; label smoothing -> arc shape, that continuous resemblance info is ERASED.

### 2. Implicit calibration (§3)
- Modern nets are over-confident / poorly calibrated (Guo 2017): confidence > accuracy. Measure: Expected Calibration Error (ECE) via reliability diagram (bin by confidence, compare avg confidence vs accuracy per bin). Standard fix: temperature scaling (divide logits by T>1 before softmax) — a post-hoc knob.
- Finding: label smoothing IMPLICITLY calibrates — training with ε≈0.05–0.1 gives ECE comparable to a temperature-scaled hard-target model, WITHOUT post-hoc tuning. Because preventing logit runaway = preventing over-confidence directly.
- ECE numbers (Table): CIFAR-100 ResNet-56 baseline ECE 0.150 -> temp-scale 0.021 (T=1.9) -> LS 0.024 (α=0.05). ImageNet Inception-v4 0.071 -> 0.022 (T=1.4) -> 0.035 (α=0.1). EN-DE Transformer 0.056 -> 0.018 (T=1.13) -> 0.019 (α=0.1).
- Why calibration MATTERS for translation: beam search ≈ Viterbi/max-likelihood sequence search; it consumes the soft next-token probabilities, so a better-calibrated model -> better beam search -> higher BLEU. (Vaswani 2017: LS α=0.1 improved BLEU 25.3->25.8 despite WORSE perplexity 4.67->4.92. The BLEU gain is only PARTLY explained by calibration; LS model has worse NLL at all temperatures but better BLEU.)

### 3. Label smoothing HURTS distillation (§4)
- Knowledge distillation (Hinton 2015): train student to match teacher's softened outputs. Loss = (1−β)H(y,p) + β H(p^t(T), p(T)), T = temperature exaggerating differences among incorrect-class probabilities. Useful only if it beats just training the student with label smoothing directly.
- Observed anomaly (MNIST): a teacher trained with LS is MORE accurate (0.59% vs 0.67% with dropout) but distills into a WORSE student (0.91% vs 0.74%). "a teacher with better accuracy is not necessarily the one that distills better."
- Explanation = the tight-cluster geometry. Distillation's value is in the RELATIVE logits among wrong classes — "this 3 looks a bit like an 8" — the dark knowledge. LS forces all wrong-class logits equal and collapses each class to a tight cluster, so different examples of a class carry near-identical relative similarities; the example-specific inter-class resemblance info is ERASED. Hard-target teacher keeps that variation, so it transmits more.
- Quantified via mutual information I(X;Y), X=training-example index, Y=difference of two logits, randomness from data augmentation. Approx as Gaussian. As training proceeds I rises then DECAYS, more so with LS, tending toward log(2) (the extreme: all info discarded except 1 bit = which class) -> nothing beyond the label itself. Estimator: Î(X;Y)=(1/N)Σ_x[−(f(d(z_x))−μ_x)²/(2σ²) − log((1/N)Σ_x e^{−(f(d(z_x))−μ_x)²/(2σ²)})].
- Same root cause (Kornblith 2018): LS impairs transfer learning, which also needs non-class-relevant info in final layers.

## Design-decision -> why (for the technique + the analysis intuitions)
- Mix with UNIFORM u=1/K (vs other priors): equal mass to all wrong classes = no prior structure; unigram LS (Pereyra) for imbalanced labels. Uniform default.
- ε small (~0.1): enough to bound logit gap without destroying the signal.
- The H(q',p)=(1−ε)H(q,p)+εH(u,p) split: shows LS = hard CE + a uniform-deviation penalty; KL view H(u,p)=D_KL(u‖p)+const.
- logit ≈ −‖x−w_k‖²: the geometric lens that explains the clustering.
- equal wrong-class targets -> equidistance -> tight equal clusters: the central mechanism.
- calibration: bounding confidence directly aligns confidence with accuracy.
- distillation harm: tight clusters erase the relative-similarity dark knowledge distillation needs.

## Eval settings (pre-method facts)
CIFAR-10 (AlexNet), CIFAR-100 (ResNet-56), ImageNet (Inception-v4) top-1; EN-DE WMT (Transformer) BLEU/perplexity/NLL; WSJ speech (BiLSTM+Att) WER. Metrics: accuracy, ECE / 15-bin reliability diagram, BLEU, mutual information. ε / α in [0, 0.75]. Train e.g. ResNet-56: batch 128, SGD Nesterov 0.9, lr 0.1 /10 at 32k/48k of 64k iters, wd 1e-4, CE×3, grad clip 1.0, crop+flip aug.

## Unsourced facts: none. Technique formulas from Inception §7 PDF; analysis from main.tex.
