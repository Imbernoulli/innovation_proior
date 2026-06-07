# SimCLR — load-bearing ancestors and field state (Phase 1 research notes)

Verified against primary sources in refs/ where possible (CPC, MoCo, InstDisc PDFs) and against SimCLR's own text (main.tex / main.bbl).

## Field state at the time (late 2019 / early 2020)
- Two big families of unsupervised visual representation learning: generative (RBM/DBN hinton2006, VAE kingma2013, GAN goodfellow2014) — model pixels, expensive, maybe unnecessary for representations; and discriminative — train on pretext tasks where inputs+labels come from the data itself.
- Handcrafted pretext tasks were the "renaissance" starter: relative patch location (doersch2015), jigsaw (noroozi2016), colorization (zhang2016), rotation prediction (gidaris2018). Good with bigger nets + longer training (kolesnikov2019 "revisiting"), but rely on ad-hoc heuristics → limited generality.
- Contrastive learning in latent space was the rising star and held SOTA: CPC/CPCv2 (oord2018, henaff2019), DIM/AMDIM (hjelm2018, bachman2019), InstDisc (wu2018), CMC (tian2019), MoCo (he2019), PIRL (misra2019), Local Aggregation (zhuang2019).
- Prevailing wisdom / pain points: these methods needed either (a) specialized architectures that constrain receptive fields (DIM/AMDIM, CPC's patching + PixelCNN context net), or (b) a memory bank / momentum queue to supply many negatives (InstDisc, MoCo, PIRL, CMC). Lots of moving parts. Also a popular but contested framing: success = maximizing mutual information (tschannen2019 questioned whether MI or the specific loss form is what matters).
- Linear evaluation protocol (train linear classifier on frozen features) was the standard proxy for representation quality.

## Load-bearing ancestors

### 1. Becker & Hinton 1992 — "self-organizing net that discovers surfaces"
Origin idea: make representations of two views of the same input AGREE (maximize mutual information between spatially adjacent patches' outputs). SimCLR explicitly traces its lineage here ("dates back to becker1992"). Gap: old, small-scale, no deep nets / augmentation / contrastive softmax machinery.

### 2. Hadsell, Chopra, LeCun 2006 — contrastive loss / DrLIM
Learn invariant mapping by pulling positive pairs together and pushing negatives apart with a margin (spring model). Foundational "contrast positives vs negatives" formulation. Gap: pairwise margin loss, no temperature/softmax-over-many-negatives, no notion of in-batch negatives at scale.

### 3. Dosovitskiy et al. 2014 — Exemplar CNN / instance discrimination (parametric)
Treat each image (its augmentations) as its own class; classify which exemplar. Parametric: one weight vector per instance → does not scale to millions, weights don't generalize to new instances.

### 4. Wu et al. 2018 — InstDisc (non-parametric instance discrimination) [PDF in refs]
Key move: replace class weight vector w_j in softmax with the L2-normalized FEATURE v_j itself → non-parametric softmax
  P(i|v) = exp(v_i^T v / τ) / Σ_j exp(v_j^T v / τ),  ||v||=1.
- Introduces temperature τ and L2 normalization explicitly (the exact ingredients SimCLR keeps).
- Computing the full denominator over n classes is prohibitive → use NCE to approximate, and a MEMORY BANK V={v_j} storing every instance's feature, updated v_i→V each step; Z estimated by Monte Carlo.
Gap SimCLR reacts to: the memory bank stores STALE features (encoder has moved on since they were written), and adds machinery. SimCLR's claim: with a big enough batch you get fresh negatives for free, no bank.

### 5. Oord et al. 2018 — CPC / InfoNCE [PDF in refs]
The crucial loss ancestor. Predict future latent from context; pick the true future among N samples (1 positive from p(x_{t+k}|c_t), N-1 negatives from proposal p(x_{t+k})):
  L_N = -E[ log ( f_k(x_{t+k},c_t) / Σ_{x_j∈X} f_k(x_j,c_t) ) ].
This is exactly categorical cross-entropy of classifying the positive correctly. Optimizing it makes f_k ∝ density ratio p(x_{t+k}|c_t)/p(x_{t+k}); minimizing L_N maximizes a lower bound on I(x_{t+k};c_t): I ≥ log(N) - L_N (tighter as N grows → motivation for many negatives). SimCLR's NT-Xent IS this loss, specialized to: positive = other augmentation of same image, similarity = cosine (L2-normalized dot)/τ, negatives = the rest of the batch, symmetric over both views.
Gap: CPC ties the loss to an autoregressive setup — patch the image deterministically, run a PixelCNN context aggregator, encoder only sees small patches. Architecture-heavy.

### 6. Sohn 2016 — N-pair loss (multi-class N-pair) [NeurIPS, no arXiv]
Generalizes triplet loss: instead of 1 negative, push against N-1 negatives at once with a softmax/log-sum-exp form. Directly the "use all the other in-batch samples as negatives" idea SimCLR uses; SimCLR cites it (and chen2017sampling) for treating the other 2(N-1) examples as negatives. Gap: framed for deep metric learning / retrieval, not a full SSL framework; still often paired with negative mining.

### 7. Schroff et al. 2015 — FaceNet / triplet loss + semi-hard mining
Margin triplet: max(0, d(a,p) - d(a,n) + m). Needs semi-hard negative mining to work (pick negatives within the margin but farther than the positive). SimCLR uses it as a baseline (Table tab:loss) to argue: triplet/logistic don't weight negatives by hardness, so they NEED mining; NT-Xent's softmax automatically up-weights hard negatives via the gradient.

### 8. He et al. 2019 — MoCo (momentum contrast) [PDF in refs]
Decouple #negatives from batch size: keep a QUEUE of keys from previous minibatches (enqueue current, dequeue oldest). Problem: can't backprop through the whole queue, and naively copying the encoder makes keys inconsistent across steps → MOMENTUM update of the key encoder:
  θ_k ← m·θ_k + (1-m)·θ_q,  m=0.999.
Slowly-moving key encoder keeps queued keys consistent. SimCLR's counter-position: just use a large batch (up to 8192 → 16382 negatives) + global BN, end-to-end, no queue, no momentum encoder. Negatives are always fresh and from the current encoder.

### 9. Bachman et al. 2019 (AMDIM) / Hjelm 2018 (DIM)
Maximize MI across views / global-to-local. Achieve the view-prediction task by constraining receptive fields in the architecture (many 1x1 convs), use a tanh+regularized critic, FastAutoAugment. SimCLR's reaction: decouple the predictive task from the architecture by doing random cropping (which subsumes global-to-local and adjacent-view prediction) so a standard powerful ResNet can be used; replace tanh-clip critic with normalized+temperature-scaled loss. NOTE: AMDIM is also where the nonlinear-projection idea is hinted (SimCLR: nonlinear head "similar to bachman2019").

### 10. Supporting machinery the framework stands on
- ResNet (he2016) base encoder; BatchNorm (ioffe2015) → global BN fix for the in-batch-positives leakage.
- LARS optimizer (you2017) for large-batch stability; linear LR warmup + cosine decay (loshchilov2016); large-minibatch SGD tricks (goyal2017).
- Inception-style random crop (szegedy2015), color augmentation (howard2013, szegedy2015), cutout (devries2017), AutoAugment (cubuk2019) — augmentation toolbox.
- Word2Vec (mikolov2013) logistic/NCE loss as the "NT-Logistic" comparison point.

## The three SimCLR findings (for orientation; results belong only to answer.md/paper, NOT reasoning/context)
1. Composition of augmentations (esp. random crop + color distortion) defines the predictive task and is crucial; color histograms otherwise leak a shortcut.
2. Nonlinear projection head g() before the loss; keep h (before g) for downstream because g is trained to be invariant and throws away color/orientation info.
3. NT-Xent with L2 norm + temperature; benefits from large batch (more negatives) and longer training.
</content>
</invoke>
