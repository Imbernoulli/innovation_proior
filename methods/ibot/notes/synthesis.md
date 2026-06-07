# iBOT synthesis (grounding notes)

Verified arXiv: 2111.07832 "iBOT: Image BERT Pre-Training with Online Tokenizer" (ICLR 2022). Canonical code: github.com/bytedance/ibot (main_ibot.py iBOTLoss; models/head.py iBOTHead/DINOHead; models/vision_transformer.py mask_model; loader.py ImageFolderMask blockwise masking).

## Pain point / research question
- MLM (BERT) made language pretraining scalable; its crux is a tokenizer that splits text into semantically meaningful pieces (WordPiece). For ViT, MLM's analog (MIM) is underexplored. Most SSL in vision (MoCo v3, DINO) works on *global views* and neglects internal image structure; we want a BERT-style *local-token* objective for ViTs.
- Crux of MIM = the visual tokenizer that turns masked patches into supervisory targets. Two challenges:
  (a) Semantics: lingual tokens carry semantics from word-frequency statistics; visual semantics don't fall out of pixels because images are continuous. Empirically semantics emerges by bootstrapping online representations enforcing similarity of distorted views (MoCo/BYOL/SwAV/DINO). This *suggests* a multi-stage pipeline: first train a semantic tokenizer, then train the target model.
  (b) Multi-stage cost: but acquiring semantics is a *common* end for tokenizer and target model, so a single-stage jointly-optimized pipeline should exist.

## Load-bearing ancestors (grounded)
- BERT/MLM (Devlin 2019): mask tokens, reconstruct from context; needs a tokenizer (WordPiece). De-facto NLP standard. The template iBOT ports to vision.
- BEiT (Bao et al. 2021, 2106.08254): MIM with offline dVAE tokenizer from DALL-E, vocabulary K=8192 visual tokens; 224 image → 14×14=196 patches; blockwise masking ~40% (min 16 patches/block, random aspect ratio); loss Eq.(1) in iBOT: `-Σ m_i P_φ(x_i)^T log P_θ(x̂_i)` where P_φ is the frozen dVAE one-hot over K classes. iBOT explicitly identifies this as knowledge distillation: knowledge distilled from pre-fixed tokenizer φ to model θ. Gap: (i) dVAE captures only low-level detail (Tab. components: 6.9 kNN as tokenizer); (ii) offline, fixed architecture + extra dataset (DALL-E data), so not domain-adaptive; (iii) one-hot discretization is sub-optimal for ambiguous patches.
- DINO (Caron et al. 2021, 2104.14294): self-distillation, no labels. Two views u,v of one image; teacher (EMA of student) and student share backbone f + projection head h; both emit a K-dim distribution from the [CLS] token. Loss Eq.(2): `L_[CLS] = -P_θ'^[CLS](v)^T log P_θ^[CLS](u)`, cross-entropy, symmetrized. Teacher params θ' = EMA of student θ (momentum λ cosine 0.996→1). Collapse avoided by *centering* (c ← m·c + (1-m)·batch-mean of teacher output; prevents one dim dominating but pushes toward uniform) + *sharpening* (low teacher temp; opposite effect). τ_s=0.1, τ_t warmed 0.04→0.07 over 30 epochs. Head: 3-layer MLP hidden 2048, GELU, L2-normalized bottleneck dim 256, weight-normalized last layer, output K=65536, BN-free. Gap as MIM ancestor: operates only on global [CLS], no local/patch-token modeling.
- Knowledge distillation (Hinton 2015): student matches teacher's soft class distribution (cross-entropy of softened softmax). Both Eq.(1) and Eq.(2) are KD; the difference is *where the teacher comes from*: pre-fixed φ (BEiT) vs past iterations θ' (DINO).
- Multi-crop (SwAV/DINO): 2 global crops 224², 10 local 96²; local crops only go to student. iBOT inherits.

## The unifying observation (the aha)
Eq.(1) (BEiT MIM) and Eq.(2) (DINO [CLS] self-distillation) have the *same algebraic form*: a cross-entropy from a teacher distribution to a student distribution. Eq.(1)'s teacher is a *pre-fixed* φ; Eq.(2)'s teacher is the *online EMA* θ'. So: replace BEiT's frozen dVAE tokenizer φ by the online θ' teacher, applied to *patch tokens*. The teacher becomes a self-distilling **online tokenizer**. This resolves both challenges at once: semantics comes from the same bootstrapping that makes DINO's [CLS] meaningful; "online" means no extra pretraining stage and domain knowledge from the current data.

## iBOT method (grounded, Eq. 3 + pseudocode + code)
Two augmented views u, v of image x. Blockwise-mask each (BEiT style) → û, v̂.
- MIM loss Eq.(3): `L_MIM = -Σ_{i=1}^N m_i · P_θ'^patch(u_i)^T log P_θ^patch(û_i)`. Student sees masked view û and outputs patch-token distributions; teacher sees the *clean* view u and outputs patch-token distributions; recover the masked patches to the teacher's output. Symmetrized with the (v̂,v) term. In code: only the v==q (same view) pairs contribute the patch loss; masked positions selected by mask, normalized by mask.sum().clamp(min 1).
- [CLS] loss Eq.(2) (DINO), but student input is now the masked view: `L_[CLS] = -P_θ'^[CLS](v)^T log P_θ^[CLS](û)`, symmetrized; cross-view (v != q pairs in code).
- Teacher = backbone f_t + patch head h_t^patch is the **online tokenizer** for the masked patches.
- Shared projection head: `h^[CLS] = h^patch` for both student and teacher (Tab shows shared best; transfers [CLS] semantics to MIM). Code iBOTHead: shared_head reuses last_layer for patches (last_layer2 = last_layer).
- **Soft targets**: use teacher softmax distribution, NOT one-hot/hardmax (Tab continuous: softmax† with centering+sharpen gives 69.1 kNN vs hardmax). Patch ambiguity → soft is better.
- Centering+sharpening for both [CLS] and patch streams, with *separate* centers C (1×K) and C' (1×1×K) and *separate* temps. H(s,t,c,τ_s,τ_t): t=detach; s=softmax(s/τ_s); t=softmax((t-c)/τ_t); return -(t·log s).sum.

## Design-decision → why (all grounded)
- Online tokenizer (EMA teacher) instead of offline dVAE: removes stage-1; tokenizer co-evolves and acquires high-level semantics; domain-adaptive. (Tab: standalone DINO-as-tokenizer 44.3 kNN, dVAE 6.9, iBOT 70.3.)
- [CLS] self-distillation kept (not dropped): MIM alone gives 9.5 kNN / 29.8 lin — visual semantics barely emerge from MIM alone; [CLS] bootstrapping is what makes the online tokenizer semantic. So L_[CLS] is *load-bearing*, not auxiliary.
- Shared head: semantics learned on [CLS] transfers to patch MIM; empirically best vs vanilla/semi-shared.
- Soft (softmax) targets w/ sharpening: patches are semantically ambiguous, one-hot is sub-optimal; centering's role weaker than for [CLS].
- Blockwise masking (BEiT): masking contiguous blocks not random pixels.
- Sum losses without scaling (λ1=λ2=1): loss-ratio ablation, 1:1 best on linear.
- Output dim K=8192 (not 65536 like DINO): each patch has its own distribution → memory; larger K gives no gain. Default 8192.
- Random MIM with multi-crop: direct multi-crop is unstable (distribution mismatch of masked global vs non-masked local crops). Fix: with prob 0.5 ratio r=0 (no masking, pure DINO), with prob 0.5 r∈[0.1,0.5]. Stable, gain. (Pipeline (b) w/ random MIM: 71.5 kNN vs (b) 62.0.)
- Prediction ratio 0.3 (±0.2): performance flat over 0.05–0.4; adding variance acts as stronger augmentation.
- Centering/sharpening hyperparams for patch: m'=0.9, τ'_t warmed 0.04→0.07; chosen by ablation.

## Hyperparameters (grounded)
AdamW, batch 1024. lr = 5e-4 × batch/256, warmup 10 epochs, cosine. Weight decay 0.04→0.4 cosine (from DINO recipe; commented in tex). ViT-S/16 800ep, B/16 400ep, L/16 250ep, Swin-T 300ep. 224 image, patch 16 → N=196 tokens. Head 3-layer MLP, L2 bottleneck, K=8192. Multi-crop 2×224² + 10×96², local scale (0.05,0.32), global (0.32,1.0). Teacher EMA λ cosine 0.996→1 (DINO). student τ=0.1.

## Evaluation settings (pre-method, no outcomes)
Datasets: ImageNet-1K (and -22K) pretraining. Protocols (from DINO/BEiT): k-NN on frozen features, linear probing on frozen features, end-to-end fine-tuning. Downstream: COCO detection/instance-seg (Cascade Mask R-CNN), ADE20K semantic seg (linear head / UPerNet), transfer to CIFAR/iNaturalist/Flowers/Cars. Metrics: top-1, AP^b, AP^m, mIoU, kNN/linear acc; clustering ACC/ARI/NMI/FMI.
