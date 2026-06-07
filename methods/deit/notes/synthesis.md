# DeiT synthesis (grounded in 2012.12877 src + ViT + Hinton distillation + facebookresearch/deit)

## Pain point (~late 2020)
- ViT (Dosovitskiy et al.) showed a pure transformer (raw 16x16 image patches as tokens) classifies images excellently — BUT only when pre-trained on a huge private dataset (JFT-300M, 300M images), with extensive compute. ViT paper concluded transformers "do not generalize well when trained on insufficient amounts of data."
- Convnets win partly because they bake in priors (locality, translation equivariance) and partly because of large training sets (ImageNet). Transformers have weaker priors → need more data.
- Question: can a convolution-free transformer be trained to convnet-competitive ImageNet accuracy using ImageNet-1k ONLY (no external data), on a single node in a few days?
- Second question: how to distill transformers?

## ViT recap (the architecture being kept)
- MSA: Attention(Q,K,V) = Softmax(QK^T/√d) V. Self-attention: Q=XW_Q, K=XW_K, V=XW_V, k=N (attention among all input vectors). Multi-head: h heads, each N×d, concat → N×dh, reproject → N×D.
- Transformer block: MSA + FFN. FFN = two linear layers w/ GeLU between; expand D→4D then 4D→D. Both MSA & FFN are residual (skip connections) w/ LayerNorm.
- Image → N patches of 16×16 px (N=14×14=196 for 224²). Each patch linearly projected (3×16×16=768 → D).
- Position: fixed or trainable positional embeddings added to patch tokens before first block.
- Class token: trainable vector appended to patch tokens, goes through layers, projected by linear → class. From NLP/BERT, replaces pooling. Net processes N+1 tokens; only class vector used for output. Forces self-attention to spread info between patches & class token.
- Models: DeiT-B = ViT-B (same arch, D=768, 12 layers, 12 heads, 86M params); DeiT-S (≈ResNet-50 counterpart); DeiT-Ti (≈ResNet-18).

## Data-efficient training recipe (DeiT-B defaults; the "data-efficient" contribution)
- Optimizer: AdamW. lr = 0.0005 × batchsize/512 (scale w/ bs, base 512 not 256 like Goyal). Batch size 1024. Cosine decay. Weight decay 0.05 (ViT used 0.3 — too high, hurts convergence here). Warmup 5 epochs. 300 epochs.
- Init: truncated normal (Hanin & Rolnick) — transformers sensitive to init, some options didn't converge.
- Data augmentation (transformers need a LOT): RandAugment (9/0.5), Mixup (prob 0.8), CutMix (prob 1.0), Random Erasing (prob 0.25), AutoAugment (ablated, RandAugment chosen over it). Almost all augs help.
- Repeated Augmentation (Berman/Hoffer): KEY ingredient, significant boost. 3 repetitions → see only 1/3 of images per epoch.
- Regularization: stochastic depth 0.1 (helps deep transformer convergence). Label smoothing ε=0.1. NO dropout (excluded — hurts). Gradient clip: ViT used it, DeiT doesn't.
- EMA: small gain (0.1) that vanishes after fine-tuning.
- Ablation findings: SGD pretraining → 74.5 vs AdamW 81.8 (AdamW critical). Removing Mixup+CutMix both → 75.8 (big drop). Removing repeated aug → 76.5. Removing erasing/stoch-depth → didn't train well.
- No batch-norm (LayerNorm only) → can reduce batch size without hurting → easier to train large models.

## Train low / fine-tune high resolution (FixRes, Touvron 2019/2020)
- Train at 224², fine-tune at 384² (or other). Speeds training, improves accuracy under augmentation.
- Patch size kept fixed → N (# patches) changes at higher res. Model/classifier unchanged (transformer handles variable token count); but positional embeddings (N of them) must be adapted → INTERPOLATE.
- Interpolation: bilinear reduces ℓ2-norm of a vector vs neighbors → low-norm vectors hurt pretrained transformer (big accuracy drop if used directly). → use BICUBIC interpolation (approximately preserves norm), then fine-tune.
- Fine-tune: schedule/reg/optim like FixEfficientNet but KEEP training-time data aug (not dampened). AdamW or SGD (similar). 25 epochs, ~20h on 8 GPU → "DeiT⚗384".

## Distillation (the second contribution)
### Soft distillation (Hinton 2015)
L_global = (1−λ) L_CE(ψ(Z_s), y) + λ τ² KL(ψ(Z_s/τ), ψ(Z_t/τ))
- Z_t, Z_s = teacher/student logits, τ = temperature, ψ = softmax, λ balances. τ²: gradient-scale correction so soft-target gradients comparable to hard-target as τ varies.
- DeiT soft-distill params: τ=3.0, λ=0.1 (following Cho et al.).

### Hard-label distillation (DeiT's variant)
L_global^hardDistill = ½ L_CE(ψ(Z_s), y) + ½ L_CE(ψ(Z_s), y_t)
- y_t = argmax_c Z_t(c) = teacher's hard decision, treated as a true label.
- Parameter-free (no τ, λ to tune), conceptually simpler. y_t plays same role as y.
- Hard label can vary w/ data augmentation (teacher re-evaluated per augmented crop).
- Can convert hard→soft w/ label smoothing ε=0.1.
- Better than traditional soft distillation in their setting.

### Distillation token (the core idea)
- Add a NEW token (distillation token) to initial embeddings alongside patch tokens + class token.
- Used like class token: interacts via self-attention, output after last layer. But its target = teacher's (hard) label, not the true label.
- Class token target = true label; distillation token target = teacher label. Both learned by backprop.
- Observation: learned class & distill tokens converge to DIFFERENT vectors (avg cosine sim 0.06 at input; grow similar through layers, 0.93 at last layer but <1). They aim at similar-but-not-identical targets.
- Control: replacing distill token with a SECOND class token (same target as class token) → the two converge to cos=0.999, identical outputs, adds nothing. So the distillation token's distinct target is what makes it useful.
- Significant improvement over vanilla distillation.

### Joint classifiers (test time)
- Both class & distillation embeddings get linear classifiers, each can predict.
- Referent method: LATE FUSION — add the softmax outputs of the two classifiers. (3 options evaluated: class only, distill only, fusion.)

### Fine-tuning with distillation
- Use both true label & teacher prediction during higher-res fine-tuning. Teacher at same target resolution (from low-res teacher via FixRes). True-labels-only reduces teacher benefit → lower perf.

## Key finding (in-frame, drives design)
- Transformers learn MORE from a CONVNET teacher than from another transformer of comparable performance. → use a convnet (e.g. RegNetY) teacher. Intuition: student inherits the convnet's inductive bias through distillation.

## Canonical implementation (facebookresearch/deit)
- DistilledVisionTransformer: adds dist_token (nn.Parameter) + pos_embed sized num_patches+2; head_dist = Linear. forward_features returns (x[:,0], x[:,1]) = (class, dist) outputs; forward returns (x, x_dist); at inference returns (x + x_dist)/2.
- losses.py DistillationLoss: outputs, outputs_kd = outputs; base_loss = base_criterion(outputs, labels); soft: F.kl_div(log_softmax(outputs_kd/T), log_softmax(teacher/T), reduction='sum', log_target=True)*(T*T)/numel; hard: F.cross_entropy(outputs_kd, teacher.argmax(1)); loss = base_loss*(1-alpha) + distillation_loss*alpha.
- timm-based ViT backbone, patch_embed Conv2d(3, D, kernel=16, stride=16), blocks (Attention + Mlp w/ GELU), LayerNorm.

## Design-decision → why table
- Pure transformer, ImageNet-only: prove no convolution & no external data needed; weaker priors compensated by strong augmentation/regularization.
- AdamW not SGD: transformers train poorly w/ SGD (74.5 vs 81.8).
- wd 0.05 not ViT's 0.3: ViT's high wd hurts convergence in single-dataset regime.
- Heavy augmentation (RandAug, Mixup, CutMix, Erasing): transformers lack conv priors → need data; almost all augs help.
- Repeated augmentation: key boost (76.5→81.8 when added).
- No dropout: hurts; stochastic depth 0.1 instead (helps deep transformer convergence).
- Truncated-normal init: transformers init-sensitive, some inits diverge.
- Train 224 / fine-tune 384 (FixRes): faster training + higher accuracy.
- Bicubic (not bilinear) positional-embedding interpolation: bilinear shrinks ℓ2 norm → low-norm vectors break pretrained transformer; bicubic preserves norm.
- Hard-label distillation: parameter-free, simpler, beats soft in this setting.
- τ² scaling in soft distillation: keeps soft-target gradient magnitude comparable across temperatures.
- Distillation token (separate from class token): a second class token (same target) collapses into the class token (cos 0.999, useless); a token with the TEACHER's target stays distinct (cos 0.06) and adds info.
- Convnet teacher over transformer teacher: student learns more (inherits conv inductive bias via distillation).
- Late-fusion of class+distill heads at test: combine both predictions.
```
