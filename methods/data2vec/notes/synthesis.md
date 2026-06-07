# data2vec synthesis (grounding notes)

Verified arXiv: 2202.03555 "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language" (ICML 2022), Baevski, Hsu, Xu, Babu, Gu, Auli (Meta AI). Canonical code: fairseq examples/data2vec (data2vec_audio.py is the cleanest reference for target construction + loss; data2vec_vision.py for the image variant).

## Pain point / research question
- SSL idea is the same across modalities but the *algorithms and objectives* differ widely because each was built for one modality. Want one identical learning objective for speech, NLP, vision.
- Each modality has a modality-specific target trick: NLP predicts words/sub-words; speech needs to *learn* a vocabulary of discrete speech units (wav2vec 2.0 quantization, HuBERT iterative clustering); vision learns discrete visual tokens (BEiT/DALL-E dVAE) or regresses pixels (MAE) or learns augmentation-invariance (SimCLR/BYOL/DINO). All these targets are (a) modality-specific and (b) *local* — a word, a patch's token, a speech unit — carrying info isolated to that position.
- Goal: a single objective. Insight: predict the *latent representation* the model itself produces, of the *full unmasked input*, from a *masked view* — in a self-distillation setup. The targets are continuous and contextualized (built by self-attention over the whole input), not local discrete tokens.

## Load-bearing ancestors (grounded)
- BERT (Devlin 2018): masked prediction of discrete sub-word tokens from context. Template; but target is a fixed closed vocabulary, one embedding per token regardless of context.
- wav2vec 2.0 (Baevski 2020): masks spans of latent speech reps, contrastive loss with quantized targets (learned discrete units via product quantization + Gumbel). Speech masking strategy borrowed: sample p=0.065 start indices, mask 10 steps → ~49% masked. Gap: needs quantization / discrete units.
- HuBERT (Hsu 2020): iterative — cluster layer reps into discrete units (k-means), predict them, re-cluster. Gap: discrete units, multi-stage, fixed vocabulary.
- BEiT (Bao 2021): MIM with offline dVAE visual tokens; blockwise masking ≥16 patches/block, random aspect ratio. data2vec borrows the blockwise masking strategy (but masks 60% not 40%). Gap: discrete tokens, offline tokenizer.
- MAE (He 2021): regress raw masked pixels (normalized per-patch). Local target, pixel-level.
- BYOL (Grill 2020) / DINO (Caron 2021): regress the latent rep of a momentum (EMA) teacher; teacher = EMA of student; BYOL uses predictor+stop-grad, DINO uses centering+sharpening. Crucially: they predict the *top* layer only and feed *two augmented views* (teacher and student see different augmentations) — the task is augmentation-invariance, NOT within-sample structure. data2vec generalizes: predict an *average of multiple top layers*, feed teacher the *unmasked same input* and student the *masked same input* — the task becomes "fill in masked content," capturing structure within a sample.
- Mean Teacher (Tarvainen 2017): EMA teacher for semi-supervised consistency. Source of the EMA teacher idea.

## The method (grounded; Method section + code)
Single Transformer used in two modes:
- **Student mode**: encode a *masked* version of the input (mask tokens replaced by a learned MASK embedding). Modality-specific feature encoder + masking (ViT patches + BEiT blockwise mask for vision; conv encoder + span mask for speech; embedding + BERT mask for text).
- **Teacher mode**: encode the *unmasked* full input with the SAME model parameterized by EMA: `Δ ← τΔ + (1−τ)θ`. τ linearly annealed τ0→τe over τn updates, then constant. Feature encoder + positional encoder shared between teacher and student (more efficient, slightly better).
- **Targets**: from the top K blocks of the teacher, at masked positions only. Output of block l at step t = a_t^l. Normalize each block → â_t^l, then average: `y_t = (1/K) Σ_{l=L-K+1}^L â_t^l`. Use the FFN output *prior to the last residual connection* in each block as the block feature (ablation: FFN works best, self-attention output unusable because it's biased toward other timesteps before the residual). Normalization (parameter-less LayerNorm for vision/NLP; instance norm over sequence for speech) prevents collapse to constant + stops high-norm layers dominating.
- **Objective**: Smooth L1 (Huber) regression of y_t by the student prediction f_t(x), only at masked positions:
  L = 0.5(y−f)²/β if |y−f|≤β else |y−f| − 0.5β. β controls squared→L1 transition; less outlier-sensitive but needs tuning. (Speech: simple L2 worked. Loss-ablation: L1/L2/SmoothL1 all close.)
- Predict only masked time-steps. Targets are contextualized (teacher sees full input via self-attention); ablation: bigger teacher context → better; restricting teacher attention hurts.

## Design-decision → why (grounded)
- Predict latent reps not discrete tokens: targets not predefined, not limited in number, adapt to the example ("open vocabulary"); contextualized (BERT learns one embedding per token for all contexts; data2vec's target for a token depends on its sentence). Simplifies away quantization/dVAE/clustering.
- Teacher sees unmasked input (vs DINO's different augmentation): want within-sample structure (fill-in-the-blank), not augmentation invariance; masking the teacher hurt in prelim experiments → teacher gets full context.
- Average top-K layers (vs BYOL's top layer K=1): different layers extract different features; averaging enriches the task; ablation K=1..12, multi-layer > top-only for all 3 modalities; all-layers nearly as good as tuned K. Inspired by wav2vec2's top layers being worse than middle for downstream.
- FFN output before last residual as block feature: self-attention output alone is biased toward other timesteps (it's before the residual that re-injects the position's own feature); FFN includes pre-attention features → usable.
- Normalize per-block then optionally normalize the average: prevents representation collapse (constant target) and prevents high-norm layers dominating. Speech uses instance norm (adjacent frames highly correlated); vision/NLP parameterless LayerNorm.
- EMA teacher with annealed τ (slow→slower): teacher updated more at start (model random), less later (good params); τ too low → student collapse propagates to teacher.
- Smooth L1: robust to outliers vs L2; β tuned per modality (vision β=2, NLP β=4, speech L2/β small).
- Mask 60% for vision (vs BEiT 40%): higher rate slightly better for images (images less semantic per token than text, need harder task). Text 15% (BERT) or spans-of-4 at 0.35. Speech p=0.065 start, len 10 → ~49%.
- Loss scale 1/sqrt(d): normalize by sqrt of feature dim (code).

## Hyperparameters (grounded, vision focus)
ViT-B: L=12 blocks, H=768, FFN 4H. 224 image, 16×16 patches → 196 tokens. Mask 60% blockwise (≥16 patches/block, random aspect). Augment: random resized crop, h-flip, color jitter; SAME modified image for teacher and student. Pretrain 800 epochs, batch 2048, Adam, cosine schedule, warmup 40 epochs to lr 0.002. β=2, K=6, τ=0.9998 constant. Stochastic depth 0.2. EMA in fp32. ViT-L: H=1024, L=24, batch 8192, 1600 epochs (reset schedule + teacher at 800; τ 0.9998 then 0.9999).
Speech ablation hyperparams: τ0=0.999, τe=0.9999, τn=30000, K=8. NLP: τ0=0.999, τe=0.9999, τn=100000, K=10, β=4.

## Evaluation settings (pre-method, no outcomes)
Vision: pretrain ImageNet-1K, fine-tune ImageNet-1K classification, mean-pool last block → softmax classifier, top-1 accuracy. Speech: pretrain Librispeech 960h, fine-tune ASR with 10min–960h labeled (wav2vec2 regime), WER. NLP: pretrain BooksCorpus+Wikipedia (BERT setup) 1M updates batch 256, fine-tune GLUE (MNLI, QNLI, RTE, MRPC, QQP, STS-B, CoLA, SST-2), avg dev accuracy over 5 runs. Architectures: ViT-B/ViT-L (vision); Transformer Base/Large.
