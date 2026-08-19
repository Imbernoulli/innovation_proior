# DeiT ladder — mining notes (working file, not part of the deliverable)

Sources: methods/deit/src/{training.tex, experiments.tex, token.tex, transformer.tex, introduction.tex,
related.tex}. All numbers below are grep-verified against these files (line numbers as of the current
checkout).

## Given starting point (goes into 00-initial-context.md, not its own rung)

- ViT-B/16 trained on ImageNet-1k only (original ViT recipe, wd 0.3, dropout, grad clip, no strong aug):
  77.91% top-1. Already-known engineering fixes (timm training pipeline) push the SAME architecture to
  79.35% top-1. `training.tex:65` (footnote): "The timm implementation already included a training
  procedure that improved the accuracy of ViT-B from 77.91% to 79.35% top-1, and trained on Imagenet-1k
  with a 8xV100 GPU machine."

## Rung 1 — data-efficient training recipe (AdamW + heavy augmentation/regularization stack), no distillation

Recipe: AdamW (lr 0.0005*bs/512, cosine, warmup 5ep), wd 0.05, truncated-normal init, batch 1024, 300
epochs; RandAugment(9,0.5), Mixup(0.8), CutMix(1.0), Random Erasing(0.25), Stochastic Depth(0.1),
Repeated Augmentation(3x), Label Smoothing(0.1), no dropout. Applied to three sizes: DeiT-Ti (5M, D=192,
h=3), DeiT-S (22M, D=384, h=6), DeiT-B (86M, D=768, h=12, = ViT-B architecture).

- `training.tex:32` ablation table, "none: DeiT-B" row: 81.8 (pretrained 224) / 83.1 (fine-tuned 384).
- `experiments.tex:224-226,228` throughput table: DeiT-Ti 72.2, DeiT-S 79.8, DeiT-B 81.8 (all @224);
  DeiT-B-up384 83.1.
- `experiments.tex:119` distillation table "no distillation" row: Ti 72.2, S 79.8, B 81.8, B-up384 83.1
  — consistent with the two tables above.
- Resolution companion sweep, same recipe/model, varying only fine-tune target size (bicubic
  interpolated positional embeddings): `training.tex:184-187` table:res_finetune — 160^2: 79.9, 224^2:
  81.8, 320^2: 82.7, 384^2: 83.1.
- `training.tex:37-38` ablation rows (optimizer): SGD pretraining -> 74.5/77.3 vs AdamW -> 81.8/83.1 —
  cited as background justification for AdamW, not its own rung.

## Rung 2 — distillation objective: soft (Hinton KD) vs hard-label, single class token

Fixed default teacher (established as background, not tested here): RegNetY-16GF convnet, 84M params,
82.9% top-1, trained with the same data/augmentation as DeiT. `experiments.tex:54` "This teacher reaches
82.9% top-1 accuracy on ImageNet." Convnet-over-transformer-teacher preference and stronger-teacher
tendency are background facts (table:teacher, `experiments.tex:70-75`), used only to justify locking in
this default, not tested as their own rung (that table's own header names the *two-token* student, so it
postdates rung 3's architecture and cannot be this rung's own feedback).

- `experiments.tex:120` soft distillation: Ti 72.2, S 79.8, B 81.8, B-up384 83.2 (no gain over rung 1 at
  224; a hair better at 384).
- `experiments.tex:121` hard-label distillation: Ti 74.3, S 80.9, B 83.0, B-up384 84.0 (clear gain at
  every size).
- `experiments.tex:84` prose: "Hard distillation significantly outperforms soft distillation for
  transformers, even when using only a class token: hard distillation reaches 83.0% at resolution
  224x224, compared to the soft distillation accuracy of 81.8%."
- `token.tex` gives both loss formulas (soft KL with tau^2 correction; hard = 1/2 CE(y) + 1/2 CE(y_t),
  y_t = argmax of teacher logits) and the augmentation-label-mismatch motivation for trying hard: "For a
  given image, the hard label associated with the teacher may change depending on the specific data
  augmentation."

## Rung 3 — architecture: dedicated distillation token + late fusion, hard-label objective fixed

- `experiments.tex:123` class embedding only: Ti 73.9, S 80.9, B 83.0, B-up384 84.2.
- `experiments.tex:124` distillation embedding only: Ti 74.6, S 81.1, B 83.1, B-up384 84.4.
- `experiments.tex:125` class+distillation (late-fused softmax sum, the referent readout): Ti 74.5, S
  81.2, B 83.4, B-up384 84.5.
- `token.tex`: distillation token added to initial embeddings alongside patch + class tokens, same
  self-attention treatment as class token, its own linear classifier trained against the teacher's hard
  label. Diagnostic: learned class/distillation token cosine similarity 0.06 at the input, rising to
  0.93 (still <1) at the last layer. Control: replacing the distillation token with a second class token
  (same target as the class token) collapses the two to cosine 0.999 (quasi-identical), "does not bring
  anything to the classification performance" — the distinct target is what matters, not the extra
  parameter count.

## Rung 4 — resolution fine-tune + extended schedule, on the winning class+distillation architecture

- Fine-tune @384, 300-epoch base schedule: already measured in rung 3 (`experiments.tex:125`,
  class+distillation B-up384 = 84.5); restated here as the input to the next lever.
- `experiments.tex:236-238` throughput table, 1000-epoch rows @224: DeiT-Ti-distil 76.6, DeiT-S-distil
  82.6, DeiT-B-distil 84.2.
- `experiments.tex:244` throughput table, 1000-epoch row @384: DeiT-B-distil-up384 = 85.2 (headline
  number).
- `experiments.tex:44,181` prose: "Our best model on ImageNet-1k is 85.2% top-1 accuracy outperforms the
  best ViT-B model pre-trained on JFT-300M at resolution 384 (84.15%) ... while being significantly
  faster to train." Also states the disjoint external-data regime SOTA (88.55%, ViT-H/JFT-300M/512,
  extra data) as a non-comparable reference point, not something this ladder targets.
- Qualitative motivation available before this rung (Figure caption + prose, `experiments.tex` "Number
  of epochs" paragraph): "With 300 epochs, our distilled network DeiT-B-distil is already better than
  DeiT-B. But while for the latter the performance saturates with longer schedules, our distilled
  network clearly benefits from a longer training time" — used as background for proposing the
  epoch extension specifically for the distilled model.

## Endpoint

DeiT-B-distil, fine-tuned at 384^2, 1000-epoch schedule: 85.2% top-1 ImageNet-1k, no external training
data, matches the paper's own published headline result. Final rung = published method.
