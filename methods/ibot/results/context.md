# Context

## Research question

Masked language modeling has made Transformer pretraining in NLP both scalable and simple: randomly mask a fraction of the input tokens, ask the model to reconstruct them from context, and the resulting representations transfer everywhere. The crux of that recipe is a *tokenizer* — WordPiece — that first splits text into semantically meaningful pieces, so that "predict the masked token" is a well-posed classification problem over a vocabulary whose entries already carry meaning.

The question is whether the same paradigm can train Vision Transformers. The analog is masked image modeling (MIM): mask a fraction of the image patches and reconstruct them. But there is no obvious visual vocabulary. The crux of MIM is therefore a *visual tokenizer* that turns a (masked) patch into a supervisory target. A solution has to settle two coupled difficulties:

1. **Where do semantic visual targets come from?** Lingual tokens inherit semantics for free from word-frequency statistics. Image patches do not — pixels are continuous and a patch is semantically ambiguous. Empirically, semantic structure in images only emerges by *bootstrapping*: training a network to make two distorted views of the same image agree. So a semantically meaningful visual tokenizer seems to require its own representation-learning run.

2. **Can the tokenizer and the target model be trained together?** If the tokenizer must be trained first and then frozen, MIM becomes a multi-stage pipeline with an extra dataset and a fixed off-the-shelf tokenizer. But acquiring visual semantics is the *same* goal for both the tokenizer and the model being trained — which hints that a single-stage, jointly optimized scheme should be possible.

A good method would give a BERT-style local-token objective for ViTs, with a tokenizer that is semantically meaningful, jointly learnable in one stage, and adaptable to whatever data is at hand.

## Background

**Masked language modeling.** BERT (Devlin et al. 2019) masks ~15% of WordPiece tokens and predicts them from the bidirectional context with a cross-entropy classification loss over the vocabulary. This scales to large models and corpora and became the default for language. The pretext task is well-posed precisely because the tokenizer produces a finite vocabulary of meaningful units.

**Bootstrapping visual semantics from views.** A family of self-supervised methods learns image representations by enforcing invariance across two augmented views of the same image while preventing collapse. Contrastive variants (instance discrimination, MoCo, SimCLR) use negatives; BYOL and SimSiam use an asymmetric predictor and a momentum/stop-gradient target; SwAV and DINO enforce that the per-image output distribution over a set of prototypes be simultaneously sharp (confident) and uniform across the batch. The recurring empirical fact is that *high-level* semantics in images do not come from raw pixels — they emerge progressively through this view-agreement bootstrapping.

**Masked prediction in images.** Predicting masked image content has been attempted by directly regressing raw pixels (inpainting, masked-patch prediction). This is observed to waste capacity on high-frequency detail and to yield weak semantic representations under frozen-feature evaluation. The reframing that helped was to predict *discrete tokens* instead of pixels, turning reconstruction into classification and removing the pressure to model every high-frequency detail.

**Knowledge distillation.** A student matches a teacher's softened output distribution by minimizing the cross-entropy between the two softmax distributions. Both the discrete-token MIM loss and the view-agreement loss above can be read as distillation — the only difference is the source of the teacher distribution.

**Empirical observations about existing tokenizers (diagnostic).** When a frozen discrete VAE is used as the MIM tokenizer and one evaluates the *quality of its patch tokens* directly, the tokens carry mostly low-level texture and little high-level semantics (a few percent k-NN accuracy when its tokens are treated as features). In contrast, a network trained by view-agreement bootstrapping produces patch features that are far more semantic. And a pure masked-patch objective with no view-agreement component yields representations that are nearly useless under frozen-feature evaluation — semantics barely emerge from masking alone. These observations frame the problem: the *target* in MIM must itself be semantically meaningful, and that meaning has to come from somewhere other than the raw patches.

## Baselines

**BEiT (Bao et al. 2021).** MIM with an offline tokenizer. A discrete VAE from DALL-E is pretrained on a large external dataset, then frozen; it maps each of the 196 patches of a 224×224 image (16×16 patches, 14×14 grid) to one of K = 8192 discrete visual-token ids. BEiT applies blockwise masking (~40% of patches, contiguous blocks with a minimum block size and random aspect ratio), feeds the corrupted image to a ViT, and predicts the masked patches' visual-token ids with a softmax cross-entropy:
`max Σ_x E_M [ Σ_{i∈M} log p(z_i | x^M) ]`,
equivalently a one-hot distillation `-Σ_i m_i · P_φ(x_i)^T log P_θ(x̂_i)`, with φ the frozen dVAE and P_φ(x_i) a one-hot over K classes. Gaps: the dVAE tokens capture only low-level detail; the tokenizer is offline, with a fixed architecture and an extra dataset, so it is not adaptable to new domains; and one-hot discretization is a poor fit for semantically ambiguous patches.

**DINO (Caron et al. 2021).** Self-distillation with no labels, operating on the *global* image. Two augmented views u, v of an image pass through a student and a teacher that share architecture: a backbone f (ViT) plus a projection head h, emitting a K-dimensional distribution from the `[CLS]` token. The loss is the cross-entropy between teacher and student distributions, symmetrized over the two views:
`L_[CLS] = -P_θ'^[CLS](v)^T log P_θ^[CLS](u)`.
The teacher parameters θ' are an exponential moving average of the student θ (momentum λ on a cosine schedule from 0.996 to 1). Collapse is prevented by two opposing operations applied to the teacher output: *centering* — subtract an EMA of the teacher's batch-mean output, `c ← m·c + (1−m)·mean(teacher out)` — which stops any single dimension from dominating but pushes toward uniform; and *sharpening* — a low teacher temperature τ_t — which pushes the opposite way. Student temperature τ_s = 0.1; teacher temperature warmed 0.04 → 0.07. The head is a 3-layer MLP (hidden 2048, GELU) with an ℓ₂-normalized bottleneck (dim 256) and a weight-normalized final layer to K = 65536, batch-norm-free. Gap as a tokenizer: DINO only models the global `[CLS]` token; it never produces per-patch targets, so it does not give a BERT-style local objective.

**Hinton-style knowledge distillation (2015).** Train a student to reproduce a teacher's class-probability distribution by minimizing cross-entropy between softened softmaxes. Provides the template that both BEiT's and DINO's losses instantiate; what is left open is making the teacher itself a *learnable, online* source of per-patch targets.

**Multi-crop augmentation (SwAV/DINO).** Sample 2 global crops (224²) and several local crops (96²); local crops go only to the student. Consistently improves view-agreement methods. It is a known training ingredient that any new global-plus-local method would want to inherit.

## Evaluation settings

- **Pretraining data:** ImageNet-1K (1.28M images, 1000 classes) and the larger ImageNet-22K.
- **Frozen-feature evaluation:** k-nearest-neighbor classification and linear probing on the frozen backbone features (sweeping k and the linear learning rate respectively), following the DINO protocol.
- **Fine-tuning:** end-to-end fine-tuning on ImageNet-1K (BEiT recipe: AdamW, layer-wise learning-rate decay), reporting top-1 accuracy.
- **Semi-supervised:** unsupervised-pretrain then supervised-fine-tune with 1% and 10% of labels.
- **Unsupervised classification:** clustering metrics — accuracy (ACC), adjusted Rand index (ARI), normalized mutual information (NMI), Fowlkes–Mallows index (FMI).
- **Dense downstream:** COCO object detection and instance segmentation with Cascade Mask R-CNN (AP^b, AP^m); ADE20K semantic segmentation with a linear head and with UPerNet (mIoU). Multi-scale training for detection.
- **Transfer:** fine-tune on CIFAR-10/100, iNaturalist 2018/2019, Flowers, Cars.
- **Architectures:** ViT-S/16, ViT-B/16, ViT-L/16, Swin-T; 224 input, patch size 16 → 196 patch tokens.

## Code framework

The primitives that already exist: a ViT/Swin backbone that emits a `[CLS]` token and per-patch tokens, an MLP projection head, a momentum (EMA) teacher built by copying-then-EMA-updating the student, multi-crop data augmentation, AdamW with a cosine learning-rate/weight-decay schedule, and the centering+sharpening cross-entropy used for view-agreement on the `[CLS]` token. The slots below are what a local-token masked objective would have to fill in.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- backbone: emits [CLS] token and patch tokens; supports masking ---
class Backbone(nn.Module):                  # ViT / Swin (exists)
    def __init__(self, embed_dim, ...):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))  # learnable [MASK]
        # ... patch embed, pos embed, cls token, transformer blocks ...

    def apply_mask(self, x, mask):
        # TODO: replace embeddings at masked positions with mask_token
        pass

    def forward(self, x, mask=None, return_all_tokens=False):
        # TODO: tokenize -> (optionally) apply_mask -> transformer
        #       return cls token and (optionally) all patch tokens
        pass

# --- projection head: MLP + L2-bottleneck + weight-normed last layer ---
class ProjectionHead(nn.Module):            # DINO-style head (exists)
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # ... MLP -> L2 normalize -> weight_norm Linear to out_dim ...

    def forward(self, x):
        # TODO: the head we will design must produce a target for BOTH the
        #       global token and the per-patch tokens (currently only one path)
        pass

# --- masking: how to choose which patches to hide ---
def sample_mask(num_patches):
    # TODO: choose masked positions (the masking strategy is a design choice)
    pass

# --- the self-supervised objective ---
class Loss(nn.Module):
    def __init__(self, out_dim, ...):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))    # [CLS] center
        # TODO: any additional state the patch-level target needs

    def H(self, s, t, center, tau_s, tau_t):
        # known: centering+sharpening cross-entropy for the teacher target
        t = t.detach()
        s = F.softmax(s / tau_s, dim=-1)
        t = F.softmax((t - center) / tau_t, dim=-1)
        return -(t * torch.log(s)).sum(dim=-1)

    def forward(self, student_out, teacher_out, mask, epoch):
        # known: global view-agreement term on the [CLS] token
        # TODO: the local masked-token term -- what is the teacher target
        #       for a masked patch, and how is it produced?
        pass

# --- training step ---
def train_step(images, masks, student, teacher, head_s, head_t, loss_fn, opt, m):
    # TODO: feed masked view(s) to student, clean view(s) to teacher,
    #       compute loss, backprop into student, EMA-update teacher
    pass
```
