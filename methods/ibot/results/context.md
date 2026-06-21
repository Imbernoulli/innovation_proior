# Context

## Research Question

Masked language modeling made Transformer pretraining simple: hide some input tokens, predict their identities from bidirectional context, and transfer the resulting encoder. Language already has a tokenizer. WordPiece turns text into a finite set of units, so "predict the missing token" is a classification problem over meaningful labels rather than a regression problem over raw characters.

Vision Transformers expose an analogous sequence structure. A 224x224 image with 16x16 patches becomes a sequence of 196 patch tokens plus a class token. The question is whether the same local-token pretraining idea can work for images: mask some patches, infer what belongs there, and use the learned encoder for recognition and dense prediction.

A BERT-like image objective needs a target for each masked patch. Raw pixels are continuous, local, and carry high-frequency detail; a 16x16 patch may contain texture, object part, background, or a mixture. One way to obtain discrete, abstract per-patch targets is a visual tokenizer that maps each patch to a target distribution tied to local image structure. The setting is how to define such per-patch targets for masked image modeling on a Vision Transformer.

## Background

**Masked language modeling.** BERT samples token positions after WordPiece tokenization, replaces selected positions with a masking/noising policy, and predicts the original vocabulary ids. The model only pays the masked-token loss on selected positions. This is a bidirectional pretraining objective because the Transformer can use both left and right context to infer each missing token.

**Vision Transformer patch tokens.** A ViT embeds non-overlapping image patches, prepends a `[CLS]` token, adds positional embeddings, and processes the sequence with self-attention. The `[CLS]` representation is typically used for image-level classification, while the patch tokens preserve local spatial information that can be useful for dense tasks.

**Masked image modeling.** A visual masked-token objective replaces selected patch embeddings with a learned `[MASK]` embedding and asks the encoder output at those positions to recover a target. If the target is raw pixels, the task is a regression toward low-level reconstruction. If the target is a discrete visual token, the task becomes a classification problem closer to MLM, with the target supplied by a visual tokenizer.

**Self-distillation in vision.** Methods such as BYOL and DINO train a student view to match a teacher view without labels. In DINO, teacher and student share architecture; the teacher is an exponential moving average of the student; the loss is cross-entropy between teacher and student softmax outputs from the `[CLS]` token. Collapse is controlled by centering the teacher logits with a moving average and sharpening them with a low teacher temperature. Multi-crop training sends global and local crops to the student, but only global crops to the teacher.

**Knowledge distillation.** A student can be trained against a teacher probability distribution rather than a hard label. The teacher distribution is turned into a target with softmax, often with a temperature. Cross-entropy from teacher distribution to student distribution transfers more information than a single hard class id when the target is ambiguous.

## Baselines

**Pixel or low-level reconstruction.** Early masked-patch objectives regress pixels or simple patch statistics. They are easy to define and target short-range texture and color.

**BEiT-style masked image modeling.** BEiT makes image masking closer to BERT by using an offline discrete VAE tokenizer. The original image is tokenized into a 14x14 grid of visual-token ids from an 8192-entry vocabulary. The corrupted patch sequence goes through a ViT, and the model predicts the original visual-token id at masked positions. BEiT uses blockwise masking, roughly 40 percent of patches, so the model cannot rely only on immediate neighboring pixels.

**DINO-style global self-distillation.** DINO uses two global crops and optional local crops. The student sees all crops; the EMA teacher sees global crops. For each teacher global crop, the student is trained to match its centered and sharpened `[CLS]` distribution on the other crops. This produces global image representations from the `[CLS]` token.

**Multi-crop augmentation.** SwAV and DINO use multiple views of different resolutions for self-supervised learning. The usual pattern is two high-resolution global crops and several lower-resolution local crops. A local crop has fewer patch tokens and less surrounding context than a global crop.

## Evaluation Settings

- **Pretraining data:** ImageNet-1K and, for larger-scale runs, ImageNet-22K.
- **Backbones:** ViT-S/16, ViT-B/16, ViT-L/16, and Swin variants; 224x224 inputs with 16x16 ViT patches give 196 patch tokens.
- **Frozen-feature evaluation:** k-nearest-neighbor classification and linear probing on frozen representations, following the DINO protocol.
- **Fine-tuning:** end-to-end ImageNet fine-tuning with ViT/BEiT-style optimizer settings, including layer-wise learning-rate decay where applicable.
- **Label-efficient and unsupervised evaluation:** fine-tuning with 1 percent and 10 percent of ImageNet labels, plus clustering metrics such as ACC, ARI, NMI, and FMI.
- **Dense transfer:** COCO object detection and instance segmentation with Cascade Mask R-CNN, and ADE20K semantic segmentation with a linear head or UPerNet.
- **General transfer and robustness:** transfer to smaller classification datasets and stress tests involving occlusion, background change, patch dropping, or corruptions.

## Code Framework

The available pieces are a ViT/Swin backbone that emits `[CLS]` and patch tokens, a DINO-style projection head, a student/teacher training loop with EMA teacher updates, multi-crop augmentation, blockwise patch masking, AdamW with cosine schedules, and a centered/sharpened cross-entropy used for self-distillation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, embed_dim, ...):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        # patch embedding, position embedding, class token, transformer blocks

    def apply_mask(self, patch_embeddings, mask):
        # replace selected patch embeddings with mask_token
        pass

    def forward(self, x, mask=None, return_all_tokens=False):
        # return either the class token or the full [CLS] + patch-token sequence
        pass


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # MLP -> l2-normalized bottleneck -> weight-normalized final layer

    def forward(self, tokens):
        # map class and/or patch tokens to logits over a target vocabulary
        pass


def sample_mask(num_patches, ratio):
    # choose which patch positions are hidden
    pass


class DistillationLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def cross_entropy(self, student_logits, teacher_logits, center, teacher_temp):
        teacher = F.softmax((teacher_logits.detach() - center) / teacher_temp, dim=-1)
        student = F.log_softmax(student_logits / self.student_temp, dim=-1)
        return -(teacher * student).sum(dim=-1)

    def forward(self, student_outputs, teacher_outputs, masks, epoch):
        # decide which tokens are matched, which view supplies the target,
        # and whether local-token and global-token losses share machinery
        pass


def train_step(images, masks, student, teacher, loss_fn, optimizer, ema_momentum):
    # run the selected student and teacher views, compute the self-supervised
    # loss, update the student by gradient descent, then update the teacher by EMA
    pass
```
