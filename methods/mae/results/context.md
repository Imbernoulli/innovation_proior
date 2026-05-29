# Context

## Research question

By 2021, the recipe that unlocked large-scale models in natural language processing has not transferred to computer vision. In NLP, self-supervised pre-training — remove part of the input, train the model to predict the removed content — lets models grow to hundreds of billions of parameters and generalize across downstream tasks without labels. In vision, the dominant practical pre-training paradigm is still *supervised*: it consumes ever-larger labeled datasets, and the strongest backbones now demand hundreds of millions of often-inaccessible labeled images to reach their potential.

The corruption-and-reconstruction idea — hide part of an image, reconstruct it — is conceptually natural in vision and predates the language successes. Yet progress on reconstruction-based pre-training in vision visibly lags NLP. The concrete goal is therefore: design a self-supervised pre-training task for images that (i) is hard enough to force genuine representation learning rather than low-level pattern completion, (ii) scales — in both accuracy and compute — to the largest vision backbones, and (iii) needs only the unlabeled images themselves, no extra tokenizer, no extra data, no heavy augmentation engineering. A solution must also explain *why* the straightforward port of the language recipe underperforms in vision.

## Background

**Self-supervised pre-training in NLP.** Two families dominate. Autoregressive language modeling (the GPT line, Radford et al. 2018/2019; Brown et al. 2020) predicts the next token left-to-right. BERT-style masked language modeling (Devlin et al. 2019) holds out a portion of the tokens and predicts them from bidirectional context. Both "remove and predict" and both scale extremely well. BERT masks roughly 15% of tokens; this works because language is a human-generated, highly semantic, information-dense signal: predicting even a few missing words requires sophisticated understanding of syntax and meaning.

**Autoencoding and denoising autoencoders.** Autoencoding is a classical representation-learning approach: an encoder maps the input to a latent code, a decoder reconstructs the input. PCA and k-means can be cast as autoencoders (Hinton & Zemel 1994). Denoising autoencoders (Vincent et al. 2008) corrupt the input and train the network to reconstruct the clean original, so the learned features are robust to that corruption; masking is one such corruption. Classically the reconstruction loss is computed over *all* outputs, and the encoder/decoder are roughly symmetric.

**Why remove-and-predict lagged in vision.** Three differences set vision apart from language.

- *Architecture.* For a decade convolutional networks (LeCun et al. 1989; Krizhevsky et al. 2012; He et al. 2016) dominated vision. Convolutions act on regular grids, and it is awkward to inject "indicator" tokens such as a mask placeholder, or to add positional embeddings the way Transformers do. This gap was closed by the Vision Transformer (Dosovitskiy et al. 2021), which slices an image into non-overlapping patches, linearly projects each into a token, adds positional embeddings, and feeds the sequence to a standard Transformer (Vaswani et al. 2017). With ViT, mask tokens and positional embeddings are native, so architecture is no longer the obstacle.

- *Information density.* Images are natural signals with heavy spatial redundancy. A missing patch can usually be recovered from its neighbors by interpolating texture and extending edges — little high-level understanding of objects or scenes is needed. So porting BERT's low masking ratio to images yields a task solvable by low-level extrapolation, which is the crux of why the naive port underperforms.

- *Decoder role.* In language the decoder predicts missing *words*, which carry rich semantics, so a trivial decoder (an MLP) suffices in BERT. In vision a pixel-reconstructing decoder outputs a signal of *low* semantic level, well below recognition. Whatever specialization reconstruction demands has to live somewhere; where it lives shapes how abstract the encoder's latent ends up being.

**Diagnostic findings from prior patch-hiding work.** The Vision Transformer's own preliminary masked-patch-prediction experiment — mask out patches and predict them, BERT-style, at a moderate ratio — gave only about a 2% gain over training from scratch for ViT-B and still trailed supervised pre-training by ~4%; the masking gains "were not as significant as in NLP." Earlier image reconstruction methods used masking ratios in the 20–50% range. These observations establish the problem: the direct transplant of the language recipe does not pay off in vision.

## Baselines

**BERT (Devlin et al. 2019).** Mask ~15% of tokens, replace them with a learned [MASK] token *fed into the encoder*, and predict the originals from bidirectional context; the loss is computed only on masked positions, and the prediction head is a trivial MLP. The template to adapt. Its two transplanted choices — the low masking ratio and feeding mask tokens through the encoder — are precisely the ones that do not carry over to the redundant image signal.

**Context Encoder (Pathak et al. 2016).** A convolutional network inpaints a large *contiguous* missing region of an image, trained with reconstruction plus an adversarial loss, to learn features. It pioneered masked-image feature learning. Limitations: it is convolutional (a regular grid that cannot simply drop tokens, so masked area is still processed), removes a single big block rather than scattered patches, relies on GAN training, and its representations lag supervised pre-training.

**iGPT (Chen et al. 2020).** Pre-trains a Transformer on sequences of *pixels*, with a k-means 9-bit color palette, using autoregressive or BERT-style objectives. It shows pixel prediction can learn useful features, but at very low resolution and very large compute cost.

**BEiT (Bao et al. 2021).** BERT for images: mask patches and predict *discrete visual tokens* produced by a separately pre-trained discrete VAE (the DALL-E dVAE, an 8192-entry codebook), with block-wise masking around 40%. Mask tokens are fed into the encoder. Limitations: it needs an extra tokenizer pre-training stage that itself depends on ~250M images; the dVAE encoder is a large convolutional network adding ~40% of ViT-L's FLOPs; encoder-side mask tokens both waste compute and create a gap between pre-training and deployment; and the recipe is tied to tokenizer targets rather than solving pixel reconstruction directly.

**Contrastive learning (Wu et al. 2018; He et al. 2020; Chen et al. 2020 SimCLR; Grill et al. 2020 BYOL; Chen & He 2021; Caron et al. 2021 DINO; Chen et al. 2021 MoCo v3).** Learns by pulling together representations of two augmented views of an image (and pushing apart different images, or only pulling together). Strong linear-probing features, but heavily dependent on hand-designed data augmentation (crop-only reduces BYOL/SimCLR sharply), and it has no reconstruction objective. It is the prevailing self-supervised direction in vision, and the yardstick a reconstruction-based method is measured against.

## Evaluation settings

Self-supervised pre-training is done on the ImageNet-1K training set (no labels). Learned representations are assessed three ways: end-to-end fine-tuning, linear probing (train a linear classifier on frozen features), and partial fine-tuning (unfreeze only the last *k* Transformer blocks, interpolating between the two). The metric is top-1 validation accuracy on a single 224×224 center crop. Backbones are vanilla ViT-B/16, ViT-L/16, and ViT-H/14. Transfer is measured on object detection and instance segmentation on COCO (Mask R-CNN with a ViT backbone adapted to an FPN), semantic segmentation on ADE20K (UperNet), additional classification on iNaturalist and Places, and robustness on ImageNet variants (IN-C/A/R/Sketch) by inference only.

## Code framework

A vision pre-training codebase already provides ViT blocks, a patch-embedding layer, fixed two-dimensional sine-cosine positions, an ImageNet image-folder dataloader, AdamW, mixed precision, gradient accumulation, and a cosine learning-rate schedule.

```python
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.lr_sched import adjust_learning_rate

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    ...

class VisionPretrainingModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # TODO: add the self-supervised prediction pathway.

    def encode(self, imgs):
        # TODO: choose the training view and produce image representations.
        pass

    def predict(self, encoded, state):
        # TODO: map representations to the self-supervised target space.
        pass

    def compute_loss(self, imgs, prediction, state):
        # TODO: define the self-supervised scalar objective.
        pass

    def forward(self, imgs):
        encoded, state = self.encode(imgs)
        prediction = self.predict(encoded, state)
        return self.compute_loss(imgs, prediction, state)


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train(True)
    optimizer.zero_grad()
    for data_iter_step, (samples, _) in enumerate(data_loader):
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss = model(samples)            # TODO: self-supervised scalar loss
        loss = loss / args.accum_iter
        loss_scaler(
            loss, optimizer, parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()
```
