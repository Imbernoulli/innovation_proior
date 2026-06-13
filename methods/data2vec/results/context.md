# Context

## Research question

Self-supervised learning has succeeded separately in NLP, speech, and vision, but each modality grew its own algorithm and its own *learning objective*, designed with that modality's quirks in mind. The objectives all differ in what target the model is asked to predict, and that target is always both modality-specific and *local* to a position: a sub-word in text, a discrete unit of speech, a visual token or pixel for an image. These per-modality designs carry per-modality inductive biases, and it is unclear which biases generalize.

The question is whether a *single* learning objective — literally the same algorithm — can train strong representations in speech, vision, and language at once, removing the modality-specific target machinery (the learned speech-unit vocabularies, the offline visual tokenizers, the pixel-regression normalizations). A solution would have to define a target that (a) can be constructed identically in every modality, (b) does not require a predefined, closed vocabulary, and (c) gives a learning signal at least as rich as the modality-specific local targets it replaces.

## Background

**Masked prediction.** The dominant pretext task in NLP and increasingly in speech and vision: hide part of the input and predict the hidden part from the visible context. BERT masks ~15% of sub-word tokens and predicts them. In speech, spans are masked because adjacent audio frames are highly correlated, so single-frame masking is trivial. In vision, blocks of patches are masked. The common structure is "encode a corrupted input, predict what was removed"; what differs is *what* is predicted.

**The target problem per modality.** NLP gets discrete targets for free from word boundaries. Speech has no natural vocabulary of units, so models *learn* one — wav2vec 2.0 quantizes latent frames into discrete codes via product quantization with a Gumbel-softmax; HuBERT runs an iterative loop that k-means-clusters layer representations into pseudo-labels, predicts them, then re-clusters. Vision either learns discrete visual tokens with an offline discrete VAE (BEiT, from DALL-E), regresses raw (per-patch-normalized) pixels (MAE), or learns augmentation invariance (SimCLR, BYOL, DINO). Each of these is extra machinery built to manufacture a target.

**Latent-target self-distillation.** A separate line predicts a *latent representation* produced by a momentum (EMA) teacher rather than a hand-defined target. BYOL trains an online network to predict the projection of a target network whose weights are an EMA of the online network's, with a predictor head and a stop-gradient on the target to avoid collapse. DINO does the analogous thing with centering+sharpening instead of a predictor. In both, the teacher and student are fed *different augmentations* of the image and only the *top* layer's representation is the target — so the task is to be invariant to augmentation, and the supervision is a single final-layer vector.

**Mean Teacher.** The EMA-teacher idea originates here (Tarvainen & Valpola 2017): a teacher whose weights track an exponential moving average of the student gives stable targets for a consistency loss.

**Contextualization (diagnostic facts).** A representation built by self-attention over the *entire* input is *contextualized*: the vector at a position carries information from the whole sample. This is qualitatively different from a discrete token (a word id, a visual-token id) or a pixel, which is *local* — it carries information isolated to that position, and a given discrete unit gets one fixed embedding regardless of the sentence/image it appears in. Two further observed facts shape the design: in self-supervised speech Transformers, the *middle* layers transfer better than the *top* layer for downstream tasks (the top layer over-specializes); and learning collapses to a constant target unless something promotes variance among target representations.

**Collapse.** When a model predicts its own (teacher's) representations, the trivial solution is for all targets to become identical. Contrastive methods avoid this with negatives; BYOL with the predictor + stop-gradient + EMA; VICReg with an explicit variance term. Any latent-regression method must contend with this.

## Baselines

**BERT (Devlin et al. 2018).** Masked language modeling: mask 15% of BPE tokens (80% MASK, 10% random, 10% unchanged), predict the original token id with cross-entropy over a closed vocabulary. Gap: the target is a fixed, predefined discrete vocabulary; each token gets a single embedding that must fit all of its contexts, and the target is local.

**wav2vec 2.0 (Baevski et al. 2020).** Mask spans of latent speech frames (sample p of start indices, mask the next several frames), contrastive loss distinguishing the true quantized latent of a masked frame from distractors; the discrete codebook is learned jointly. Gap: requires a learned quantization module; targets are discrete and local.

**HuBERT (Hsu et al. 2020).** Alternate between k-means clustering of layer representations (to make pseudo-label units) and masked prediction of those units. Gap: multi-stage, a fixed number of discrete units, targets local.

**BEiT (Bao et al. 2021).** MIM predicting discrete visual-token ids from an offline frozen discrete VAE (vocabulary 8192), with blockwise masking (≥16 patches per block, random aspect ratio) of a 224×224 image's 196 patches. Gap: needs a separately trained offline tokenizer on extra data; targets discrete and local.

**MAE / SimMIM (He et al. 2021; Xie et al. 2021).** Regress the raw (normalized) pixels of masked patches. Gap: pixel targets are local and low-level; capacity spent on high-frequency detail.

**BYOL / DINO (Grill et al. 2020; Caron et al. 2021).** Regress the EMA-teacher's *top-layer* latent representation; teacher and student see *different augmentations* of the image; collapse avoided by predictor+stop-grad (BYOL) or centering+sharpening (DINO). Gap as a general SSL target: the objective is augmentation-invariance, not masked fill-in; only the final layer is used; and it is a vision recipe, not used across speech and text.

## Evaluation settings

- **Vision:** pretrain on ImageNet-1K images; fine-tune for ImageNet-1K classification (mean-pool the last block, softmax classifier); report top-1 validation accuracy. Architectures ViT-B (12 layers, hidden 768) and ViT-L (24 layers, hidden 1024); 224 input, 16×16 patches, 196 tokens.
- **Speech:** pretrain on Librispeech 960h; fine-tune for ASR with labeled amounts from 10 minutes to 960 hours, following the wav2vec 2.0 regime; report word error rate. Also AudioSet for audio tagging.
- **NLP:** pretrain on BooksCorpus + English Wikipedia (BERT setup), 1M updates, batch 256, sequences up to 512 tokens; fine-tune and evaluate on GLUE (MNLI, QNLI, RTE, MRPC, QQP, STS-B, CoLA, SST-2); report average development accuracy over five runs.
- **Optimizers/schedules:** Adam; cosine schedule (vision), tri-stage warmup/hold/decay (speech, NLP). EMA updates kept in fp32 for numerical stability.

## Code framework

The primitives that already exist: a standard Transformer encoder; modality-specific feature encoders and masking strategies taken from prior work (ViT patch embedding + blockwise masking for images; conv encoder + span masking for speech; token embedding + BERT masking for text); a learned MASK embedding; an EMA mechanism for building a teacher from a student; Adam with cosine/tri-stage schedules. The slots below are what a single cross-modal objective would fill in.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):     # modality-specific (exists): patches / conv / embedding
    def forward(self, x):
        ...

def apply_mask(x, mask_emb):         # modality-specific masking (exists)
    # replace masked positions with the learned MASK embedding; return x, mask_indices
    ...

class TransformerEncoder(nn.Module): # standard Transformer (exists)
    def forward(self, x, return_layer_results=False):
        # returns final hidden states; optionally per-block intermediate features
        ...

class EMA:                           # mean-teacher EMA (exists)
    def __init__(self, model, decay): ...
    def step(self, model):           # Delta <- tau*Delta + (1-tau)*theta
        ...
    def set_decay(self, d): ...

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.encoder = TransformerEncoder()
        self.mask_emb = nn.Parameter(torch.zeros(cfg.dim))
        self.ema = None              # built lazily as EMA of self
        # TODO: the single cross-modal objective goes here.

    def forward(self, x, mask=True):
        feats = self.feature_encoder(x)
        student_in, mask_idx = apply_mask(feats, self.mask_emb)
        student_out = self.encoder(student_in)
        # TODO: define the prediction target and the training loss.
        loss = None
        return loss
```
