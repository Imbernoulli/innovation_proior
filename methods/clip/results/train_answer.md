The problem is how to train a visual model whose vocabulary of recognizable concepts is not fixed in advance and whose predictions can be specified at test time with ordinary language. Standard supervised vision fits a fixed softmax head to a predetermined set of categories, so every new concept requires a fresh labeled dataset and a new output layer. That design puts a hard ceiling on supervision and leaves no mechanism for zero-shot transfer. Earlier work did try to learn from image–text pairs, but it either generated captions, which is expensive because it models phrasing the task does not require, or predicted fixed bags of words, which still locked the model to a closed vocabulary. A few pioneering systems such as Visual N-Grams showed the right interface — classify by scoring class names under a pretrained image–text model — but they operated at small scale with weak objectives and reached only proof-of-concept accuracy.

What is needed is a scalable, cheap objective that couples images and language without forcing the model to generate text, plus an inference rule that turns class descriptions into classifier weights. The right move is to stop predicting words and start matching whole texts to whole images within a training batch.

The method is CLIP, Contrastive Language–Image Pre-training. It jointly trains an image encoder and a text encoder so that, given a batch of N real image–text pairs, the model must pick the matching text for each image and the matching image for each text among the N examples in the batch. Both encoders project their outputs into a shared embedding space, the projections are L2-normalized so that dot products become cosine similarities, and the N-by-N matrix of pairwise similarities is fed to a symmetric cross-entropy loss whose labels are simply the diagonal indices 0 through N minus one. The other examples in the batch act as negatives, so no extra forward passes are needed. Because raw cosine similarities lie in [-1, 1] and therefore give the softmax too little dynamic range, the logits are scaled by a learned inverse temperature. The temperature is stored as a log-space scalar so it stays positive, initialized to the equivalent of 0.07, and clamped so its exponential never exceeds 100, which prevents the scale from running away and destabilizing training. This objective is an instance of the multi-class N-pair or InfoNCE contrastive family, and it is dramatically cheaper per example than caption generation: generation models exact token sequences, while matching asks only one classification decision per example over batch-minus-one distractors.

At inference time, the trained text encoder behaves like a hypernetwork that emits the weight vector of a linear classifier from any natural-language description. For each candidate class, one or more prompt templates such as "a photo of a {label}." are encoded and averaged in the shared embedding space. Averaging embeddings rather than probabilities keeps inference cost unchanged while supplying context that disambiguates polysemous words and matches the sentence-like distribution seen during pretraining. The image is encoded the same way, and classification becomes scaled cosine similarity followed by an argmax. No labels are fit; describing a class is equivalent to building its classifier.

Architecture choices are kept practical. The image encoder is either a modified ResNet with ResNet-D antialiasing tweaks and a final attention-pooling layer replacing global average pooling, or a Vision Transformer with an extra layer normalization on the patch-plus-position embeddings. The text encoder is a GPT-2-style Transformer over byte-pair tokens, using causal masking so the model can later accept language-model initialization or an auxiliary LM objective if desired. The representation at the end-of-sequence token is layer-normalized and linearly projected into the shared space. Linear rather than non-linear projection heads are used because the shared space is itself the retrieval and classification space. Training starts from scratch with only a random square crop for augmentation, since hundreds of millions of image–text pairs already provide enough diversity to make heavy augmentation unnecessary.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoderModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_image, d_text, d_embed):
        super().__init__()
        self.image_encoder = image_encoder          # CNN or ViT -> [n, d_image]
        self.text_encoder = text_encoder            # Transformer -> [n, d_text]
        self.image_proj = nn.Linear(d_image, d_embed, bias=False)   # W_i
        self.text_proj = nn.Linear(d_text, d_embed, bias=False)     # W_t
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return F.normalize(self.image_proj(self.image_encoder(image)), dim=-1)

    def encode_text(self, text):
        return F.normalize(self.text_proj(self.text_encoder(text)), dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)   # [n, d_embed]
        text_features = self.encode_text(text)      # [n, d_embed]
        return image_features, text_features


def training_objective(image_features, text_features, model):
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    logits_per_image = logit_scale * (image_features @ text_features.t())  # [n, n]
    logits_per_text = logits_per_image.t()
    n = logits_per_image.shape[0]
    labels = torch.arange(n, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)   # each image picks its text
    loss_t = F.cross_entropy(logits_per_text, labels)    # each text picks its image
    return (loss_i + loss_t) / 2                          # symmetric


def train(model, dataloader, optimizer):
    for images, texts in dataloader:
        image_features, text_features = model(images, texts)
        loss = training_objective(image_features, text_features, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def _format_prompt(template, description):
    if "{label}" in template:
        return template.format(label=description)
    return template.format(description)


@torch.no_grad()
def build_classifier_from_descriptions(model, class_descriptions, tokenize,
                                       templates=("a photo of a {label}.",)):
    weights = []
    device = next(model.parameters()).device
    for description in class_descriptions:
        prompts = tokenize([_format_prompt(t, description) for t in templates]).to(device)
        embeds = model.encode_text(prompts)                          # [n_templates, d_embed]
        class_vec = F.normalize(embeds.mean(dim=0), dim=-1)          # ensemble in embedding space
        weights.append(class_vec)
    return torch.stack(weights, dim=0)                               # [K, d_embed]


@torch.no_grad()
def classify(model, image, classifier):
    image_features = model.encode_image(image)                       # [1, d_embed]
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    logits = logit_scale * (image_features @ classifier.t())         # [1, K]
    return logits.argmax(dim=-1)                                     # no labels were fit
```
