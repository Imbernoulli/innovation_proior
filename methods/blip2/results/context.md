# Context

## Research question

Can a vision-language model be built by **reusing already-trained, off-the-shelf unimodal models** — a strong vision encoder and a large language model — kept **frozen**, with only a small trainable module learned to connect them, and still align what the image shows with what the language model can say about it?

The dominant recipe for vision-language pre-training trains large image and text networks **end to end** on hundreds of millions of image–text pairs, so the cost grows with the backbones. Meanwhile the vision and NLP communities release excellent pre-trained backbones — high-quality image encoders and, increasingly, large language models with strong generation and zero-shot transfer abilities. Keeping these unimodal models frozen would cut compute and reuse what they already know.

A frozen large language model takes text-token embeddings as input and was never trained on pixels or visual features. With the language model frozen, the cross-modal connection has to be carried entirely by whatever module sits between the frozen vision encoder and the frozen language model. The setting is to design that small trainable bridge and a way to train it so the frozen language model can consume and reason over its output.

## Background

**The field state.** Vision-language pre-training had converged on a handful of architectures — dual encoders, fusion encoders, encoder-decoders, and unified transformers — and a handful of time-tested objectives: image-text contrastive learning, image-text matching, and (masked) language modeling. The prevailing practice trains these large models end to end on web-scale image–text pairs.

**Frozen unimodal models as a resource.** Pre-trained vision transformers (e.g. the image encoders distilled by large contrastive or supervised pre-training) give high-quality visual features. Large language models give strong text generation and the ability to follow instructions and transfer zero-shot. Reusing them frozen makes VLP cheaper and more *generic*, in that better unimodal backbones can be swapped in. The task is alignment: connecting frozen-vision features to a frozen-LLM's text-embedding space, given that the LLM was never trained on anything visual.

**Conditioning a frozen LM on images.** Methods that condition a frozen (or lightly tuned) language model on images train with an **image-to-text generation loss**: feed the image (as soft prompts or via inserted cross-attention) and ask the language model to produce the caption.

**The shape of the visual input.** A frozen image encoder emits a large, variable-length grid of features (hundreds of patch tokens, each high-dimensional), and the count changes with image resolution. Available in the toolbox are architectures that map a variable-size set of features to a fixed-size set of outputs via cross-attention with a learned, fixed-count set of latent vectors (e.g. Perceiver, Jaegle et al. 2021; the object queries of DETR, Carion et al. 2020).

## Baselines

**Dual-encoder contrastive (CLIP, Radford et al. 2021; ALIGN, Jia et al. 2021).** Two separate encoders, one per modality, trained so matched image–text pairs have high cosine similarity and mismatched pairs low, via an in-batch softmax contrastive loss (image-text contrastive, ITC). Used for retrieval and zero-shot classification.

**Fusion / unified encoders with multi-objective training (ALBEF, Li et al. 2021; BLIP, Li et al. 2022).** Add a multimodal encoder on top of the unimodal ones and train with several objectives jointly — image-text contrastive (ITC), image-text matching (ITM, a binary matched/unmatched classifier, sharpened with **hard-negative mining**), and (masked or autoregressive) language modeling. BLIP also introduces CapFilt, generating and filtering synthetic captions for noisy web images. These give fine-grained alignment and support both understanding and generation. Trained end to end.

**Frozen-LLM conditioning via image-to-text generation (Frozen, Tsimpoukelli et al. 2021; Flamingo, Alayrac et al. 2022).** Keep the language model frozen and feed it the image. Frozen trains a vision encoder whose outputs are used directly as **soft prompts** prepended to the frozen LM, supervised by an image-caption (language-modeling) loss. Flamingo inserts new **gated cross-attention** layers into the frozen LM to inject visual features and trains them on billions of pairs, also with a language-modeling loss. Both train with the generation loss.

**Frozen image encoder for contrastive (LiT, Zhai et al. 2022).** Locks a pre-trained image encoder and only trains the text tower for contrastive alignment. Shows that a frozen, strong image encoder is a basis for cross-modal learning.

## Evaluation settings

The natural yardsticks are the established vision-language benchmarks. **Visual question answering** (VQAv2, OK-VQA, GQA) — answer a natural-language question about an image; metric is VQA accuracy (open-ended generation scored against reference answers), evaluated zero-shot or after fine-tuning. **Image captioning** (COCO Karpathy split, NoCaps) — generate a caption; metrics CIDEr, SPICE, BLEU. **Image–text retrieval** (COCO, Flickr30k) — rank texts given an image and vice versa; metric Recall@K. For the contrastive/matching pieces, the protocol mirrors prior VLP: encode a batch of pairs, score all-pairs similarity, train with in-batch negatives (and hard negatives for matching). Zero-shot image-to-text generation is probed by prompting the model with a natural-language instruction (e.g. a question or a caption prefix) and decoding with beam search.

## Code framework

What already exists: a paired `(image, text)` dataloader; a **pre-trained, frozen** vision encoder that maps an image to a grid of visual features; a **pre-trained, frozen** language model with its own text-token embedding space and an autoregressive (or prefix) language-modeling loss; a standard transformer block (self-attention + cross-attention + feed-forward, as in BERT) to initialize a small trainable module from; and an AdamW optimizer with a cosine schedule. What is open: the design of the lightweight trainable bridge between the two frozen models, and how it is trained so that the frozen language model can consume and reason over its output.

```python
import torch.nn as nn

frozen_image_encoder = build_frozen_image_encoder()   # -> visual features [n, num_patches, d_v], requires_grad=False
frozen_llm           = build_frozen_llm()              # text-token in, text out; requires_grad=False


class VisionLanguageBridge(nn.Module):
    """The small trainable module between the two frozen models."""
    def __init__(self, transformer_block):
        super().__init__()
        # TODO: design the trainable bridge
        pass


def bridge_objective(bridge, image_features, text, frozen_llm=None):
    # TODO: how should the bridge be trained so that, from the frozen encoder's
    #       features, it produces something the frozen LLM can ground its text on?
    pass


def train(bridge, dataloader, frozen_image_encoder, frozen_llm, optimizer):
    for image, text in dataloader:
        image_features = frozen_image_encoder(image)      # no grad
        loss = bridge_objective(bridge, image_features, text, frozen_llm)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
