# Context

## Research question

Can a vision-language model be built **cheaply by reusing already-trained, off-the-shelf unimodal models** — a strong vision encoder and a large language model — without retraining either of them, and still achieve effective alignment between what the image shows and what the language model can say about it?

The dominant recipe for vision-language pre-training trains large image and text networks **end to end** on hundreds of millions of image–text pairs. As the models grow, that end-to-end cost grows with them, and the recipe is structurally wasteful: the vision and NLP communities are *already* releasing excellent pre-trained backbones — high-quality image encoders and, increasingly, large language models with strong generation and zero-shot transfer abilities — yet an end-to-end VLP pipeline cannot easily absorb them, because everything is trained jointly from a shared starting point.

The obvious move is to keep the pre-trained unimodal models **frozen**. Freezing slashes compute and sidesteps catastrophic forgetting of what those models already know. But it sharpens the central difficulty. A large language model has never seen an image during its unimodal pre-training; its input space is text-token embeddings, not pixels or visual features. If the language model is frozen, *nothing inside it will adapt to vision* — so the entire burden of cross-modal alignment falls on whatever small module sits between the frozen vision encoder and the frozen language model. A satisfactory solution must (a) keep both unimodal models frozen, (b) introduce only a lightweight trainable bridge, and (c) make that bridge produce a visual representation the frozen language model can actually consume and reason over.

## Background

**The field state.** Vision-language pre-training had converged on a handful of architectures — dual encoders, fusion encoders, encoder-decoders, and unified transformers — and a handful of time-tested objectives: image-text contrastive learning, image-text matching, and (masked) language modeling. The prevailing practice trained these large models end to end on web-scale image–text pairs, pushing accuracy up while pushing compute cost up with it.

**Frozen unimodal models are an attractive but underexploited resource.** Pre-trained vision transformers (e.g. the image encoders distilled by large contrastive or supervised pre-training) give high-quality visual features for free. Large language models give strong text generation and the ability to follow instructions and transfer zero-shot. Reusing them — frozen — would make VLP both cheaper and more *generic*, in that better unimodal backbones could simply be swapped in. The catch is alignment: bridging frozen-vision features to a frozen-LLM's text-embedding space, given that the LLM was never trained on anything visual.

**The diagnostic that motivates a new bridge.** Methods that condition a frozen (or lightly tuned) language model on images existed, and they all leaned on a single training signal — an **image-to-text generation loss**: feed the image (as soft prompts or via inserted cross-attention) and ask the language model to produce the caption. This is the most natural objective, but it turns out to be *insufficient on its own* to bridge the modality gap when the language model is frozen — generation alone does not force the visual representation into a form that aligns tightly with language, leaving the LLM struggling to ground its output in the image. That gap — generation-only is not enough — is the empirical fact a better bridge has to answer.

**The shape of the visual input.** A frozen image encoder emits a large, variable-length grid of features (hundreds of patch tokens, each high-dimensional), and the count changes with image resolution. Most of that grid is irrelevant to any given piece of text. Feeding the full grid into a frozen LLM is both expensive (cross-attention or prefixing over hundreds of visual tokens, per example) and noisy. Available in the toolbox are architectures that map a variable-size set of features to a fixed-size set of outputs via cross-attention with a learned, fixed-count set of latent vectors (e.g. Perceiver, Jaegle et al. 2021; the object queries of DETR, Carion et al. 2020).

## Baselines

**Dual-encoder contrastive (CLIP, Radford et al. 2021; ALIGN, Jia et al. 2021).** Two separate encoders, one per modality, trained so matched image–text pairs have high cosine similarity and mismatched pairs low, via an in-batch softmax contrastive loss (image-text contrastive, ITC). Excellent for retrieval and zero-shot classification. Gap: there is no *fusion* of the two modalities, so the model cannot do joint vision-language reasoning such as visual question answering, and it cannot generate language.

**Fusion / unified encoders with multi-objective training (ALBEF, Li et al. 2021; BLIP, Li et al. 2022).** Add a multimodal encoder on top of the unimodal ones and train with several objectives jointly — image-text contrastive (ITC), image-text matching (ITM, a binary matched/unmatched classifier, sharpened with **hard-negative mining**), and (masked or autoregressive) language modeling. BLIP also introduces CapFilt, generating and filtering synthetic captions for noisy web images. These give fine-grained alignment and both understanding and generation. Gap: trained end to end, so they are expensive, and the architecture does not readily plug in a separately pre-trained frozen LLM.

**Frozen-LLM conditioning via image-to-text generation (Frozen, Tsimpoukelli et al. 2021; Flamingo, Alayrac et al. 2022).** Keep the language model frozen and feed it the image. Frozen trains a vision encoder whose outputs are used directly as **soft prompts** prepended to the frozen LM, supervised by an image-caption (language-modeling) loss. Flamingo inserts new **gated cross-attention** layers into the frozen LM to inject visual features and trains them on billions of pairs, also with a language-modeling loss. Both rely solely on the generation loss. Gap: the image-to-text generation objective alone is insufficient to learn tight vision-language alignment under a frozen LLM; and Flamingo in particular needs an enormous amount of trainable cross-attention capacity and data.

**Frozen image encoder for contrastive (LiT, Zhai et al. 2022).** Locks a pre-trained image encoder and only trains the text tower for contrastive alignment. Establishes that a frozen, strong image encoder is a fine basis for cross-modal learning. Gap: still a dual-encoder with no fusion and no generation.

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
