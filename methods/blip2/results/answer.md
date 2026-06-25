# BLIP-2: Bootstrapping vision-language pre-training from frozen unimodal models

## Problem

Vision-language pre-training that trains a large image encoder and a large text model jointly, end to end, on hundreds of millions of image–text pairs is structurally wasteful: the cost grows with the backbones, and the pipeline cannot absorb the strong unimodal models — high-quality vision encoders and large language models — that the vision and NLP communities already release. The goal is a **generic, compute-efficient** strategy that keeps both unimodal models **frozen** and learns only a small trainable bridge between them.

Freezing is cheap and avoids catastrophic forgetting, but it sharpens the central difficulty: a frozen large language model (LLM) was never trained on anything visual and will not adapt to images, so *all* of the cross-modal alignment must happen inside the bridge. Prior frozen-LLM methods (e.g. Frozen, Tsimpoukelli et al. 2021; Flamingo, Alayrac et al. 2022) rely on a single **image-to-text generation loss**, which is insufficient on its own to bridge the modality gap: a fluent frozen LLM's language prior absorbs the loss without forcing the visual representation into a form tightly aligned with text.

## Key idea — the Querying Transformer (Q-Former)

The bridge is a lightweight transformer, the **Q-Former**, that acts as an **information bottleneck** between the frozen image encoder and the frozen LLM. It introduces a fixed set of **learnable query embeddings** (32 queries, each 768-dim) that:

- **self-attend** to each other,
- **cross-attend** to the frozen image features (cross-attention inserted **every other** transformer block, so the fresh randomly-initialized cross-attention capacity does not overwhelm the pretrained weights),
- and **share the self-attention layers with text tokens**, so the same module can also process text and the queries can be supervised against it.

The Q-Former is initialized from BERT-base (cross-attention layers random), totals 188M parameters, and the queries are themselves model parameters. Its output `Z` (32×768) is much smaller than the frozen image features (e.g. 257×1024 for ViT-L/14). This bottleneck, together with the pre-training objectives, forces the queries to keep only the visual information most relevant to the text. The query↔text interaction is controlled by **different self-attention masks** depending on the objective.

## Two-stage pre-training

### Stage 1 — vision-language representation learning (frozen image encoder)

Three objectives share the same input format and parameters, differing only by their attention mask between queries and text:

1. **Image-Text Contrastive (ITC).** Align query outputs `Z` with the text representation `t` (the `[CLS]` output of the text transformer). Since `Z` has 32 vectors, compute the similarity of *each* query output to `t` and take the **maximum** as the image-text similarity; train with a two-way in-batch softmax contrastive loss. A **unimodal** mask keeps queries and text from attending to each other to prevent information leak. Because the image encoder is frozen, large batches fit, so **in-batch negatives** replace a momentum queue.

2. **Image-grounded Text Generation (ITG).** Generate the text conditioned on the image. Text tokens have no direct path to the image features, so the information needed to generate the text must be extracted by the queries first and then reach the text through the shared self-attention — forcing the queries to capture all text-relevant visual content. The mask is **multimodal causal** (UniLM-style): queries attend to each other but not to text; each text token attends to all queries and to its earlier text tokens. The first text token is replaced from `[CLS]` to a new `[DEC]` token to signal decoding.

3. **Image-Text Matching (ITM).** Binary matched/unmatched classification with a **bidirectional** mask (all queries and text attend to everything), so each output query embedding captures multimodal information. Each query output is fed to a two-class linear head, and the logits are **averaged over the queries** for the matching score. **Hard-negative mining** (from ALBEF/BLIP), driven by the ITC similarities, supplies the most confusable wrong pairs.

### Stage 2 — vision-to-language generative learning (frozen LLM)

Attach the Q-Former (with the frozen image encoder) to a frozen LLM. A single **fully-connected layer** linearly projects the output query embeddings `Z` into the LLM's text-embedding dimension; the projected queries are **prepended** to the input text embeddings as **soft visual prompts**. Because stage 1 already made the queries carry language-informative visual content, the LLM receives a clean, small, pre-aligned signal — the bottleneck feeds only useful information, reducing the alignment burden and (since the LLM is never updated) avoiding catastrophic forgetting.

Two LLM types:

- **Decoder-only LLM (OPT family):** language-modeling loss — the frozen LLM generates the text conditioned on the visual prompt.
- **Encoder-decoder LLM (FlanT5 family):** prefix language-modeling loss — split the text; concatenate the visual prompt with the **prefix** as the encoder input, and use the **suffix** as the decoder target.

Gradients flow only into the Q-Former and the projection; both frozen models stay fixed.

## Concrete settings

- **Q-Former:** 32 queries, dim 768, 188M params, init from BERT-base, cross-attention random and inserted every other block.
- **Frozen image encoder:** CLIP ViT-L/14 or EVA-CLIP ViT-g/14; use the **second-to-last** layer's output features (slightly better than the last). Cast to FP16.
- **Frozen LLM:** OPT (decoder) or instruction-tuned FlanT5 (encoder-decoder); FP16, BF16 for FlanT5.
- **Data:** 129M images (COCO, Visual Genome, CC3M, CC12M, SBU, 115M from LAION400M), with CapFilt synthetic captions (10 captions from a BLIP-large captioner, ranked by CLIP ViT-L/14 image-text similarity, keep top-2, sample one per step).
- **Schedule:** 250k stage-1 steps, 80k stage-2 steps; AdamW (β₁=0.9, β₂=0.98, weight decay 0.05); cosine decay, peak lr 1e-4, 2k-step linear warmup; stage-2 minimum lr 5e-5; 224×224 images with random-resized-crop and horizontal flip.

## Reference implementation

The code below is a compact reference sketch for the method-level mechanics: a BERT-style Q-Former that takes query embeddings (with `encoder_hidden_states` = frozen image features for cross-attention) and/or text, plus the three masked stage-1 objectives and the stage-2 prefix projection.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormer(nn.Module):
    """Lightweight querying transformer bridging a frozen image encoder and a frozen LLM.

    `self.bert` is a BERT-base-initialized transformer whose self-attention layers are
    SHARED between the learned queries and text tokens; cross-attention to the frozen
    image features is inserted every other block (randomly initialized). It accepts:
      - query_embeds:          the learned query vectors (image transformer input)
      - input_ids / attention_mask: text tokens (text transformer)
      - encoder_hidden_states: frozen image features -> keys/values of cross-attention
    """

    def __init__(self, bert, num_query=32, d=768, proj_dim=256, llm_dim=2048,
                 dec_token_id=None, pad_id=0):
        super().__init__()
        self.bert = bert
        # the bottleneck: a fixed set of learnable query vectors (these ARE parameters)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, d))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.num_query = num_query
        self.dec_token_id = dec_token_id
        self.pad_id = pad_id

        self.temp = nn.Parameter(0.07 * torch.ones([]))      # contrastive temperature
        self.vision_proj = nn.Linear(d, proj_dim)            # ITC similarity space (image)
        self.text_proj = nn.Linear(d, proj_dim)              # ITC similarity space (text)
        self.itm_head = nn.Linear(d, 2)                      # matched/unmatched per query
        self.llm_proj = nn.Linear(d, llm_dim)                # stage-2 soft visual prompt


def stage1_losses(qf: QFormer, image_feats, text):
    """image_feats: [B, num_patches, d_v] from the FROZEN encoder (no grad).
       text: tokenized batch with .input_ids [B, L] and .attention_mask [B, L]."""
    B = image_feats.size(0)
    queries = qf.query_tokens.expand(B, -1, -1)              # [B, 32, 768]
    query_atts = torch.ones(queries.shape[:-1], dtype=torch.long, device=queries.device)

    # ---- ITC: queries cross-attend image ONLY (unimodal mask: no text leak) ----
    q_out = qf.bert(query_embeds=queries, encoder_hidden_states=image_feats,
                    use_cross_attention=True)               # [B, 32, 768]
    img_feat = F.normalize(qf.vision_proj(q_out), dim=-1)   # [B, 32, proj]
    t_out = qf.bert(input_ids=text.input_ids, attention_mask=text.attention_mask)
    txt_feat = F.normalize(qf.text_proj(t_out[:, 0, :]), dim=-1)   # [B, proj] ([CLS])
    # all-pairs similarity of EACH image query to EACH text, then MAX over queries
    sim = torch.einsum("iqd,jd->ijq", img_feat, txt_feat)    # [image B, text B, 32]
    sim_i2t = sim.max(-1)[0] / qf.temp                       # [B, B]
    sim_t2i = sim.permute(1, 0, 2).max(-1)[0] / qf.temp      # [B, B]
    labels = torch.arange(B, device=queries.device)
    loss_itc = (F.cross_entropy(sim_i2t, labels) +
                F.cross_entropy(sim_t2i, labels)) / 2

    # ---- ITG: multimodal CAUSAL mask; text must read the image THROUGH the queries ----
    dec_ids = text.input_ids.clone()
    dec_ids[:, 0] = qf.dec_token_id                          # [DEC] signals decoding
    lm_labels = dec_ids.masked_fill(dec_ids == qf.pad_id, -100)
    # queries attend each other (not text); each text token attends all queries + its past
    attn = torch.cat([query_atts, text.attention_mask], dim=1)
    loss_itg = qf.bert(query_embeds=queries, input_ids=dec_ids, attention_mask=attn,
                       encoder_hidden_states=image_feats,
                       labels=lm_labels, causal_text=True).loss

    # ---- ITM: bidirectional mask, hard negatives, per-query 2-class head, averaged ----
    img_pairs, txt_pairs, match_labels = mine_hard_negatives(
        sim_i2t.detach(), sim_t2i.detach(), image_feats, text)
    num_pairs = img_pairs.size(0)                            # positives + mined hard negatives
    pair_queries = qf.query_tokens.expand(num_pairs, -1, -1)
    pair_query_atts = torch.ones(pair_queries.shape[:-1], dtype=torch.long,
                                 device=pair_queries.device)
    pair_atts = torch.cat([pair_query_atts, txt_pairs.attention_mask], dim=1)
    fused = qf.bert(query_embeds=pair_queries,
                    input_ids=txt_pairs.input_ids, attention_mask=pair_atts,
                    encoder_hidden_states=img_pairs)          # bidirectional fusion
    # take the query slice of the fused output, score each query, average the logits
    logits = qf.itm_head(fused[:, :qf.num_query, :]).mean(dim=1)   # [N, 2]
    loss_itm = F.cross_entropy(logits, match_labels)

    return loss_itc + loss_itg + loss_itm


def encode_visual_prompt(qf: QFormer, frozen_image_encoder, image):
    with torch.no_grad():
        image_feats = frozen_image_encoder(image)           # freeze only the image encoder
    queries = qf.query_tokens.expand(image_feats.size(0), -1, -1)
    return qf.bert(query_embeds=queries, encoder_hidden_states=image_feats,
                   use_cross_attention=True)                 # [B, 32, 768]


def stage2_loss(qf: QFormer, frozen_image_encoder, frozen_llm, image, text,
                encoder_decoder=False):
    q_out = encode_visual_prompt(qf, frozen_image_encoder, image)
    visual_prompt = qf.llm_proj(q_out)                       # [B, 32, llm_dim]

    if not encoder_decoder:
        # decoder-only LLM (OPT): prepend visual prompt, language-modeling loss
        text_embeds = frozen_llm.get_input_embeddings()(text.input_ids)
        inputs = torch.cat([visual_prompt, text_embeds], dim=1)
        prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                                 device=visual_prompt.device)
        attn = torch.cat([prompt_atts, text.attention_mask], dim=1)
        # mask out the visual-prompt positions in the LM targets
        prompt_labels = torch.full(visual_prompt.shape[:-1], -100,
                                   dtype=torch.long, device=inputs.device)
        labels = torch.cat([prompt_labels, text.labels], dim=1)
        # no torch.no_grad() around the frozen LLM call: gradients must flow to inputs
        return frozen_llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss
    else:
        # encoder-decoder LLM (FlanT5): prefix-LM. prefix -> encoder, suffix -> decoder
        prefix_embeds = frozen_llm.get_input_embeddings()(text.prefix_ids)
        enc_inputs = torch.cat([visual_prompt, prefix_embeds], dim=1)
        prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                                 device=visual_prompt.device)
        enc_attn = torch.cat([prompt_atts, text.prefix_attention_mask], dim=1)
        suffix_labels = text.suffix_ids.masked_fill(text.suffix_ids == qf.pad_id, -100)
        return frozen_llm(inputs_embeds=enc_inputs, attention_mask=enc_attn,
                          labels=suffix_labels).loss
```

## Why it works (causal chain)

Freeze a strong vision encoder and a strong LLM to stop relearning what already exists and to avoid catastrophic forgetting → freezing the LLM means it never adapts to vision, so all alignment must live in a small bridge → the generation-only objective prior frozen-LLM methods use is too weak (the fluent LLM's language prior absorbs it), so a representation-learning stage comes first → a fixed set of learned queries cross-attending the frozen features (the Perceiver/DETR device) compresses the variable, mostly-irrelevant feature grid into a 32-vector bottleneck → three mask-differentiated objectives in one shared module (contrastive with a unimodal mask and max-over-queries similarity; grounded generation with a causal-on-text mask that routes information through the queries; matching with a bidirectional mask, per-query averaged classifier, and hard negatives) make the queries language-aligned → stage 2 projects those aligned queries into the frozen LLM's input space as soft visual prompts and trains with the LLM's own (prefix-)language-modeling loss, feeding the LLM exactly the visual information it needs and nothing more.

---

### Verification

- **Problem / motivation** — frozen-unimodal, compute-efficient VLP; generation-only loss insufficient: restated in `results/context.md` (Research question, Background) and `results/reasoning.md` (L1–9).
- **Q-Former architecture** — two submodules sharing self-attention; 32 queries × 768; cross-attention every other block; BERT-base init, cross-attn random; 188M params; bottleneck 32×768 vs 257×1024. Faithful in `answer.md` (Key idea, `QFormer.__init__`).
- **ITC** — `Z` vs `t`=`[CLS]`; all image-text pairs score each query then take **max** over queries; unimodal mask (no leak); in-batch negatives, no momentum queue. Implemented in `stage1_losses` ITC block (`einsum` all-pairs similarity, max over queries / temp, two-way cross-entropy).
- **ITG** — multimodal causal/UniLM mask; queries attend each other not text; each text token attends all queries + previous text tokens; info routed through queries; `[CLS]`→`[DEC]`. Implemented in ITG block (`causal_text=True`, `[DEC]` first token, `-100` pad labels).
- **ITM** — bidirectional mask; per-query two-class head; **average** logits over queries; hard-negative mining. Implemented in ITM block (positive + mined hard-negative pairs, `itm_head` on query slice, `.mean(dim=1)`).
- **Stage 2 prefix projection** — FC layer projects `Z` into LLM embedding dim; prepended as soft visual prompts; decoder LLM = LM loss, encoder-decoder = prefix-LM (prefix→encoder, suffix→decoder target). Implemented in `stage2_loss` (`llm_proj`, prepend, decoder vs encoder-decoder branches; only the frozen image encoder is wrapped in `torch.no_grad()`, while the frozen LLM call remains differentiable with respect to `inputs_embeds`).
- **Settings** — image encoders (CLIP ViT-L/14, EVA-CLIP ViT-g/14, 2nd-last layer), OPT / FlanT5, FP16/BF16, data + CapFilt, 250k/80k steps, AdamW β=(0.9,0.98), wd 0.05, peak lr 1e-4, 2k warmup, stage-2 min lr 5e-5, 224×224. Summarized in `answer.md` (Concrete settings).
- **In-frame** — names BLIP-2 / Q-Former as the method being built; no author/venue/arXiv citation of the source; only prior-art ancestors (CLIP, ALIGN, ALBEF, BLIP, Frozen, Flamingo, Perceiver, DETR) cited by author/year. No proposed-method evaluation numbers used (experiment/limitation sections excluded).
