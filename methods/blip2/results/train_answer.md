Vision-language pre-training has converged on training large image encoders and large language models jointly, end to end, on hundreds of millions of image-text pairs. That recipe works, but it is structurally wasteful: the cost grows with the backbones, and it cannot absorb the excellent pre-trained unimodal models that the vision and NLP communities already release. A better strategy would keep both a strong vision encoder and a strong language model frozen, and learn only a small trainable bridge between them. Freezing is cheap and avoids catastrophic forgetting, but it also makes alignment much harder, because a frozen language model never saw an image during its pre-training and will not adapt to vision. All of the cross-modal work must therefore live in the bridge.

Existing frozen-LLM approaches try to solve this with a single image-to-text generation loss. They feed image features into a frozen language model as soft prompts or through inserted cross-attention layers and ask the model to generate the caption. That objective is natural, but it is not enough on its own. A fluent frozen language model has a powerful language prior, so it can produce plausible text from weak visual conditioning. The generation loss is then partly satisfied by the LM's own prior, which dilutes the pressure on the visual representation to become tightly aligned with the actual image content. The bridge needs a representation-learning stage before generation, one that forces a small set of visual vectors to carry the specific information language will need.

The method is BLIP-2, short for Bootstrapping vision-language pre-training from frozen unimodal models. Its core is a lightweight trainable bridge called the Querying Transformer, or Q-Former. The Q-Former takes the variable-length, high-dimensional feature grid output by a frozen image encoder and compresses it into a small fixed set of learned query embeddings. These queries are themselves parameters: 32 learnable vectors, each 768-dimensional. They self-attend to each other, cross-attend to the frozen image features, and share their self-attention layers with text tokens. Cross-attention is inserted only every other transformer block, so the freshly initialized cross-attention capacity does not overwhelm the pretrained BERT-base weights used to initialize the rest of the module. Because the number of queries is fixed, the output size is independent of image resolution, and the small 32-by-768 bottleneck forces the queries to drop visual information that is irrelevant to the text.

The Q-Former is trained in two stages. In stage one, only the image encoder is attached and kept frozen, and the Q-Former is trained with three objectives that share the same weights but use different self-attention masks between queries and text. Image-text contrastive learning aligns the query outputs with a text representation computed from the [CLS] token. Since there are 32 query vectors, the image-text similarity is computed by taking the maximum similarity over all queries, which lets different queries specialize and captures whichever query aligns with the text. A unimodal attention mask prevents queries and text from attending to each other, so the alignment cannot cheat by copying. Image-grounded text generation uses a causal-on-text mask: queries attend only to each other, while each text token attends to all queries and to its own past tokens. Text has no direct path to the image features, so any visual information needed for generation must pass through the queries first. The first text token is replaced by a special [DEC] token to signal decoding. Image-text matching uses a bidirectional mask so queries and text fully fuse, and a per-query binary matched-or-unmatched head averages its logits over the queries. Hard negatives are mined using the contrastive similarities, so the classifier learns fine-grained discrimination. Together these three objectives shape the queries into a compact, language-aligned visual representation.

In stage two, the Q-Former is attached to a frozen large language model. A single fully-connected layer projects the 32 query outputs into the LLM's text-embedding dimension, and the projected vectors are prepended to the input text embeddings as soft visual prompts. Because stage one already aligned the queries with language, the LLM receives a clean, small, pre-aligned signal rather than a noisy raw feature grid. For a decoder-only LLM such as OPT, the model is trained with a standard language-modeling loss. For an encoder-decoder LLM such as FlanT5, a prefix language-modeling loss is used: the visual prompt and a text prefix go to the encoder, and the suffix is the decoder target. Gradients flow only into the Q-Former and the projection layer; both unimodal models remain frozen.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormer(nn.Module):
    """Lightweight querying transformer bridging a frozen image encoder and a frozen LLM."""

    def __init__(self, bert, num_query=32, d=768, proj_dim=256, llm_dim=2048,
                 dec_token_id=None, pad_id=0):
        super().__init__()
        self.bert = bert
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, d))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.num_query = num_query
        self.dec_token_id = dec_token_id
        self.pad_id = pad_id

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.vision_proj = nn.Linear(d, proj_dim)
        self.text_proj = nn.Linear(d, proj_dim)
        self.itm_head = nn.Linear(d, 2)
        self.llm_proj = nn.Linear(d, llm_dim)


def stage1_losses(qf, image_feats, text):
    """image_feats: [B, num_patches, d_v] from the frozen image encoder."""
    B = image_feats.size(0)
    queries = qf.query_tokens.expand(B, -1, -1)
    query_atts = torch.ones(queries.shape[:-1], dtype=torch.long, device=queries.device)

    # Image-Text Contrastive: unimodal mask, max over queries.
    q_out = qf.bert(query_embeds=queries, encoder_hidden_states=image_feats,
                    use_cross_attention=True)
    img_feat = F.normalize(qf.vision_proj(q_out), dim=-1)
    t_out = qf.bert(input_ids=text.input_ids, attention_mask=text.attention_mask)
    txt_feat = F.normalize(qf.text_proj(t_out[:, 0, :]), dim=-1)
    sim = torch.einsum("iqd,jd->ijq", img_feat, txt_feat)
    sim_i2t = sim.max(-1)[0] / qf.temp
    sim_t2i = sim.permute(1, 0, 2).max(-1)[0] / qf.temp
    labels = torch.arange(B, device=queries.device)
    loss_itc = (F.cross_entropy(sim_i2t, labels) +
                F.cross_entropy(sim_t2i, labels)) / 2

    # Image-grounded Text Generation: causal on text, info routed through queries.
    dec_ids = text.input_ids.clone()
    dec_ids[:, 0] = qf.dec_token_id
    lm_labels = dec_ids.masked_fill(dec_ids == qf.pad_id, -100)
    attn = torch.cat([query_atts, text.attention_mask], dim=1)
    loss_itg = qf.bert(query_embeds=queries, input_ids=dec_ids, attention_mask=attn,
                       encoder_hidden_states=image_feats,
                       labels=lm_labels, causal_text=True).loss

    # Image-Text Matching: bidirectional mask, hard negatives, averaged per-query logits.
    img_pairs, txt_pairs, match_labels = mine_hard_negatives(
        sim_i2t.detach(), sim_t2i.detach(), image_feats, text)
    num_pairs = img_pairs.size(0)
    pair_queries = qf.query_tokens.expand(num_pairs, -1, -1)
    pair_query_atts = torch.ones(pair_queries.shape[:-1], dtype=torch.long,
                                 device=pair_queries.device)
    pair_atts = torch.cat([pair_query_atts, txt_pairs.attention_mask], dim=1)
    fused = qf.bert(query_embeds=pair_queries, input_ids=txt_pairs.input_ids,
                    attention_mask=pair_atts, encoder_hidden_states=img_pairs)
    logits = qf.itm_head(fused[:, :qf.num_query, :]).mean(dim=1)
    loss_itm = F.cross_entropy(logits, match_labels)

    return loss_itc + loss_itg + loss_itm


def encode_visual_prompt(qf, frozen_image_encoder, image):
    with torch.no_grad():
        image_feats = frozen_image_encoder(image)
    queries = qf.query_tokens.expand(image_feats.size(0), -1, -1)
    return qf.bert(query_embeds=queries, encoder_hidden_states=image_feats,
                   use_cross_attention=True)


def stage2_loss(qf, frozen_image_encoder, frozen_llm, image, text,
                encoder_decoder=False):
    q_out = encode_visual_prompt(qf, frozen_image_encoder, image)
    visual_prompt = qf.llm_proj(q_out)

    if not encoder_decoder:
        text_embeds = frozen_llm.get_input_embeddings()(text.input_ids)
        inputs = torch.cat([visual_prompt, text_embeds], dim=1)
        prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                                 device=visual_prompt.device)
        attn = torch.cat([prompt_atts, text.attention_mask], dim=1)
        prompt_labels = torch.full(visual_prompt.shape[:-1], -100,
                                   dtype=torch.long, device=inputs.device)
        labels = torch.cat([prompt_labels, text.labels], dim=1)
        return frozen_llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss
    else:
        prefix_embeds = frozen_llm.get_input_embeddings()(text.prefix_ids)
        enc_inputs = torch.cat([visual_prompt, prefix_embeds], dim=1)
        prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                                 device=visual_prompt.device)
        enc_attn = torch.cat([prompt_atts, text.prefix_attention_mask], dim=1)
        suffix_labels = text.suffix_ids.masked_fill(text.suffix_ids == qf.pad_id, -100)
        return frozen_llm(inputs_embeds=enc_inputs, attention_mask=enc_attn,
                          labels=suffix_labels).loss
```
