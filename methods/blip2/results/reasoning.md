The thing that keeps nagging me about how we train vision-language models is the sheer waste. Every new VLP system trains a large image network and a large text network jointly, end to end, on a few hundred million image-text pairs, and as the backbones get bigger the bill gets bigger with them. Meanwhile the vision community keeps releasing excellent pre-trained image encoders, and the NLP community keeps releasing large language models that generate fluently and transfer to new tasks zero-shot. Those models cost accelerator-years to make. And my end-to-end pipeline can't really absorb them — it wants to learn everything from a shared starting point. I'm paying, over and over, to relearn things that already exist in a box on a shelf.

So let me take the reuse seriously: keep a pre-trained vision encoder and a pre-trained large language model, and *don't train them*. Freeze both. Two things fall out immediately, both good. Freezing is cheap — the only gradients I compute are for whatever small thing I add. And freezing protects the knowledge: if I were to fine-tune the language model on a narrow vision-language objective it would drift, forgetting the broad competence that made it worth using; locking it in place rules that out by construction.

But the moment I freeze the language model I'm staring at the hard part. This model has never seen an image. Its inputs are text-token embeddings; its whole interior is organized around language. It will not bend toward vision, because I've forbidden it to change. So all of the work of getting visual information into a form it can use has to happen *outside* it, in the bridge I insert between the frozen encoder and the frozen LM. Every bit of cross-modal alignment lives in that bridge. That's the whole problem, restated: align frozen vision to a frozen, vision-blind language model, using only a small trainable module.

Let me look at how people have tried to feed images into a frozen language model, because the failure mode there is exactly the gap I have to close. The pattern is consistent: take the image, turn it into something the LM can read, and train the whole thing with an image-to-text generation loss — show the picture, ask for the caption, backprop the language-modeling loss. Frozen does this by learning a vision encoder whose outputs are soft prompts prepended to the LM. Flamingo does it by splicing new cross-attention layers into the frozen LM and training them on billions of pairs. Both lean entirely on the generation loss, because it's the obvious one: it's exactly what the LM already does.

And here's the wall. Generation alone is not enough to bridge the modality gap when the LM is frozen. Why would that be? Think about what the generation loss actually pressures. It says: given this visual conditioning, make the next text token likely. But a frozen, fluent LM is *very* good at producing plausible text from weak conditioning — it has a strong language prior. So the gradient that's supposed to teach "encode the image faithfully" can be partly satisfied by the LM's own prior filling in plausible words, and the pressure on the visual representation to be *tightly aligned* with the content is diluted. The visual vectors can stay vague — a soft hint the LM smooths over — and the loss still goes down. Nothing in pure generation forces the bridge to extract the *specific* visual facts and put them where language can grab them. So I can't rely on the generation loss by itself. I need a stage, before generation, that hammers the bridge into producing a visual representation that is genuinely, fine-grainedly aligned with text. Then the generation step can ride on top of that.

Now, what should the bridge even output? The frozen encoder hands me a big, variable-size grid — a couple hundred patch tokens for a high-res image, each a thousand-dimensional vector, and the count changes with resolution. Two problems with shoving that straight at the LM. It's expensive: cross-attention or prefixing over hundreds of visual tokens, inside a huge frozen LM, for every example. And it's mostly irrelevant: any given caption is about a few things in the image, not all of it. So I want to *compress* — turn that variable grid into a small, fixed set of vectors that carry the text-relevant content and drop the rest. A fixed, small output also means the cost into the LM is fixed regardless of image resolution.

How do I turn a variable-size set into a fixed-size one? There's a clean device for this: learn a fixed number of latent query vectors and let them cross-attend to the encoder's features — the queries are the parameters, the features are keys and values, and the number of outputs equals the number of queries, completely independent of how many input features there are. That's the Perceiver move, and the same trick as DETR's object queries. So: a set of, say, 32 learnable query embeddings. They cross-attend into the frozen image features and come out as 32 vectors. For ViT-L the frozen grid is 257×1024; my output is 32×768. That's a real bottleneck — and the bottleneck is the point. If the queries can only carry 32 vectors out, and I supervise them correctly, they're *forced* to spend that budget on what matters for the text.

Let me make this a small transformer, not just a single attention layer, so the queries can also reason among themselves and, crucially, interact with text. Here's the shape I want. The queries are the input sequence to an "image transformer." They self-attend to each other; they cross-attend to the frozen image features; and I'll insert the cross-attention only every other block, not every block, because the cross-attention layers are the new randomly-initialized capacity and I don't want to drown the pretrained weights — let self-attention do work in between. I'll initialize the whole thing from BERT-base so the self-attention and feed-forward start from competent language-capable weights; only the cross-attention layers are fresh and random.

But I also need text to flow through this module, because the only way to *supervise* the queries to be language-aligned is to let them be scored against, and interact with, text. So I'll let the same module also process text tokens, sharing the self-attention layers between queries and text. One transformer, two kinds of tokens — image queries and text — sharing self-attention. Now I can control, with attention masks, exactly how much the queries and the text are allowed to see each other. That mask is going to be my knob for getting several different training signals out of one architecture.

So stage one: frozen image encoder, queries + text through this querying module, and I want to drive the queries to extract text-relevant visual information. I learned from the fusion-encoder line that the way to get fine-grained alignment isn't one objective, it's a small family of them, each pressuring a different facet. Let me build three, all sharing the same input format and weights, differing only by the attention mask.

First, contrastive alignment — pull matched image and text together, push mismatched apart. I take the query outputs Z (32 vectors) and the text representation t. For text I'll use the output at a [CLS] token. But here's a wrinkle the dual-encoder methods don't have: Z is *32* vectors, not one. So how do I get a single image-text similarity? Compute the similarity between *each* of the 32 query outputs and t, and take the maximum. The intuition: I don't need every query to match the text; I need *some* query to have captured the text's content. Max-pooling lets different queries specialize and picks whichever one aligns. Scale by a temperature, softmax over the batch, the usual two-way in-batch contrastive. And the mask here matters: I must not let the queries peek at the text while computing this, or the "image" representation would just copy the text and the alignment would be a cheat. So a unimodal mask — queries can't see text tokens and text can't see queries. Side note on negatives: because the image encoder is frozen, the forward pass is cheap and I can fit a big batch on each device, so I don't need the momentum queue the end-to-end methods used to manufacture negatives — plain in-batch negatives suffice.

Second, generation — but now used as a *representation-shaping* signal, not as the only signal. Image-grounded text generation: condition on the image, produce the text. The subtlety is architectural. In this module the text tokens have no direct path to the frozen image features — only the queries cross-attend to the image. So if I ask the text to be generated, the information needed to generate it *has to pass through the queries first* and then reach the text via the shared self-attention. That's exactly the pressure I want: it forces the queries to capture all the visual information the text will need. The mask for this is causal-on-the-text, UniLM-style: queries attend among themselves but not to text; each text token attends to all the queries and to its own earlier text tokens. And I swap the first text token from [CLS] to a [DEC] token to signal "decode" mode.

Third, matching — the finest-grained signal. A binary classifier: is this image-text pair matched or not? Here I want full fusion, so a bidirectional mask: every query and every text token can attend to everything. Now each output query embedding has soaked up multimodal information. Feed each query's output through a two-class linear head to get a matched/unmatched logit, and average those logits over the 32 queries for the final score. And matching is only as good as its negatives — a random mismatched pair is trivially rejected. So I mine hard negatives: use the contrastive similarities to pick the *most confusable* wrong text for each image (and wrong image for each text) and make the classifier separate those. That's what teaches fine-grained discrimination.

Three objectives, one module, three masks: unimodal for contrastive (no leakage), causal-on-text for generation (force info through the queries), bidirectional for matching (full fusion). After this stage the queries are extracting a compact, language-aligned visual representation — which is exactly what the generation-only approaches failed to produce.

Stage two: now bring in the frozen LLM. I have 32 query vectors that already encode language-relevant visual content. I just need to put them into the LLM's input space. The query outputs live in the module's 768-dim space; the LLM has its own text-embedding dimension. So a single fully-connected layer projects the 32 query vectors into the LLM's embedding dimension, and I *prepend* them to the embeddings of the input text. They act as soft visual prompts — the LLM reads them as if they were leading tokens, and generates conditioned on them.

This is where the bottleneck pays off twice. Because stage one already made the queries carry the useful, language-shaped visual information and dropped the rest, the frozen LLM receives a clean, small, pre-aligned signal rather than a noisy grid it would have to make sense of itself. The module is an information bottleneck feeding the LLM only what's useful. That reduces the alignment burden on the frozen LLM — and since the LLM is frozen, "reducing its burden" is the only lever I have. It's also why catastrophic forgetting isn't a worry: I never touch the LLM's weights; I only learn to speak its input language.

Two flavors of LLM. For a decoder-only LLM, plain language modeling: prepend the projected queries, the LLM generates the caption, cross-entropy on the text. For an encoder-decoder LLM, prefix language modeling: split the text in two; concatenate the visual prompt with the prefix as the encoder's input, and make the suffix the decoder's target. Either way the gradient flows only into the querying module and its projection — both frozen models stay fixed.

Let me pin the concrete choices. 32 queries, each 768-dim — same as the module's hidden size, which I get from initializing from BERT-base. 32 is the budget: small enough to be a real bottleneck and cheap into the LLM, large enough to carry the few things any caption talks about. Cross-attention every *other* block, not every block, to limit the fresh random capacity relative to the pretrained self-attention/FFN. For the frozen image encoder I'll use a strong pre-trained ViT — a CLIP ViT-L/14 or an EVA-CLIP ViT-g/14 — and take the second-to-last layer's features (the very last layer is over-specialized to the encoder's own pretraining head; one layer back transfers a touch better). For the frozen LLM, an OPT model (decoder) or an instruction-tuned FlanT5 (encoder-decoder); cast their weights to FP16 (BF16 for FlanT5) since they're frozen and I just need forward passes — no accuracy cost observed. Optimizer AdamW, β = (0.9, 0.98), weight decay 0.05, cosine schedule, peak learning rate 1e-4 with a 2k-step warmup. Data: the same web-scale image-text mix used before — and I'll use the CapFilt trick, generating synthetic captions for the noisy web images with a captioner and keeping the best by image-text similarity, so the text the queries align to is actually about the image.

Let me write the core of stage one, the way it actually runs — queries through the module against the frozen image features, the three masked objectives, grounded in how this is implemented.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryingTransformer(nn.Module):
    def __init__(self, num_query=32, d_q=768, llm_dim=2048, transformer_block=None):
        super().__init__()
        # the bottleneck: a fixed set of learnable query vectors (these ARE parameters)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, d_q))
        # a BERT-initialized transformer; queries self-attend + cross-attend to image
        # features, text tokens share the self-attention layers; cross-attn every other block
        self.module = transformer_block            # has .bert(query_embeds, attention_mask,
                                                   #   encoder_hidden_states=image_feats, ...)
        self.temp = nn.Parameter(0.07 * torch.ones([]))   # contrastive temperature
        self.itm_head = nn.Linear(d_q, 2)                 # matched/unmatched per query
        self.vision_proj = nn.Linear(d_q, 256)            # for ITC similarity space
        self.text_proj = nn.Linear(d_q, 256)
        self.llm_proj = nn.Linear(d_q, llm_dim)           # stage-2 soft visual prompt


def stage1_losses(qformer, image_feats, text_tokens):
    B = image_feats.size(0)
    q = qformer.query_tokens.expand(B, -1, -1)            # [B, 32, 768]
    query_atts = torch.ones(q.shape[:-1], dtype=torch.long, device=q.device)

    # --- ITC: queries cross-attend image only (unimodal mask: no text leak) ---
    query_out = qformer.module.bert(query_embeds=q, encoder_hidden_states=image_feats,
                                    use_query=True)       # [B, 32, 768]
    image_feat = F.normalize(qformer.vision_proj(query_out), dim=-1)        # [B, 32, d]
    text_out = qformer.module.bert(text_tokens.input_ids,
                                   attention_mask=text_tokens.attention_mask)
    text_feat = F.normalize(qformer.text_proj(text_out[:, 0, :]), dim=-1)   # [B, d] (CLS)
    # all-pairs similarity of EACH image query to EACH text, then MAX over queries
    sim = torch.einsum("iqd,jd->ijq", image_feat, text_feat)     # [image B, text B, 32]
    sim_i2t = sim.max(-1)[0] / qformer.temp               # [B, B]
    sim_t2i = sim.permute(1, 0, 2).max(-1)[0] / qformer.temp
    labels = torch.arange(B, device=q.device)
    loss_itc = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2

    # --- ITG: causal-on-text mask; text must read the image THROUGH the queries ---
    dec_ids = text_tokens.input_ids.clone()
    dec_ids[:, 0] = qformer.module.dec_token_id           # [DEC] signals decoding
    labels_lm = dec_ids.masked_fill(dec_ids == qformer.module.pad_id, -100)
    attn = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    loss_itg = qformer.module.lm(query_embeds=q, input_ids=dec_ids, attention_mask=attn,
                                 encoder_hidden_states=image_feats, labels=labels_lm,
                                 causal_text=True).loss

    # --- ITM: bidirectional mask, hard negatives, per-query 2-class head, averaged ---
    pairs = mine_hard_negatives(sim_i2t.detach(), sim_t2i.detach(), image_feats, text_tokens)
    pair_queries = qformer.query_tokens.expand(pairs.image_feats.size(0), -1, -1)
    pair_query_atts = torch.ones(pair_queries.shape[:-1], dtype=torch.long,
                                 device=pair_queries.device)
    fused = qformer.module.bert(pairs.text_ids, query_embeds=pair_queries,
                                attention_mask=cat_mask(pair_query_atts, pairs.text_atts),
                                encoder_hidden_states=pairs.image_feats)
    logits = qformer.itm_head(fused[:, :q.size(1), :]).mean(dim=1)   # avg logits over queries
    loss_itm = F.cross_entropy(logits, pairs.match_labels)

    return loss_itc + loss_itg + loss_itm
```

And stage two — hand the queries to the frozen LLM as a soft visual prompt:

```python
def encode_visual_prompt(qformer, frozen_image_encoder, image):
    with torch.no_grad():
        image_feats = frozen_image_encoder(image)         # freeze only the image encoder
    q = qformer.query_tokens.expand(image_feats.size(0), -1, -1)
    query_out = qformer.module.bert(query_embeds=q, encoder_hidden_states=image_feats,
                                    use_query=True)        # [B, 32, 768]
    return query_out


def stage2_loss(qformer, frozen_image_encoder, frozen_llm, image, text,
                encoder_decoder=False):
    query_out = encode_visual_prompt(qformer, frozen_image_encoder, image)
    visual_prompt = qformer.llm_proj(query_out)            # [B, 32, llm_dim]

    if not encoder_decoder:
        # decoder LLM -> language-modeling loss, with prompt targets masked out
        text_embeds = frozen_llm.embed(text.input_ids)     # frozen embedding table
        inputs = torch.cat([visual_prompt, text_embeds], dim=1)
        prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                                 device=visual_prompt.device)
        attn = torch.cat([prompt_atts, text.attention_mask], dim=1)
        prompt_labels = torch.full(visual_prompt.shape[:-1], -100,
                                   dtype=torch.long, device=visual_prompt.device)
        labels = torch.cat([prompt_labels, text.labels], dim=1)
        # no torch.no_grad() around the frozen LLM call: gradients must flow to inputs
        return frozen_llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss

    # encoder-decoder LLM -> prefix-LM: visual prompt + prefix into encoder, suffix target
    prefix_embeds = frozen_llm.embed(text.prefix_ids)
    enc_inputs = torch.cat([visual_prompt, prefix_embeds], dim=1)
    prompt_atts = torch.ones(visual_prompt.shape[:-1], dtype=torch.long,
                             device=visual_prompt.device)
    enc_attn = torch.cat([prompt_atts, text.prefix_attention_mask], dim=1)
    suffix_labels = text.suffix_ids.masked_fill(text.suffix_ids == qformer.module.pad_id, -100)
    return frozen_llm(inputs_embeds=enc_inputs, attention_mask=enc_attn,
                      labels=suffix_labels).loss
```

So the causal chain, end to end: I wanted to stop relearning what's already trained, so I freeze a strong vision encoder and a strong LLM; freezing the LLM means it never adapts to vision, so all alignment must happen in a small bridge; the generation-only objective that prior frozen-LLM methods use is too weak, because a fluent frozen LM's language prior absorbs the loss without forcing the visual representation to align — so I add a representation-learning stage *first*; to turn the frozen encoder's variable, mostly-irrelevant feature grid into a small fixed signal I use a set of learned queries that cross-attend the features (the Perceiver/DETR device), giving a 32-vector bottleneck that's forced to keep only text-relevant content; I supervise it with three mask-differentiated objectives in one shared module — contrastive (unimodal mask, max-over-queries similarity, in-batch negatives), grounded generation (causal-on-text mask, which routes all the generation-needed info through the queries), and matching (bidirectional mask, per-query classifier averaged, hard negatives) — and only then, in stage two, project the now-aligned queries into the frozen LLM's input space as soft visual prompts and train with the LLM's own (prefix-)language-modeling loss, the bottleneck feeding the LLM exactly the visual information it needs and nothing more.
