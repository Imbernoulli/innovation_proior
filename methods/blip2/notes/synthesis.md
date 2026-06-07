# BLIP-2 synthesis (grounded in src/*.tex + LAVIS blip2_qformer.py)

## Pain point / research question
VLP at the time = end-to-end training of large image+text encoders on huge data → very expensive, and cannot reuse off-the-shelf unimodal models (LLMs). Want generic + compute-efficient VLP that bootstraps from FROZEN pretrained vision encoder + FROZEN LLM. Freezing avoids catastrophic forgetting + saves compute. But: LLM never saw images → freezing it makes vision-language ALIGNMENT hard. Existing frozen-LLM methods (Frozen, Flamingo) use only an image-to-text generation loss, which is shown insufficient to bridge the modality gap.

## Key ancestors
- CLIP/ALIGN: dual-encoder contrastive (ITC). Unimodal alignment. No fusion → no VQA/generation.
- ALBEF/BLIP: fusion encoders, multi-objective ITC+ITM+(M)LM, hard-neg mining, momentum queue. End-to-end → expensive.
- Frozen (Tsimpoukelli 2021): finetune image encoder, outputs as soft prompts to frozen LM, image-caption loss.
- Flamingo: insert gated cross-attn into frozen LM, train on billions of pairs, LM loss.
- LiT: frozen image encoder for contrastive.
- Perceiver/DETR: learned latent queries cross-attend variable features → fixed tokens.

## The method (Q-Former)
A lightweight transformer (init from BERT-base, 188M params) that is an information bottleneck. 32 learnable query embeddings (dim 768) as input to an "image transformer". Queries:
- self-attend to each other AND to text (shared self-attn layers)
- cross-attend to frozen image features (cross-attn inserted every OTHER block)
Output Z = 32×768, MUCH smaller than frozen image feats (257×1024 for ViT-L). This bottleneck + objectives force queries to extract text-relevant visual info.

Two transformer submodules SHARE self-attention layers: an image transformer (queries) and a text transformer (encoder+decoder). Different self-attention MASKS control query↔text interaction per objective.

### Stage 1 (rep learning, frozen image encoder) — 3 objectives (from BLIP), same input/params, different masks:
- ITC: align Z with text rep t (=[CLS] output). Z has 32 embeds → compute pairwise sim of each query output with t, take MAX as image-text sim. Temperature. Unimodal mask (queries & text can't see each other → no info leak). In-batch negatives (frozen encoder → big batch fits).
- ITG: image-grounded text generation. Multimodal CAUSAL mask (UniLM-style): queries attend each other but NOT text; each text token attends all queries + its previous text tokens. Replace [CLS] with [DEC] as first token. Forces queries to capture all info needed to generate text.
- ITM: binary match/no-match. Bidirectional mask (all queries+text attend each other). Each output query embed → 2-class linear classifier → logit; average logits over queries. Hard-neg mining (ALBEF/BLIP).

### Stage 2 (generative learning, frozen LLM):
FC layer projects Z into LLM text-embedding dim. Projected query embeds PREPENDED to input text embeds → soft visual prompts. Q-Former already extracts language-informative visual rep → info bottleneck feeds useful info, reduces LLM burden, mitigates catastrophic forgetting.
- decoder LLM (OPT): language modeling loss.
- encoder-decoder LLM (FlanT5): prefix LM loss — split text; prefix concatenated with visual rep → encoder; suffix = decoder target.

## Hyperparams (grounded)
- 32 queries, dim 768, Q-Former 188M, init BERT-base, cross-attn random init.
- Frozen image enc: CLIP ViT-L/14 or EVA-CLIP ViT-g/14; use 2nd-last layer.
- Frozen LLM: OPT (decoder), FlanT5 (enc-dec). FP16 (BF16 for FlanT5).
- Data: 129M images (COCO, VG, CC3M, CC12M, SBU, LAION400M-115M). CapFilt synthetic captions.
- 250k stage-1 steps, 80k stage-2. AdamW β1=0.9 β2=0.98 wd=0.05, cosine, peak lr 1e-4, 2k warmup, min lr stage2 5e-5. 224×224 images.

## Code structure (LAVIS): query_tokens = nn.Parameter expand per batch; sim_q2t = matmul; sim_i2t = max over queries / temp; ITM avg over queries; ITG attention_mask = cat([query_atts, text_atts]); total = itc+itm+lm.
