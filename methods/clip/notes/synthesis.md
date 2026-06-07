# CLIP — Synthesis Notes (Phase 1.5)

## The pain point / research question
Computer vision in 2020 still pretrains on *fixed-label* crowd-curated datasets (ImageNet 1000 classes; JFT-300M 18291 classes). Two structural limits:
1. **Supervision is capped by the label vocabulary.** Every new visual concept needs new labeled data in the canonical "1-of-N gold label" format. Scaling supervision = paying annotators.
2. **The output head is a static softmax.** No mechanism for dynamic outputs → no real zero-shot. A model trained to predict 1000 ImageNet classes cannot name "a German shepherd doing agility" or transfer to OCR without a new head + new data.

Meanwhile NLP had escaped this: task-agnostic objectives (autoregressive / masked LM) scale across orders of magnitude of compute/data, and "text-to-text" gives zero-shot transfer with no task-specific heads (GPT-2/GPT-3). The question: **can web-scale pretraining directly from natural language give vision the same task-agnostic, zero-shot capability?**

A solution must: (a) draw supervision from the open vocabulary of natural language (not a fixed label set), (b) be cheap enough to scale to hundreds of millions of pairs, (c) produce a model whose output can be *specified at test time by language* (dynamic classifier).

## Load-bearing ancestors (verified against paper)

- **mori1999image** — image→word transformation; earliest predict-words-from-paired-text-for-image-retrieval. Establishes the *idea* of natural language supervision; tiny scale, vector-quantized features.
- **quattoni2007learning** — data-efficient representations via manifold learning in weight space of caption-word classifiers.
- **joulin2016learning** — modernized this: CNN (AlexNet) trained to predict **bag-of-words** of YFCC100M title/description/hashtag metadata (multi-label classification). Showed text-predicted features rival ImageNet pretraining on transfer. KEY: this is the **BoW baseline** CLIP starts from. Predicting the *set* of words, not the *sequence*.
- **li2017learning (Visual N-Grams)** — extended to predicting phrase n-grams (dictionary of 142,806 n-grams, differential Jelinek-Mercer smoothing). FIRST to do **zero-shot transfer to standard image-classification datasets** by scoring class-name n-grams. But only **11.5% ImageNet zero-shot** — proof of concept, far below even classic CV (50%) or SOTA (88.4%). The best reference point CLIP contextualizes against.
- **desai2020virtex** — VirTex: train image CNN + text transformer from scratch to **generate the caption** (autoregressive LM). Transformer-based; small (MSCOCO ~100k). This is the **caption-generation baseline** CLIP first tried and abandoned.
- **bulent2020learning (ICMLM)**, **zhang2020contrastive (ConVIRT)** — ConVIRT is the direct parent: contrastive (image,text) on **medical** images, with non-linear projection heads, a text sampling function t_u, image augmentation t_v, and ImageNet/pretrained init. CLIP = "a simplified version of ConVIRT trained from scratch."
- **mahajan2018exploring** / **kolesnikov2019large (BiT)** / **dosovitskiy2020image (ViT)** — the weakly-supervised "pragmatic middle ground": predict Instagram hashtags (3.5B images, 19 GPU-yrs) or JFT-300M noisy labels. Big gains but: (a) carefully *designed and limited* to 1000/18291 classes; (b) static softmax → no dynamic outputs. Scale is huge but supervision is still fixed-label.
- **tian2019contrastive (CMC)** — contrastive objectives learn better representations than the equivalent *predictive* objective. Motivation to swap predict→contrast.
- **chen2020generative (Image GPT)** — generative pixel models learn good reps but need **>1 order of magnitude more compute** than contrastive for equal performance. Reinforces: generation is expensive.
- **sohn2016improved (N-pair loss)** — the batch-construction-as-classification objective: 1 positive vs N−1 negatives, softmax cross-entropy. Origin of the loss.
- **oord2018representation (InfoNCE / CPC)** — popularized the same objective as InfoNCE; it's a lower bound on mutual information. The math template CLIP uses.
- **wu2018unsupervised (instance discrimination)** — non-parametric softmax over instances with temperature τ; τ=0.07 init comes from here.
- **bachman2019learning / chen2020simple (SimCLR)** — introduced/popularized the **non-linear projection head** between representation and contrastive space. CLIP *removes* it (uses linear only) — found no efficiency difference, speculates nonlinear head co-adapts with image-only SSL details.
- **lei2015predicting / elhoseiny2013write ("Write a classifier")** — generate the **weights of a classifier from a text description**. This is the lens for CLIP's zero-shot: the text encoder is a hypernetwork generating linear-classifier weights from class names.
- **larochelle2008zero (zero-data learning)**, **liu2018generating / radford2018improving (GPT-1) / radford2019language (GPT-2)** — zero-shot transfer as a measure of *task learning*; task learning emerges as a side effect of scaled generative pretraining. Frames why zero-shot is the right yardstick.
- **vaswani2017attention** (Transformer), **he2016deep** (ResNet), **tan2019efficientnet** (compound scaling), **kingma2014adam / loshchilov2017decoupled** (AdamW), **radford2019language** (GPT-2-style text transformer + BPE sennrich2015neural).

## The derivation chain (insight → method)

### 1. Why not predict captions (the wall)
Start where VirTex/joulin left off: train image+text encoders jointly to predict the text. Two flavors:
- **Caption generation** (VirTex-style autoregressive LM over the exact word sequence).
- **BoW prediction** (joulin-style: predict the set of words).

Observed (diagnostic finding, Fig 2): a 63M-param transformer LM (already 2× the compute of its ResNet-50 image encoder) learns ImageNet recognition **3× slower** than the much simpler BoW baseline. WHY: predicting the *exact words* is needlessly hard — the same image co-occurs with wildly varied descriptions, comments, hashtags. The model burns capacity modeling the full distribution of *how* the caption is phrased (word order, function words, style), most of which is irrelevant to the visual content. Generation also pays the chen2020generative tax: generative >1 order of magnitude more compute than contrastive.

### 2. The relaxation: predict *which* text, not *what* text
Insight: we don't need the words; we need the *association*. Replace "produce the caption" with the easier proxy "**of the texts in this batch, which one goes with this image?**" This throws away all the modeling effort spent on phrasing. Empirically (Fig 2): swapping BoW-prediction → contrastive gives a further **4× efficiency**. So total ~12× over caption generation.

WHY cheaper, made precise:
- Caption generation cost: per image, a softmax over vocab V at every one of L positions → O(L·V) prediction targets, and the model must get the *whole sequence* right.
- Contrastive cost: per image, **a single softmax over the N texts in the batch** (N−1 negatives). One classification, no per-token decoding, no decoder. The negatives come *for free* from the other examples already in the batch — no extra forward passes, no generation. The supervision signal is "is this the matching text — yes/no among N" instead of "reproduce these L tokens."

### 3. The objective falls out: symmetric InfoNCE with temperature
Given a batch of N (image, text) pairs, embed all images → I_e[N,d], all texts → T_e[N,d], L2-normalize both (so dot product = cosine similarity, bounded in [−1,1]). Similarity matrix S = I_e · T_eᵀ, shape [N,N]. Diagonal = the N true pairs; off-diagonal = N²−N impostors.

Treat **each row as a classification**: row i (image i) should pick column i among N texts → cross-entropy with label i over softmax(row i). Symmetrically **each column** (text j) should pick row j among N images. The objective is the average:
  L = ½ [ CE(softmax over texts | each image) + CE(softmax over images | each text) ].
This is exactly the multi-class N-pair loss (sohn2016) / InfoNCE (oord2018), made **symmetric** because the matching is bidirectional (image↔text, not image→fixed-classes). labels = arange(N). Pseudocode (Fig 3):
```
logits = I_e @ T_e.T * exp(t)
loss = (CE(logits, arange(N), axis=0) + CE(logits, arange(N), axis=1))/2
```

WHY temperature, and WHY learned + log-parameterized + clipped:
- Cosine sims live in [−1,1]. Feeding [−1,1] straight into softmax gives an almost-flat distribution (max logit gap = 2) → tiny gradients → no learning. Need to scale: logits = S/τ, equivalently S·exp(t) with t = −log τ. The scale controls softmax sharpness.
- τ is delicate and interacts with everything; tuning it as a hyperparameter is wasteful. Make it **learnable** and let SGD set the sharpness.
- Parameterize as t = log(1/τ) and use **exp(t)** so the multiplier is always positive (no need to constrain τ>0); gradient is well-behaved. Init t = log(1/0.07) (the 0.07 from wu2018).
- **Clip** exp(t) ≤ 100: without a ceiling, the optimizer can drive the scale up unboundedly (sharper softmax always lowers train loss once pairs separate), which causes training instability. Cap the inverse-temperature at 100.

### 4. The reuse: zero-shot classification via prompt embeddings
Pretraining already learned "does this text describe this image?" To classify image x into classes {c_1..c_K} with no training:
- Turn each class name into text ("A photo of a {c_k}."), embed with the text encoder → weight vectors w_k = T_e(prompt_k), L2-normalized.
- Embed the image → I_e(x), L2-normalized.
- logits_k = exp(t) · I_e(x)·w_k ; softmax → predict argmax.

This **is** a multinomial logistic regression with L2-normalized inputs, L2-normalized weights, no bias, temperature scaling — except the weights are *generated by the text encoder from language* (lei2015/elhoseiny2013 "write a classifier" / hypernetwork view). The text encoder synthesizes a classifier on the fly → dynamic outputs, open vocabulary. Cache the K text embeddings once → amortized cost ~ one matmul per image. Every pretraining step is then secretly optimizing a proxy 32,768-way (=batch size) 1-example-per-class classifier defined by language.

### 5. Prompt engineering (why)
- **Polysemy**: a bare class name ("crane", "boxer") is ambiguous without context; the text encoder has no way to disambiguate. ImageNet even contains both senses of "crane."
- **Distribution gap**: pretraining text is almost never a single word — it's a sentence. A bare label is off-distribution for the text encoder. The template "A photo of a {label}." closes the gap (+1.3% ImageNet). Task-specific templates ("a satellite photo of a {label}", "a type of pet") add task context.
- **Ensembling** over many templates, averaged **in embedding space** (not probability space) so a single averaged text vector per class → no extra inference cost. ~+3.5% on ImageNet from 80 prompts.

## Design-decision → why table

| Decision | Why this, not the alternative |
|---|---|
| Contrastive matching, not caption generation | Generation models phrasing (O(L·V), high variance, generative compute tax); matching is one softmax over batch, 12× more efficient. |
| Symmetric loss (image→text AND text→image) | Pairing is bidirectional; both encoders supervise each other; image-side and text-side both must retrieve. |
| Negatives from in-batch examples | Free negatives; no extra forward passes / no memory bank needed at batch=32768. |
| Very large batch (32,768) | Contrastive signal quality scales with #negatives; each row is a 32768-way classification. |
| L2-normalize embeddings | Makes dot product = cosine; comparable scale across examples; removes magnitude from the matching decision. |
| Learned temperature, log-parameterized (exp(t)), clipped at 100 | Cosine∈[−1,1] → flat softmax without scaling; learning τ avoids a sensitive hyperparameter; exp keeps it positive; clip prevents runaway sharpening → instability. |
| **Linear** projection to embed space (drop SimCLR nonlinear head) | No efficiency difference observed at this scale; nonlinear head likely co-adapts with image-only SSL specifics, not needed here. |
| Train from scratch (no ImageNet/pretrained-LM init) | 400M pairs → overfitting not a concern; init unnecessary; simplifies ConVIRT. |
| Drop ConVIRT's t_u (sentence sampling) and heavy t_v augmentation; only random square crop | Most WIT pairs are a single sentence (t_u pointless); large data makes heavy augmentation unnecessary. |
| Image encoder: ResNet-D + blur-pool + **attention pooling** (QKV, query=global-avg-pool) | ResNet-D & antialias = known improvements; attention pool replaces global-avg-pool with a learned transformer-style aggregation → richer pooled feature. |
| Also ViT image encoder (+extra LN on patch+pos embeds) | ViT is more compute-efficient than convnets on large data (dosovitskiy); extra LN stabilizes. |
| Text encoder: GPT-2-style Transformer, BPE 49152 vocab, max len 76, [EOS] activation = text feature, **masked (causal) self-attn** | Reuse NLP's scalable arch; causal mask keeps option to init from / add LM objective later; [EOS] is the sequence summary. |
| Scale width=depth=resolution together (EfficientNet-style), text width only | Compound scaling beats single-axis; CLIP less sensitive to text capacity so don't waste compute there. |
| AdamW, cosine LR schedule, decoupled WD on non-gain/bias weights | Standard scalable optimization; WD regularizes, decoupled from LR. |
| Prompt template + ensembling, averaged in embedding space | Closes single-word↔sentence gap, disambiguates polysemy, free at inference when amortized. |

## Canonical implementation (1.4)
- `code/openai-clip/clip/model.py` — `CLIP` module: `encode_image`, `encode_text`, `forward` returns `logits_per_image/text = logit_scale.exp() * I_e @ T_e.T`. `logit_scale = nn.Parameter(ones*log(1/0.07))`. Text feature = activation at argmax token (=EOS, highest id) @ text_projection. ViT/ModifiedResNet image encoders.
- `code/open_clip_loss.py` — `ClipLoss`: labels=arange(N); total = (CE(logits_per_image,labels)+CE(logits_per_text,labels))/2. This is the training loss the OpenAI repo doesn't ship (inference-only release); OpenCLIP is the canonical training reference.
- Paper Fig 3 pseudocode (exact, extracted): the 25-line numpy sketch — this is the ground truth for `answer.md` / `reasoning.md` final code.
- Zero-shot: `clip.tokenize(["A photo of a {c}." ...])` → `encode_text` → normalize → text weight matrix; `encode_image` → normalize; `logit_scale.exp() * img @ text.T` → softmax.

## Scaffold (pre-method skeleton) — corresponds piece-for-piece to final code
Pre-method vocab: image encoder, text encoder, linear projections, L2-normalize, a batch of paired (image,text), an optimizer, cross-entropy. One big empty slot = "the objective that ties image and text reps together" + "how to use the trained encoders to classify with no labels." Stubs: `class DualEncoder` (two backbones + two linear projections, forward returns embeddings — TODO the objective), `def training_objective(img_emb, txt_emb)` → pass, `def build_classifier_from_descriptions(...)` → pass.

## In-frame cautions
- Never name "CLIP/the paper/Radford et al." in context.md/reasoning.md as a citation. May name CLIP as the method being built in answer.md.
- Prior-art citations (oord2018, sohn2016, joulin2016, zhang2020 ConVIRT, VirTex, Visual N-Grams, etc.) stay — that's the lineage.
- No proposed-method eval results (no 76.2% ImageNet zero-shot as a *win*). The Fig-2 *diagnostic* efficiency findings (caption 3× slower than BoW; contrastive 4× over BoW) are motivating/diagnostic → allowed in context+reasoning as the wall that drove the design.
