# GPT-2 synthesis (verified vs OpenAI report PDF pages 2-5 + openai/gpt-2 src/model.py)
# NO arXiv. Source: "Language Models are Unsupervised Multitask Learners" (OpenAI tech report).

## Verified core thesis (Section 2, from PDF)
- LM = unsupervised distribution estimation over symbol sequences. Factorize: p(x) = prod_{i=1}^n p(s_n | s_1,...,s_{n-1}). (Eq 1)
- Learning a single task = estimate p(output | input). A general system that does many tasks should condition on the task too: p(output | input, task). Task conditioning can be architectural (task-specific encoders/decoders, Kaiser 2017) or algorithmic (MAML inner/outer loop). But (McCann 2018, MQAN) language provides a flexible way to specify task/input/output as a sequence of symbols, e.g. translation example = (translate to french, english text, french text); reading comprehension = (answer the question, document, question, answer).
- Key: supervised objective = same as unsupervised objective but evaluated on a subset of the sequence; so the global minimum of the unsupervised objective is also the global minimum of the supervised one. If a sufficiently large LM trained on enough varied text learns p(output|input,task) implicitly, it can perform tasks ZERO-SHOT — no parameter or architecture modification.
- Demonstrate downstream tasks in zero-shot setting. Natural demonstrations of tasks (e.g. English-French translation pairs) occur throughout web text.

## Training dataset: WebText (Section 2.1, verified)
- Prior LMs trained on single domain (news, Wikipedia, fiction books). Want large + diverse to collect natural demonstrations.
- Common Crawl is huge but has data-quality issues ("content mostly unintelligible").
- Instead: scrape only human-curated/filtered pages. Heuristic: all outbound links from Reddit with at least 3 karma (proxy for "other users found it interesting/educational/funny").
- WebText = text subset of these 45 million links. Extract HTML text with Dragnet + Newspaper content extractors.
- Preliminary version: NO links created after Dec 2017; after de-duplication + heuristic cleaning, slightly over 8 million documents, total 40 GB.
- Removed all Wikipedia documents (common data source for other datasets; avoid train/test overlap).

## Input representation: byte-level BPE (Section 2.2, verified)
- General LM should compute/generate probability of ANY string. Current large LMs use lowercasing, tokenization, OOV tokens that restrict modelable strings.
- UTF-8 bytes elegantly handle any string (Gillick 2015), but byte-level LMs not competitive with word-level on large datasets (1B Word Benchmark; observed same gap in own attempts).
- BPE (Sennrich 2015): middle ground, interpolates word-level (frequent) and char-level (infrequent). Despite name, reference BPE operates on UNICODE code points not bytes -> base vocab >130,000 for all Unicode before merges. Too large vs 32k-64k typical.
- Byte-level BPE only needs base vocab of size 256. BUT directly applying BPE to byte sequence gives suboptimal merges (greedy frequency heuristic). Observed BPE including many versions of common words like "dog" since occur as "dog." "dog!" "dog?" -> wastes vocab slots + model capacity.
- Fix: PREVENT BPE from merging across character categories for any byte sequence. Add EXCEPTION for spaces (improves compression efficiency, minimal word fragmentation across vocab tokens).
- Result: combine word-level benefits + byte-level generality. Can assign probability to any Unicode string, evaluate on any dataset regardless of preprocessing/tokenization. Vocab 50,257.

## Model (Section 2.3, verified)
- Transformer (Vaswani 2017), largely follows OpenAI GPT (Radford 2018), with modifications:
  - Layer normalization moved to the INPUT of each sub-block (like pre-activation residual network, He 2016) — i.e. pre-LN.
  - An ADDITIONAL layer normalization added AFTER the final self-attention block.
  - Modified initialization accounting for accumulation on the residual path with depth: scale weights of residual layers at init by factor 1/sqrt(N), N = number of residual layers.
  - Vocabulary 50,257. Context size increased from 512 to 1024 tokens. Batch size 512.

## Model sizes (Table 2, verified)
- 117M params: 12 layers, d_model 768. (= original GPT)
- 345M: 24 layers, d_model 1024. (= largest BERT size)
- 762M: 36 layers, d_model 1280.
- 1542M: 48 layers, d_model 1600. (= GPT-2)
- Log-uniformly spaced sizes. LR manually tuned for best perplexity on 5% held-out WebText. All models still UNDERFIT WebText; held-out perplexity would improve with more training.

## Diagnostic / motivating findings (about existing systems, in-scope)
- Single-task supervised systems are brittle; ML systems are "narrow experts". Want general systems.
- Common Crawl quality issue (motivates WebText).
- Byte-level worse than word-level on large datasets (motivates byte-BPE fix).
- Wikipedia overlap (motivates removal).

## Canonical implementation (openai/gpt-2 src/model.py, verified)
- hparams: n_vocab=0(set to 50257), n_ctx=1024, n_embd=768, n_head=12, n_layer=12 (base).
- norm(x): normalize to mean 0 std 1 over last axis, then diagonal affine (gain g, bias b). LayerNorm.
- gelu(x): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
- attention_mask: lower-triangular (causal), 1s in lower triangle from lower-right corner; prevents attending to future.
- attn: c_attn conv1d -> q,k,v; split heads; scaled dot product w * 1/sqrt(d_k); mask; softmax; merge heads; c_proj.
- mlp: h = gelu(conv1d(x, 'c_fc', 4*n_embd)); conv1d(h, 'c_proj', nx). 4x inner.
- block (PRE-LN): a = attn(norm(x)); x = x + a; m = mlp(norm(x)); x = x + m.
- model: h = wte[tokens] + wpe[positions]; loop blocks; h = norm(h) [ln_f, the extra final LN]; logits = h @ wte^T (tied embedding).

## Design decisions -> why
- Zero-shot via p(output|input,task) in language: tasks/inputs/outputs all expressible as token sequences (per MQAN); a strong enough LM that models p(output|input,task) needs no task-specific heads or fine-tuning. Supervised objective is a subset-evaluation of the unsupervised one, so the LM optimum subsumes the task optima.
- WebText (Reddit-curated links): need diverse natural task demonstrations; Common Crawl too noisy; human upvote is a cheap quality filter; removing Wikipedia avoids contaminating downstream eval.
- Byte-level BPE w/ category-restricted merges: must assign probability to ANY string (no UNK, no preprocessing) so a single LM evaluates on any dataset; byte base (256) keeps vocab small but raw byte-BPE wastes slots on punctuation variants of words -> forbid cross-category merges (keep space exception for compression). Best of word-level (efficiency) + byte-level (universality).
- Pre-LN (LN at sub-block input): the original post-LN puts layernorm on the residual path, which interacts badly with depth/optimization; moving LN to the sub-block input keeps a clean identity residual path (like pre-activation ResNet), stabilizing very deep training. Needed because GPT-2 is much deeper (up to 48 layers).
- Extra final LN after last block: with pre-LN, the output of the residual stack is unnormalized; a final LN normalizes it before the output projection.
- Residual init scaling 1/sqrt(N): each of the N residual additions accumulates variance along the residual path; scaling residual-layer init by 1/sqrt(N) keeps the accumulated signal magnitude controlled at initialization for deep stacks.
- Context 1024 (from 512): capture longer-range dependencies; web documents are long.
- Scale model size (117M->1.5B): zero-shot task performance increases log-linearly with capacity; the whole thesis is that capacity + data unlocks implicit multitask learning.
```
