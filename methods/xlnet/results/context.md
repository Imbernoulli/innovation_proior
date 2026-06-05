## Research question

We have unlabeled text in essentially unlimited quantity and a small amount of labeled data for each downstream natural language understanding (NLU) task — classification, natural language inference, reading comprehension, span-extraction question answering, document ranking. The dominant recipe is to *pretrain* a deep neural network on the raw text with a self-supervised objective, then *finetune* it on each task. The open question is: **what self-supervised objective should we pretrain with?**

A good objective must satisfy several things at once that, as of now, are in tension:

1. **It should expose every token to deep bidirectional context.** Downstream tasks (e.g. deciding whether a span answers a question) need a representation of a token that depends on what comes *both before and after* it. An objective that only ever conditions on one side leaves a gap between what is learned in pretraining and what the task needs.
2. **It should not introduce artifacts at pretraining that are absent at finetuning.** If pretraining feeds the network special placeholder symbols that never appear in real downstream text, the network is adapted to inputs it will never see again, and finetuning has to correct for that mismatch.
3. **It should model the joint distribution of the tokens it predicts honestly** — without assuming the predicted tokens are mutually independent given the context. Natural language is full of high-order, long-range dependencies (a multi-token name, an agreement relation), and an objective that predicts each target in isolation cannot capture the dependency *between* two targets.
4. **It should stay inside a principled density-estimation framework** so that ongoing progress in language modeling (longer context, better architectures) transfers directly into pretraining.

No existing objective hits all four. The two leading families each get some and miss others, and the precise shape of what each one misses is what a better objective would have to repair.

## Background

**Pretraining-then-finetuning for NLP.** Learning reusable text representations from large unlabeled corpora and transferring them to downstream tasks has been repeatedly successful (Dai & Le 2015, semi-supervised sequence learning with recurrent nets; McCann et al. 2017, representations from machine-translation encoders; Peters et al. 2018, ELMo; Radford et al. 2018, the GPT generative-pretraining-then-finetuning protocol). Two families of self-supervised objective dominate: **autoregressive (AR) language modeling** and **autoencoding (AE) denoising**.

**Language modeling is density estimation.** Estimating p(x) for a text sequence x = (x_1,…,x_T) is a long-standing unsupervised problem. The standard handle is the chain rule, which factorizes the joint into a product of conditionals. This is exact and assumption-free; the only design choice is the *order* in which the chain rule is applied. Conventional language models apply it in the fixed forward order p(x) = Π_t p(x_t | x_{<t}) (or the backward order). Progress in this area — deeper Transformers, longer effective context — has been rapid, which makes it attractive to have a pretraining objective that lives in this framework so the progress carries over.

**Self-attention Transformer backbone (Vaswani et al. 2017).** Tokens are embedded; each layer computes multi-head scaled-dot-product attention, softmax(QKᵀ/√d_head)·V, followed by a position-wise feed-forward network, each sublayer wrapped in a residual connection and layer normalization. Order information enters through positional encodings added to the embeddings. A causal attention mask makes the model autoregressive; no mask makes every position see every other. The mask is a free knob: *which positions a given position may attend to is decided entirely by the mask*, independent of the actual sequence layout.

**Transformer-XL (Dai et al. 2019), the state of the art in AR language modeling.** Two ideas matter here.
- *Segment recurrence.* Process a long text in fixed-length segments; cache the hidden states of the previous segment and let the current segment attend to them as a read-only memory (no gradient flows into the cache). This extends the effective context far beyond one segment and improves optimization.
- *Relative positional encoding.* Absolute position embeddings break when you reuse states across segments (position 1 of segment 2 would collide with position 1 of segment 1). Instead, the attention logit between query position i and key position j is decomposed so that position enters only through the *relative distance* i−j:
  A_{ij} = (q_i + u)ᵀ k_j + (q_i + v)ᵀ W_R r_{i−j},
  where q_i, k_j are the content-based query/key, r_{i−j} is a sinusoidal embedding of the relative distance, W_R projects it, and u, v are two global learnable bias vectors (one for the content term, one for the position term) that replace the absolute-position-dependent query bias. The first term is content-to-content; the second is content-to-relative-position. This relative scheme is also a better inductive bias and generalizes well.

**Permutation / order-agnostic density estimators (Germain et al. 2015, MADE; Uria et al. 2016, orderless NADE).** These train a single model to factorize the density under *many* factorization orders rather than one fixed order, by sampling an order per example and masking the network accordingly. Their motivation is purely better density estimation — baking an "orderless" inductive bias into the estimator — and they lean on the implicit position-awareness of an MLP architecture. They are essentially *orderless*: the model is not told which position it is currently predicting beyond what the masking implies, which for a feed-forward estimator degenerates toward a bag-of-words view of the context.

**The motivating diagnostic — what each existing objective provably cannot capture.** Consider the sentence [New, York, is, a, city] and ask which target–context dependencies a pretraining objective actually trains. A fixed forward AR model can train the dependency of "York" on "New" but never the dependency of "New" on "York", because "York" is to its right. A denoising AE that masks both "New" and "York" and reconstructs each from the unmasked remainder {is, a, city} trains "New" given {is, a, city} and "York" given {is, a, city}, but — because it factorizes the joint over the two masked tokens as a product — it *never* trains "York" given "New": the dependency between the two predicted tokens is structurally absent from its loss. These are not empirical accidents; they fall out of the form of each objective, and they are exactly the gaps a better objective should close.

## Baselines

**Autoregressive (AR) language modeling** (Dai & Le 2015; Peters et al. 2018; Radford et al. 2018). Pretrain by maximizing the forward-factorized likelihood
  max_θ log p_θ(x) = Σ_{t=1}^T log p_θ(x_t | x_{<t}) = Σ_t log [ exp(h_θ(x_{1:t−1})ᵀ e(x_t)) / Σ_{x′} exp(h_θ(x_{1:t−1})ᵀ e(x′)) ],
where h_θ(x_{1:t−1}) is a context representation (RNN or Transformer) and e(·) the token embedding. *Strengths:* it is exact density estimation via the universal product rule, with no independence assumption and no artificial input symbols, so it has no pretrain–finetune mismatch and inherits all language-modeling progress. *Gap it leaves:* h_θ(x_{1:t−1}) is conditioned only on the left context, so the model never learns bidirectional representations. ELMo (Peters et al. 2018) tries to recover the other direction by training a separate backward LM and concatenating the two directions' features, but the forward and backward models are independent and combined only shallowly at the top, so there is no deep joint reasoning over both sides.

**Autoencoding / denoising (BERT, Devlin et al. 2018), the current state of the art for NLU pretraining.** Corrupt x into x̂ by replacing a fraction (≈15%) of tokens with a special symbol [MASK]; let x̄ be the set of masked tokens; reconstruct them:
  max_θ log p_θ(x̄ | x̂) ≈ Σ_{t=1}^T m_t · log p_θ(x_t | x̂) = Σ_t m_t · log [ exp(H_θ(x̂)_tᵀ e(x_t)) / Σ_{x′} exp(H_θ(x̂)_tᵀ e(x′)) ],
where m_t = 1 marks a masked position and H_θ maps the length-T sequence to a sequence of hidden vectors. Because reconstruction is not density estimation, every H_θ(x̂)_t may attend to *both* sides — this is how BERT gets bidirectional context, and it is why it beats AR pretraining on NLU. *Gaps it leaves, all visible in that one equation:*
- *Independence assumption.* The "≈" is doing real work: the joint p(x̄ | x̂) is replaced by the product Π over masked positions, i.e. all masked tokens are reconstructed independently given the unmasked ones. Long-range dependency between two masked tokens (the "New"/"York" case above) cannot be modeled.
- *Input noise / pretrain–finetune discrepancy.* [MASK] appears throughout pretraining but never in downstream data. Replacing some [MASK]s with the original token to soften this does not fix it, because the replacement must stay rare — otherwise the objective becomes trivial (the model can just copy the visible answer).
- It also sits outside the density-estimation framework, so language-modeling progress does not transfer cleanly.

**Order-agnostic density estimators (MADE, orderless NADE).** Train one model over many factorization orders via masking. They demonstrate that a *single shared model can be trained under many orders at once* — the mechanism we will want. But their goal and their inductive bias differ: they aim at orderless density estimation and are position-unaware (degenerating toward bag-of-words for an MLP), whereas an NLU pretrainer needs to remain *order-aware* (positions matter for language) while exploiting the multi-order training trick to get bidirectional context.

## Evaluation settings

The natural yardsticks:
- **GLUE** (Wang et al. 2019): a suite of nine NLU tasks (e.g. MNLI, SST-2, QNLI, QQP, RTE, MRPC, CoLA), evaluated via a held-out submission server; single-task and multi-task, single-model and ensemble protocols.
- **Reading comprehension:** RACE (Lai et al. 2017), ~100K questions from English exams with long passages (avg >300 tokens), a long-context test; SQuAD 1.1 (Rajpurkar et al. 2016, always-answerable) and SQuAD 2.0 (Rajpurkar et al. 2018, with unanswerable questions, needing a joint answerability + span-extraction loss).
- **Text classification:** IMDB, Yelp-2/5, DBpedia, AG, Amazon-2/5 (following Zhang et al. 2015).
- **Document ranking:** ClueWeb09-B (Dai 2018 setting), reranking the top-100 retrieved documents for TREC Web Track queries; probes the quality of word embeddings via a kernel-pooling ranker, often without finetuning.
- **Protocol:** finetune the pretrained encoder per task with a light task head; sequence length up to 512; metrics are task-native (accuracy, F1/EM, MRR/NDCG). Compare against AR-LM and denoising-AE pretraining under matched data and model size.

## Code framework

The available primitives are a token-embedding lookup, a Transformer-XL relative-attention block (multi-head relative attention + position-wise FFN + residual/LayerNorm, with segment-recurrence memory), a tied-embedding softmax cross-entropy head, and Adam with warmup + linear decay. The data pipeline yields token-id sequences, next-token targets, segment ids, and a boolean candidate-selection vector. The open slot is the self-supervised objective: how to choose prediction positions, convert the generic streams into attention controls and auxiliary inputs, route representations through relative attention, and compute the loss.

```python
import tensorflow as tf

def embedding_lookup(x, n_token, d_embed, initializer):
    table = tf.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
    return tf.nn.embedding_lookup(table, x), table

def head_projection(h, d_model, n_head, d_head, initializer, name):
    # Existing Transformer-XL projection primitive.
    pass

def post_attention(h, attn_vec, d_model, n_head, d_head, dropout,
                   is_training, initializer, residual=True):
    # Existing Transformer-XL output projection, residual, dropout, LayerNorm.
    pass

def rel_shift(x, klen=-1):
    # Existing Transformer-XL relative-position alignment primitive.
    pass

def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt,
                  is_training, scale):
    # Relative attention score core; the objective can decide whether any
    # auxiliary pairwise term is needed.
    pass

def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,
                     activation, is_training, reuse=None):
    # Existing Transformer-XL feed-forward sublayer.
    pass

def cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    # Existing segment-recurrence cache.
    pass

def build_pretraining_inputs(inputs, targets, is_selected, perm_size, seq_len,
                             sep_id, cls_id, num_predict=None):
    """Create objective-specific masks, stream inputs, targets, and optional
    compact prediction mapping from the generic token stream.
    TODO: fill in the self-supervised objective."""
    pass

def objective_attention_layer(h, g, r, mems, r_w_bias, r_r_bias,
                              seg_mat, r_s_bias, seg_embed,
                              attn_mask_h, attn_mask_g, target_mapping,
                              d_model, n_head, d_head, dropout, dropatt,
                              is_training, kernel_initializer, scope='rel_attn'):
    """Fill the objective-specific attention slot inside one Transformer-XL layer.
    TODO: decide what representations the objective needs and how they attend."""
    pass

def pretraining_loss(hidden, target, target_mask, lookup_table, n_token,
                     d_model, initializer, use_tpu=False):
    """Tied-embedding cross-entropy over the selected prediction positions."""
    pass
```
