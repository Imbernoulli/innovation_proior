## Research question

We have unlabeled text in essentially unlimited quantity and a small amount of labeled data for each downstream natural language understanding (NLU) task — classification, natural language inference, reading comprehension, span-extraction question answering, document ranking. The dominant recipe is to *pretrain* a deep neural network on the raw text with a self-supervised objective, then *finetune* it on each task. The open question is: **what self-supervised objective should we pretrain with?**

## Background

**Pretraining-then-finetuning for NLP.** Learning reusable text representations from large unlabeled corpora and transferring them to downstream tasks has been repeatedly successful (Dai & Le 2015, semi-supervised sequence learning with recurrent nets; McCann et al. 2017, representations from machine-translation encoders; Peters et al. 2018, ELMo; Radford et al. 2018, the GPT generative-pretraining-then-finetuning protocol). Two families of self-supervised objective dominate: **autoregressive (AR) language modeling** and **autoencoding (AE) denoising**.

**Language modeling is density estimation.** Estimating p(x) for a text sequence x = (x_1,…,x_T) is a long-standing unsupervised problem. The standard handle is the chain rule, which factorizes the joint into a product of conditionals. This is exact and assumption-free; the only design choice is the *order* in which the chain rule is applied. Conventional language models apply it in the fixed forward order p(x) = Π_t p(x_t | x_{<t}) (or the backward order). Progress in this area — deeper Transformers, longer effective context — has been rapid.

**Self-attention Transformer backbone (Vaswani et al. 2017).** Tokens are embedded; each layer computes multi-head scaled-dot-product attention, softmax(QKᵀ/√d_head)·V, followed by a position-wise feed-forward network, each sublayer wrapped in a residual connection and layer normalization. Order information enters through positional encodings added to the embeddings. A causal attention mask makes the model autoregressive; no mask makes every position see every other. The mask is a free knob: *which positions a given position may attend to is decided entirely by the mask*, independent of the actual sequence layout.

**Transformer-XL (Dai et al. 2019), the state of the art in AR language modeling.** Two ideas matter here.
- *Segment recurrence.* Process a long text in fixed-length segments; cache the hidden states of the previous segment and let the current segment attend to them as a read-only memory (no gradient flows into the cache). This extends the effective context far beyond one segment and improves optimization.
- *Relative positional encoding.* Absolute position embeddings break when you reuse states across segments (position 1 of segment 2 would collide with position 1 of segment 1). Instead, the attention logit between query position i and key position j is decomposed so that position enters only through the *relative distance* i−j:
  A_{ij} = (q_i + u)ᵀ k_j + (q_i + v)ᵀ W_R r_{i−j},
  where q_i, k_j are the content-based query/key, r_{i−j} is a sinusoidal embedding of the relative distance, W_R projects it, and u, v are two global learnable bias vectors (one for the content term, one for the position term) that replace the absolute-position-dependent query bias. The first term is content-to-content; the second is content-to-relative-position. This relative scheme is also a better inductive bias and generalizes well.

**Permutation / order-agnostic density estimators (Germain et al. 2015, MADE; Uria et al. 2016, orderless NADE).** These train a single model to factorize the density under *many* factorization orders rather than one fixed order, by sampling an order per example and masking the network accordingly. Their motivation is purely better density estimation — baking an "orderless" inductive bias into the estimator — and they lean on the implicit position-awareness of an MLP architecture.

## Baselines

**Autoregressive (AR) language modeling** (Dai & Le 2015; Peters et al. 2018; Radford et al. 2018). Pretrain by maximizing the forward-factorized likelihood
  max_θ log p_θ(x) = Σ_{t=1}^T log p_θ(x_t | x_{<t}) = Σ_t log [ exp(h_θ(x_{1:t−1})ᵀ e(x_t)) / Σ_{x′} exp(h_θ(x_{1:t−1})ᵀ e(x′)) ],
where h_θ(x_{1:t−1}) is a context representation (RNN or Transformer) and e(·) the token embedding. It is exact density estimation via the universal product rule, with no independence assumption and no artificial input symbols, and inherits all language-modeling progress. ELMo (Peters et al. 2018) trains a separate backward LM and concatenates the two directions' features to incorporate both sides.

**Autoencoding / denoising (BERT, Devlin et al. 2018), the current state of the art for NLU pretraining.** Corrupt x into x̂ by replacing a fraction (≈15%) of tokens with a special symbol [MASK]; let x̄ be the set of masked tokens; reconstruct them:
  max_θ log p_θ(x̄ | x̂) ≈ Σ_{t=1}^T m_t · log p_θ(x_t | x̂) = Σ_t m_t · log [ exp(H_θ(x̂)_tᵀ e(x_t)) / Σ_{x′} exp(H_θ(x̂)_tᵀ e(x′)) ],
where m_t = 1 marks a masked position and H_θ maps the length-T sequence to a sequence of hidden vectors. Because reconstruction is not density estimation, every H_θ(x̂)_t may attend to *both* sides — this is how BERT gets bidirectional context, and it is why it outperforms AR pretraining on NLU.

**Order-agnostic density estimators (MADE, orderless NADE).** Train one model over many factorization orders via masking. Their goal and inductive bias are specific to that aim: they target orderless density estimation and are position-unaware (degenerating toward bag-of-words for an MLP).

## Evaluation settings

The natural yardsticks:
- **GLUE** (Wang et al. 2019): a suite of nine NLU tasks (e.g. MNLI, SST-2, QNLI, QQP, RTE, MRPC, CoLA), evaluated via a held-out submission server; single-task and multi-task, single-model and ensemble protocols.
- **Reading comprehension:** RACE (Lai et al. 2017), ~100K questions from English exams with long passages (avg >300 tokens), a long-context test; SQuAD 1.1 (Rajpurkar et al. 2016, always-answerable) and SQuAD 2.0 (Rajpurkar et al. 2018, with unanswerable questions, needing a joint answerability + span-extraction loss).
- **Text classification:** IMDB, Yelp-2/5, DBpedia, AG, Amazon-2/5 (following Zhang et al. 2015).
- **Document ranking:** ClueWeb09-B (Dai 2018 setting), reranking the top-100 retrieved documents for TREC Web Track queries; probes the quality of word embeddings via a kernel-pooling ranker, often without finetuning.
- **Protocol:** finetune the pretrained encoder per task with a light task head; sequence length up to 512; metrics are task-native (accuracy, F1/EM, MRR/NDCG). Compare against AR-LM and denoising-AE pretraining under matched data and model size.

## Code framework

The available primitives are a token-embedding lookup, a Transformer-XL relative-attention block (multi-head relative attention + position-wise FFN + residual/LayerNorm, with segment-recurrence memory), a tied-embedding softmax cross-entropy head, and Adam with warmup + linear decay. The data pipeline yields token-id sequences, next-token targets, segment ids, and a boolean candidate-selection vector. The open slot is the self-supervised objective: how to choose prediction positions, convert the generic token stream into whatever attention controls and auxiliary inputs the objective needs, route representations through relative attention, and compute the loss.

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

def objective_attention_layer(inp, r, mems, r_w_bias, r_r_bias,
                              seg_mat, r_s_bias, seg_embed,
                              d_model, n_head, d_head, dropout, dropatt,
                              is_training, kernel_initializer, scope='rel_attn'):
    """Fill the objective-specific attention slot inside one Transformer-XL layer.
    TODO: decide what representation(s) the objective needs, what controls (masks,
    mappings) drive them, and how they attend."""
    pass

def pretraining_loss(hidden, target, target_mask, lookup_table, n_token,
                     d_model, initializer, use_tpu=False):
    """Tied-embedding cross-entropy over the selected prediction positions."""
    pass
```
