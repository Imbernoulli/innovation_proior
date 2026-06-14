# Context: how a neural language model represents words at its two ends

## Research question

A neural language model touches its vocabulary in exactly two places. At the front, the
current input token is a one-hot vector `c ∈ R^C` (C = vocabulary size) that is looked up
in a word-embedding matrix `U ∈ R^{C×H}` to produce a dense vector `Uᵀc`, which the model
body then consumes. At the back, after the body has produced an activation `h ∈ R^H`, a
second matrix projects `h` to one real-valued score per vocabulary word, `Vh ∈ R^C`, and
a softmax turns those scores into the next-word distribution. Both `U` and `V` have shape
`C × H`: one row per word, one column per hidden dimension. The question is what
relationship, if any, these two matrices should bear to each other. Concretely: should a
language model use one matrix for both ends — looking the word up on the way in and
scoring it on the way out with the *same* row — or two independent matrices? The cheap
move is one matrix (it halves the most parameter-heavy part of the model, since `C·H`
dominates for any realistic vocabulary). The question is whether that sharing helps the
model or quietly forces one set of rows to do two jobs that are not actually the same job,
and under what conditions the answer flips. A good resolution has to say *why* — grounded
in how the two matrices are actually trained and what each one is being asked to capture —
not merely report that one configuration scores better.

## Background

**The model family.** In the common neural language model (Bengio et al. 2003;
Mikolov et al. 2010; the LSTM language models of Sundermeyer et al. 2012 and
Zaremba et al. 2014), the input word is embedded by `U`, a recurrent or otherwise deep
body computes an activation `h` from the embedded history, the scores are `Vh`, and the
loss is the cross-entropy / negative log-likelihood of the true next word `o_t`:
`L_t = −log p_t(o_t | i_{1:t})` with
`p_t(o_t|·) = exp(V_{o_t}ᵀ h) / Σ_x exp(V_xᵀ h)`, where `U_k` (`V_k`) is row `k` of
`U` (`V`). The same two-matrix structure carries over to a transformer decoder
(an input token-embedding table and an output linear projection over the vocabulary),
and to the decoder of a neural machine translation model, whose decoder is itself a
conditional language model. Optimization is by stochastic gradient descent.

**What each end is being asked to do.** The two matrices are word embeddings in two
different senses. For the *input* matrix `U` we want the body to *react similarly* to
words that play similar roles — synonyms should drive the recurrent/transformer state in
nearly the same way, so that downstream computation generalizes across them. For the
*output* matrix `V` we want words that are *interchangeable as continuations* to receive
similar *scores* — `V` is literally the weight matrix of a `C`-way softmax classifier
over the next token, so two words that the model should predict in the same contexts want
nearby rows (Mnih & Teh 2012). These two notions of "similar words" overlap but are not
identical: the first is about how a word, as an input, perturbs the state; the second is
about how a word, as a candidate output, is scored against a state. A representation
optimized for one is not automatically optimal for the other.

**The training dynamics differ at the two ends — a fact about the gradients themselves.**
With separate matrices, differentiating the per-step loss gives, for the input matrix,
a nonzero gradient only on the row of the *current input word*: for `k ≠ i_t`,
`∂L_t/∂U_k = 0`, and for `k = i_t`,
`∂L_t/∂U_{i_t} = (Σ_x p_t(x|·) V_xᵀ − V_{o_t}ᵀ) · ∂h/∂U_{i_t}`. For the output matrix,
*every* row is updated at *every* step: `∂L_t/∂V_{o_t} = (p_t(o_t|·) − 1) h` for the true
word and `∂L_t/∂V_k = p_t(k|·) h` for every other word `k`. So an input row sees a sparse
signal (it learns only on the steps where its word is read, which for a rare word is a
handful of times in all of training), while every output row sees a dense signal (a push
proportional to its predicted probability, every single step). The two matrices, even
when they have the same shape and represent the same vocabulary, are trained by
structurally different gradient streams.

**Where sharing the same matrix is known to be harmful.** In the word2vec skip-gram model
(Mikolov et al. 2013), the same word appears in both an input ("center") and an output
("context") matrix, and the trained vectors that are kept are the input ones; the output
matrix is usually discarded. Goldberg & Levy (2014) argued that for skip-gram the output
representation *needs* to differ from the input one: there the body is the identity, so
the analogue of `h` is the input vector itself. If the center and context tables are
forced to be one table, a word's self-score is `U_iᵀU_i = ||U_i||²`; because words are
rarely their own contexts, suppressing that self-prediction pushes the squared norm down.
This is a concrete failure mode of using one matrix for both ends — but it is argued
specifically for the identity-body case.

**The role of the body.** When there is a nontrivial model body between the two ends — an
LSTM or a transformer — the activation `h` is a nonlinear function of the entire history,
not of the current input row directly. This decouples the input and output ends: the
self-prediction pathology that makes one-matrix sharing harmful in word2vec does not
transmit through a deep body in the same way, because `h` is no longer simply the input
word's own vector.

**Parameter budget and overfitting.** The two `C × H` matrices are, for any moderate
vocabulary, the largest parameter blocks in the model. Sharing them removes one such
block — a large fraction of the total parameter count — which both shrinks the model and
acts as a constraint that can reduce overfitting. That constraint is most valuable when
the corpus is small enough that the output classifier can over-specialize; with more data,
the saved parameters are less clearly the binding concern. Keeping the matrices
independent spends those parameters to give the output classifier degrees of freedom that
are not forced to serve the input lookup too, at the cost of more parameters to
regularize.

## Baselines

**One shared matrix for both ends (tied embeddings; Press & Wolf 2017; Inan et al. 2017;
implicit in earlier log-bilinear and feed-forward models such as Mnih & Hinton 2009 and
Bengio et al. 2003).** Set `U = V = S`: the row that embeds a word on the way in is the
same row that scores it on the way out. Core idea: a word has one representation,
re-used at both ends; the body learns to map from "the space of input embeddings" back
into "the space of output embeddings" by absorbing any needed linear transformation.
Inan et al. (2017) make this precise with a loss-framework argument: augmenting the
cross-entropy with a KL term toward an embedding-similarity target distribution, at high
softmax temperature, makes zero gradient require the model logits `Wh_t` to match
`Lᵀu_t`; under rank assumptions this makes the column space of `W` match the column space
of `Lᵀ`, so `W = LᵀA` for some square map `A`. Reusing the input embedding in the output
projection makes that subspace constraint explicit as `W = Lᵀ`, with the network body
absorbing the needed map into its hidden state. The benefit: roughly a `C·H`-parameter
reduction and a regularizing constraint.
The limitation this leaves open: a single matrix is forced to
satisfy two different similarity structures at once (synonyms-react-alike for the input
role, interchangeable-continuations-score-alike for the output role) and is driven by two
different gradient streams (sparse per-input-row vs. dense every-row). In the tied
gradient, every row except the current input row receives exactly the output-style update.
The current-input row receives an additional input-style term, while its output-style term
is usually the non-target case `p_t(i_t|·)h`, small because immediate repetition is rare.
Thus the special row is locally dominated by the input-role term, but almost all row-time
pairs are output-only updates, making the shared matrix evolve more like the output role
overall. The shared matrix also forces every classifier row to double as an input row, and
its logit vectors lie in the column space of that shared matrix, removing output-specific
capacity. That regularization is a plausible benefit when parameter reduction and
overfitting control are the binding needs; it is a cost when output-specific classifier
capacity is the binding need.

**The loss-augmentation framework alone (Inan et al. 2017).** Rather than constrain the
matrices, add a training-time term that uses the input embedding to build a soft target
distribution over the vocabulary (high probability to words near the target in embedding
space) and penalize KL from it. Core idea: inject the embedding metric into the *loss*,
which both improves supervision (every step updates toward a whole soft distribution, not
one hard label) and implicitly pulls the output projection toward the column space spanned
by the transposed input embedding. Limitation left open: it adds a temperature and a
loss-weight hyperparameter and an
iterative soft-target estimate; and its analysis is what *motivates* equating the two
matrices, so as a standalone it does not by itself decide whether to reuse the same rows
as classifier weights.

**word2vec-style two-matrix model with the output matrix discarded (Mikolov et al.
2013).** Two matrices are maintained (center and context), but the output one is thrown
away after training. Core idea: only the input vectors are treated as "the embedding".
Limitation for language modeling: the output matrix is exactly the next-word classifier
we care about at the back of an LM, so discarding it is not an option here; and the
reason its sharing is harmful in skip-gram (identity body, self-prediction norm collapse;
Goldberg & Levy 2014) is tied to the absence of a deep body.

## Evaluation settings

The natural yardsticks for this question, all pre-existing:

- **Datasets.** Penn Treebank (Marcus et al. 1993; the Mikolov et al. 2010 processed
  version, 10k vocabulary), and `text8` (Mahoney) for word-level language modeling;
  Wikitext-2 (Merity et al. 2016, ~33k vocabulary) as a larger alternative; large token
  streams tokenized with a fixed sub-word tokenizer for GPT-style pretraining. For machine
  translation, WMT EN→FR and EN→DE with BPE sub-words.
- **Models.** LSTM language models of the small (200-unit) and large (1500-unit,
  dropout-regularized) configurations of Zaremba et al. (2014); recurrent highway
  networks (Zilly et al. 2016); and transformer decoders (GPT-2-style: a token-embedding
  table, an output linear projection over the vocabulary, a stack of attention/MLP blocks
  with a fixed body, and a learned absolute position embedding). The output projection
  carries no bias.
- **Protocol.** Identical optimizer, learning-rate schedule, batch size, and body across
  the configurations being compared, so the only difference is the relationship between
  the input embedding and the output projection. Position parameters are excluded from
  any reported parameter count, so a configuration cannot win by spending parameters on
  positions. Cross-entropy / negative-log-likelihood is the training objective.
- **Metrics.** Validation cross-entropy / perplexity (primary; lower is better);
  word-embedding-quality correlations (Spearman ρ against human judgments on Simlex999,
  Verb-143, MEN, Rare-Word, MTurk-771) used diagnostically to compare the *quality* of an
  input vs. an output embedding; for translation, BLEU and model size.

## Code framework

The pieces that already exist: a `TokenEmbedding` module owns the token and position
representations; the surrounding `GPT` model builds the body and an output linear layer,
and wires the output layer's weight from whatever the embedding module hands it; the
training loop computes logits and a cross-entropy loss. What is *not* settled is the
vocabulary-interface choice made inside the embedding module. The harness exposes that
choice through `get_lm_head_weight()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Owns the token/position representation and supplies the weight used by the
    output projection. The body and loss are fixed; this module leaves one generic
    vocabulary-interface slot open."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # input token lookup
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned positions
        self.drop = nn.Dropout(config.dropout)
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        # TODO: vocabulary-interface state, if the chosen interface needs any.

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)                          # (B, T, n_embd)

    def get_lm_head_weight(self):
        # TODO: return the (vocab_size x n_embd) weight the output softmax will use.
        pass

    def get_num_pos_params(self):
        return self.wpe.weight.numel()       # excluded from the reported parameter count

    def get_value_embed(self, layer_idx):
        return None                          # no per-layer residual in this slot


class GPT(nn.Module):
    """Fixed body and loss; the only freedom exercised here is which weight the
    output projection borrows from the embedding module."""

    def __init__(self, config):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # fixed blocks
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.get_lm_head_weight()   # wired from the module

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for i, block in enumerate(self.transformer.h):
            ve = self.embedding.get_value_embed(i)
            if ve is not None:
                x = x + ve
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                    # scores = body activation @ Wᵀ
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss
```
