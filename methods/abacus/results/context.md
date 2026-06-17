# Context: length generalization for multi-digit addition in transformers (circa 2023–2024)

## Research question

Can a decoder-only transformer trained from scratch on *short* addition problems solve
*much longer* ones at test time? Concretely: train on operands of at most 20 digits and ask
the model to add 100-digit numbers it has never seen. This is a clean laboratory probe of
**logical / length extrapolation** — the input format is trivial, modest models have ample
capacity, and yet the failure is sharp and reproducible. The goal is an architecture (really,
an input representation) that lets a small model trained on a one-GPU-day budget extrapolate
several times beyond its training length, and ideally one that transfers to neighbouring
algorithmic tasks (multiplication, sorting) rather than being a one-off addition hack.

Why it matters: addition is the simplest member of the broad class of algorithmic-reasoning
problems where transformers must *execute* a procedure and have it keep working as the problem
grows. If a transformer cannot extrapolate on the easiest such task, the same weakness is what
makes large models fail on long arithmetic without a code interpreter. Understanding the
precise reason for the failure on addition is a wedge into the general problem.

## Background

The empirical picture that sets up the problem comes from a line of work on teaching
transformers arithmetic, and from the broader study of positional information.

**Addition is hard for transformers, and the difficulty is positional, not arithmetic.**
Prior studies (Lee et al. 2023; Shen et al. 2023; Zhou et al. 2023, 2024) repeatedly find
that standard transformers struggle to learn multi-digit addition even when training data is
abundant (millions of stratified examples) and even when inputs are written least-significant
digit first. A recurring diagnostic observation: the models err in ways consistent with
*misaligning digits of equal significance* — outputting one too few or too many digits,
adding the units digit of one operand to the tens digit of the other. The natural human
algorithm for long addition is to first stack digits into columns by significance and only
then add column by column; the transformer is being asked to do the addition without an easy
way to represent which digits share a column.

**Positional embeddings and where they break.** A transformer's attention is
permutation-invariant, so position must be injected explicitly (Vaswani et al. 2017). Two
broad families exist. *Absolute* positional embeddings (APE) — learned or sinusoidal vectors
added to the token embeddings before the first layer, indexed by the token's offset from the
*start of the whole sequence*. APE is known to inhibit length generalization (Press et al.
2022): a test sequence of length M > N (the max training length) presents positions
N+1,…,M whose embeddings were never trained, so the model sees out-of-distribution position
inputs exactly where it most needs to be reliable. Learned absolute embeddings beyond the
training range are simply uninitialized garbage at test time. *Relative* positional embeddings
inject a function of the offset (i−j) between query and key inside the attention computation
(Shaw et al. 2018; Raffel et al. 2020), which removes the dependence on absolute index and
generalizes better; ALiBi (Press et al. 2022), Kerple, Sandwich, and RoPE (Su et al. 2024)
are members of this family. RoPE in particular is the workhorse of open-source LLMs but is
weak at length generalization because it is only ever trained on rotations up to the training
length. A third option is to use *no* explicit positional embedding at all (NoPE,
Kazemnejad et al. 2023): a causal decoder can still infer position from the mask, and on small
algorithmic tasks NoPE generalizes surprisingly well, sometimes beating specialized schemes.

**The key prior observation about extrapolation.** Ruoss et al. (2023) localized *why*
standard positional encodings fail to extrapolate: the resulting position vectors at test
time fall outside the distribution seen in training, and this distribution shift grows with M.
Their fix (detailed as a baseline below) is the load-bearing antecedent — train the model on
positions drawn from a much wider range than the training lengths so that the large positions
are no longer novel at test time. This reframed length generalization as a *coverage* problem
in position-embedding space, knowable before any digit-specific method exists.

**Data-format facts that already help.** Writing operands least-significant-digit-first
(reversed) is a popular and effective trick (Lee et al. 2023; Shen et al. 2023; Zhou et al.
2023, 2024), because it lets the model generate the answer in the same order carries
propagate. Adding explicit index/alignment characters into the string also helps (Zhou et al.
2023). These are established pre-method facts about the data representation.

## Baselines

The methods a new approach for addition would be measured against and reacts to.

**Learned absolute positional embeddings (APE; Vaswani et al. 2017; Gehring et al. 2017).**
A table `E_pos` of learned vectors indexed by absolute position; `x_t ← token_embed(t) +
E_pos[pos_t]`, `pos_t` counted from the start of the sequence. Simple and effective
in-distribution. **Limitation:** the rows of `E_pos` for positions beyond the training length
are never updated, so at test time on longer inputs the model is fed untrained embedding rows;
even within range, the model has only ever associated a given absolute index with one role,
and a digit's significance is not its absolute index (the units digit of a 5-digit and a
15-digit number sit at different absolute positions). It cannot expose "these digits share a
column."

**No positional embeddings (NoPE; Kazemnejad et al. 2023).** Drop explicit position entirely;
rely on the causal mask, which already breaks permutation symmetry, to let the decoder learn
positional structure. Among existing schemes it is one of the best for reversed addition.
**Limitation:** the position information it recovers is implicit and tied to attention depth;
on long sequences the signal it can reconstruct about *within-number significance* is weak,
and accuracy degrades as numbers grow past the training range.

**FIRE — functional interpolation for relative position (Li et al. 2023).** A learned additive
relative bias in attention: `A(X) = X W_Q (X W_K)^T + B`, with
`B_{ij} = f_θ( log(c·(i−j)+1) / log(c·max(i,L)+1) )`, where `f_θ` is a small MLP and `c, L`
are learnable scalars. The log-and-normalize argument keeps the bias bounded and smooth as
(i−j) grows, so it interpolates to relative distances larger than those seen in training;
combined with randomized positions it gives the strongest reported length generalization on
addition (≈2.5× when paired with index-style randomization, Zhou et al. 2024). **Limitation:**
the bias is a function of the raw query–key offset over the *whole* sequence; it has no notion
of where a number begins, so it does not directly represent a digit's significance within its
own number — the alignment-by-column signal still has to be inferred.

**Index hints (Zhou et al. 2023, 2024).** Insert explicit alignment characters tying together
digits of equal significance across operands and answer, e.g. `a6b7c5 + a1b6c3 = a7b3c9`. This
hands the model the column structure directly and substantially improves addition.
**Limitation:** it inflates the input context and roughly *doubles* the output length and
inference cost; and the resulting generalization is brittle — Zhou et al. (2024) report that
varying only the random seed swings accuracy from near-perfect on 100-digit addition to 0% at
90 digits.

**Randomized positional encodings (Ruoss et al. 2023).** The load-bearing antecedent. To stop
test positions from being out-of-distribution, train on positions sampled from a much larger
range. Let `U(S)` be the uniform distribution over set `S`. Each training step: sample a
batch length `n ~ U(1,N)`, then sample an *ordered* index set `I = {i_1 < i_2 < … < i_n}`
uniformly without replacement from `{1,…,L}` with `L ≫ M`; token at position j gets
`PE(i_j, ·)` instead of `PE(j, ·)`. Applied once per batch (not per sequence). Because the
indices range over `{1,…,L}`, the large embeddings get trained even when the actual sequences
are short, so at test on length `M ≤ L` no position is novel — in-distribution accuracy is
left unchanged while length generalization rises (≈12% average across algorithmic tasks).
An ablation confirms the *ordering* of the sampled indices is necessary. **Limitation:** the
sampled indices are *random gaps* — consecutive tokens receive arbitrary, non-adjacent
indices. This is fine for generic length extrapolation, but it discards the simple
consecutive-counting structure (position k, then k+1, then k+2) that a representation of
"first digit, second digit, third digit *of this number*" would need; and the scheme counts
over the whole sequence, with no per-number reset.

## Evaluation settings

The natural yardsticks, all pre-method facts (datasets generated on the fly, metrics already
in use):

- **Task / data.** Multi-digit addition of two natural numbers, operands written reversed
  (LSD first), no padding between digits and no zero-padding of operands; e.g.
  `98282 + 3859172 = 2787472`. Training sets of 20M samples, sampled with replacement and
  *stratified* so every operand-length pair `(i,j)` with `i,j ≤ 20` is equally represented;
  the loss is computed only on the answer digits (question tokens masked). Character-level
  tokenizer, greedy decoding at test.
- **Splits.** (i) in-distribution: operand lengths up to the training maximum; (ii) OOD:
  longer than training but both operands ≤ 100 digits; (iii) extreme OOD: both operands of
  equal length, > 100 and < 160 digits. Accuracy is tabulated per `(i,j)` pair, including
  `i ≠ j`, to surface every kind of extrapolation.
- **Metric.** Strict *exact-match accuracy* — a sample counts as correct only if every output
  digit is right; one wrong digit fails the whole example. Means are taken over three training
  runs rather than best-of-ten.
- **Budget / protocol.** A language-model "cramming" setup (Geiping & Goldstein 2023): each
  addition run is capped at ≈8 exaFLOP — a single Nvidia RTX A4000 for 24 hours. Optimizer
  AdamW; hidden size 1024, 16 heads; GELU-GLU activation; pre-/post-LayerNorm; trapezoid LR
  schedule; learning rate 1e-4; global batch 8192 (with batch-size ramp).
- **Neighbouring tasks for transfer.** Multiplication of up to 15-digit operands; sorting of
  arrays of variable-length reversed integers indexed by letters; and a bitwise-OR alignment
  probe on binary vectors — all length-generalization testbeds with the same exact-match
  metric.

## Code framework

The encoder plugs into a standard decoder-only transformer training harness that already
exists. The token-embedding table, the stacked `nn.TransformerDecoderLayer`s, the causal mask,
the AdamW loop, and the loss are all in place. What is *not* settled is how positional
information should be injected so that a model trained on short numbers stays in-distribution
on long ones — that injection is the single empty slot below. The scaffold therefore exposes
only the generic "add a positional signal to the token embeddings" interface, with an empty
module to fill in.

```python
import torch
import torch.nn as nn


class PositionalSignal(nn.Module):
    """Produces a positional vector for every token, to be added to the token
    embeddings before the first decoder layer. The whole question of *how* to
    index positions so a model trained on short sequences stays in-distribution
    on long ones lives here; nothing about that indexing is decided yet."""

    def __init__(self, dim, vocab_info, **kwargs):
        super().__init__()
        # whatever parameters/tables the chosen positional scheme needs
        # TODO: the positional representation we will design
        pass

    def forward(self, input_ids):
        # input_ids: LongTensor[B, T]
        # return: FloatTensor[B, T, dim] to add onto the token embeddings
        # TODO: compute the positional vectors
        raise NotImplementedError


class AdditionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = PositionalSignal(cfg.dim, cfg.vocab_info)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim, nhead=cfg.nheads,
            dim_feedforward=cfg.ffn, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)

    def forward(self, input_ids, causal_mask):
        x = self.tok_embed(input_ids)
        x = x + self.pos(input_ids)          # the slot under design
        h = self.decoder(x, mask=causal_mask)
        return self.lm_head(h)


# existing training loop the model plugs into (reversed-digit addition, answer-only loss)
def train(model, loss_fn, data_loader, optimizer, causal_mask):
    for input_ids, targets, answer_mask in data_loader:
        optimizer.zero_grad()
        logits = model(input_ids, causal_mask)
        loss = loss_fn(logits[answer_mask], targets[answer_mask])  # loss on answer digits only
        loss.backward()
        optimizer.step()
```

The contribution will fill in `PositionalSignal`; the rest of the harness is untouched.
