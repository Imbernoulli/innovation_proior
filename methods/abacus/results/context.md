# Context: length generalization for multi-digit addition in transformers (circa 2023–2024)

## Research question

Can a decoder-only transformer trained from scratch on *short* addition problems solve
*much longer* ones at test time? Concretely: train on operands of at most 20 digits and ask
the model to add 100-digit numbers it has never seen. This is a clean laboratory probe of
**logical / length extrapolation** — the input format is trivial, modest models have ample
capacity, and the question is how to inject positional information so that a small model
trained on a one-GPU-day budget can extrapolate several times beyond its training length, and
ideally transfer to neighbouring algorithmic tasks (multiplication, sorting).

Why it matters: addition is the simplest member of the broad class of algorithmic-reasoning
problems where transformers must *execute* a procedure and have it keep working as the problem
grows. Understanding the behaviour on addition is a wedge into the general problem of long
arithmetic without a code interpreter.

## Background

The empirical picture that sets up the problem comes from a line of work on teaching
transformers arithmetic, and from the broader study of positional information.

**Learning addition with transformers.** Prior studies (Lee et al. 2023; Shen et al. 2023;
Zhou et al. 2023, 2024) train standard transformers on multi-digit addition with abundant
data (millions of stratified examples), including with inputs written least-significant digit
first. A recurring diagnostic observation: errors are consistent with *misaligning digits of
equal significance* — outputting one too few or too many digits, adding the units digit of one
operand to the tens digit of the other. The human algorithm for long addition is to stack
digits into columns by significance and then add column by column.

**Positional embeddings.** A transformer's attention is permutation-invariant, so position
must be injected explicitly (Vaswani et al. 2017). Two broad families exist. *Absolute*
positional embeddings (APE) — learned or sinusoidal vectors added to the token embeddings
before the first layer, indexed by the token's offset from the *start of the whole sequence*.
A test sequence of length M > N (the max training length) presents positions N+1,…,M whose
learned embeddings were never updated in training (Press et al. 2022). *Relative* positional
embeddings inject a function of the offset (i−j) between query and key inside the attention
computation (Shaw et al. 2018; Raffel et al. 2020), removing the dependence on absolute index;
ALiBi (Press et al. 2022), Kerple, Sandwich, and RoPE (Su et al. 2024) are members of this
family. RoPE in particular is the workhorse of open-source LLMs, trained on rotations up to
the training length. A third option is to use *no* explicit positional embedding at all (NoPE,
Kazemnejad et al. 2023): a causal decoder can still infer position from the mask, and on small
algorithmic tasks NoPE generalizes surprisingly well, sometimes beating specialized schemes.

**Coverage in position-embedding space.** Ruoss et al. (2023) characterized why standard
positional encodings extrapolate as they do: the position vectors at test time fall outside
the distribution seen in training, and this distribution shift grows with M. Their approach
(detailed as a baseline below) trains the model on positions drawn from a much wider range
than the training lengths so that the large positions are no longer novel at test time,
framing length generalization as a *coverage* problem in position-embedding space.

**Data-format facts that already help.** Writing operands least-significant-digit-first
(reversed) is a popular and effective trick (Lee et al. 2023; Shen et al. 2023; Zhou et al.
2023, 2024), letting the model generate the answer in the same order carries propagate. Adding
explicit index/alignment characters into the string also helps (Zhou et al. 2023). These are
established pre-method facts about the data representation.

## Baselines

The methods a new approach for addition would be measured against and reacts to.

**Learned absolute positional embeddings (APE; Vaswani et al. 2017; Gehring et al. 2017).**
A table `E_pos` of learned vectors indexed by absolute position; `x_t ← token_embed(t) +
E_pos[pos_t]`, `pos_t` counted from the start of the sequence. Simple and effective
in-distribution.

**No positional embeddings (NoPE; Kazemnejad et al. 2023).** Drop explicit position entirely;
rely on the causal mask, which already breaks permutation symmetry, to let the decoder learn
positional structure. Among existing schemes it is one of the strongest for reversed addition.

**FIRE — functional interpolation for relative position (Li et al. 2023).** A learned additive
relative bias in attention: `A(X) = X W_Q (X W_K)^T + B`, with
`B_{ij} = f_θ(ψ(i−j) / ψ(max{L,i}))` under the causal-distance convention `j ≤ i`, where
`ψ(x)=log(c x + 1)`, `f_θ` is a small MLP, and `c, L` are learnable scalars. The released
FIRE code applies the log transform to the absolute relative-distance tensor, so the safe
reading is "bounded normalized distance," not a signed logarithm. The log-and-normalize
construction keeps the bias bounded and smooth as the query-key distance grows, so it
interpolates to relative distances larger than those seen in training; combined with
randomized positions it gives the strongest reported length generalization on addition (≈2.5×
when paired with index-style randomization, Zhou et al. 2024).

**Index hints (Zhou et al. 2023, 2024).** Insert explicit alignment characters tying together
digits of equal significance across operands and answer, e.g. `a6b7c5 + a1b6c3 = a7b3c9`. This
hands the model the column structure directly and substantially improves addition. It inflates
the input context and roughly doubles the output length and inference cost; Zhou et al. (2024)
report that varying the random seed swings accuracy from near-perfect on 100-digit addition to
0% at 90 digits.

**Randomized positional encodings (Ruoss et al. 2023).** To keep test positions in
distribution, train on positions sampled from a much larger range. Let `U(S)` be the uniform
distribution over set `S`. Each training step: sample a batch length `n ~ U(1,N)`, then sample
an *ordered* index set `I = {i_1 < i_2 < … < i_n}` uniformly without replacement from
`{1,…,L}` with `L ≫ M`; token at position j gets `PE(i_j, ·)` instead of `PE(j, ·)`. Applied
once per batch (not per sequence). Because the indices range over `{1,…,L}`, the large
embeddings get trained even when the actual sequences are short, so at test on length `M ≤ L`
no position is novel — in-distribution accuracy is left unchanged while length generalization
rises (≈12% average across algorithmic tasks). An ablation confirms the *ordering* of the
sampled indices is necessary. The sampled indices are random gaps drawn over the whole
sequence: consecutive tokens receive arbitrary, non-adjacent indices.

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
the AdamW loop, and the loss are all in place. How positional information should be injected is
the single empty slot below. The scaffold exposes a generic "add a positional signal to the
token embeddings" interface, with an empty module to fill in.

```python
import torch
import torch.nn as nn


class PositionalSignal(nn.Module):
    """Produces a positional vector for every token, to be added to the token
    embeddings before the first decoder layer. How to index positions so a model
    trained on short sequences stays in-distribution on long ones is decided here."""

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
