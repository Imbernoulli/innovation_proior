## Research question

How fast can eight H100 GPUs be made to train a GPT-2-scale language model from scratch to a fixed
quality bar? The bar is concrete and unforgiving: **3.28 cross-entropy loss on the FineWeb validation
set**, the same number Andrej Karpathy's `llm.c` GPT-2 (small) reproduction reaches after about 45
minutes of 8×H100 wallclock. Everything about the *target* is frozen — the dataset (FineWeb), the
hardware (8×H100), the validation metric (mean token-level cross-entropy on a fixed held-out shard),
and the loss threshold (≤3.28). The single free variable is **the training algorithm itself**: the
architecture of the network, the optimizer, the attention implementation, the numerical precision, the
schedule. A "record" is any change that reaches ≤3.28 val loss in *less total wallclock* than the
previous record, validated to be statistically below the bar across repeated runs. Lower wallclock is
better; the val loss must stay at or below 3.28, so a method that trims minutes is only admissible if it
still clears the bar.

The substrate is a single-file PyTorch trainer descended from `llm.c`'s `train_gpt2.py` (itself a
descendant of NanoGPT). It reads pre-tokenized FineWeb shards, runs a decoder-only transformer under
DistributedDataParallel across the 8 GPUs, evaluates cross-entropy on a fixed `val_tokens` budget every
so often, and logs each step as `step:S/N val_loss:L train_time:Tms`. The final logged line of a run is
the record: its `val_loss` and total `train_time`. The whole game is to drive that `train_time` down
while keeping `val_loss ≤ 3.28`.

## Prior art before the first rung (the GPT-2 training recipe of 2024)

The ladder climbs out of the standard 2024 way to train a small GPT, which is the `llm.c`/NanoGPT
recipe and the architectural and optimization defaults it inherited.

- **The llm.c / NanoGPT trainer (Karpathy).** A 124M-parameter GPT-2: 12 transformer blocks, model
  width 768, 12 heads, learned absolute positional embeddings, LayerNorm, GELU MLPs of width 4×, tied
  input/output embeddings, trained with **AdamW** under a cosine (or trapezoidal warmup/warmdown) learning-rate
  schedule, cross-entropy loss, gradient accumulation to a large global batch, fp32/tf32 logits. On
  8×H100 this reaches 3.28 FineWeb val loss in roughly **45 minutes** over ~10B tokens. Gap: it is a
  faithful 2019-era GPT-2, so every component is the *conservative* choice — the optimizer touches each
  weight coordinate independently, the positional scheme is the original learned-absolute one, the MLP
  nonlinearity and the normalization are the GPT-2 defaults, and attention is dense over the full
  context. Nothing in it is tuned for *wallclock-to-a-fixed-loss* on Hopper hardware.

- **AdamW as the universal optimizer (Loshchilov & Hutter 2019; Kingma & Ba 2015).** Adam maintains
  per-parameter first and second moment estimates m, v and updates each weight by `−lr · m̂ / (√v̂ + ε)`;
  AdamW decouples the weight decay. It is the default for transformers because it is robust and needs
  little tuning. Gap: it is a *coordinate-wise* method — it rescales each scalar entry of a weight matrix
  by its own running gradient statistics and is blind to the matrix structure of a linear layer. Two
  weight matrices with very different conditioning get the same elementwise treatment, and there is no
  mechanism that conditions the *update direction* as a matrix.

- **The original positional / normalization / nonlinearity choices.** Learned absolute position
  embeddings tie the model to a fixed maximum length and inject position additively at the input only;
  LayerNorm carries a learned gain and bias; the GELU MLP is the GPT-2 default. By 2024 each of these
  had a known, better-behaved alternative in the literature, but the baseline ships the originals.

- **Dense causal attention over the full sequence.** Every query attends to every earlier key, an
  O(T²) cost in both compute and memory that grows with context length. The baseline runs a modest
  context (~1k tokens) precisely because dense attention makes long contexts expensive. Gap: most of the
  useful signal for next-token prediction is local, yet dense attention pays the full quadratic price to
  let every token see every other token, and that cost is the dominant per-step expense as soon as the
  context is widened.

- **fp32/tf32 logits and a single tied embedding.** Logits are computed in high precision and the input
  embedding is reused as the output classifier (weight tying). This is memory-frugal and numerically
  safe. Gap: the largest matmul in a small-vocab-but-50k-token model is the language-model head, and
  computing it in high precision leaves Hopper's low-precision tensor cores idle; tying the head to the
  embedding couples two roles (lookup table vs. classifier) that may want different representations.

The ladder below is the sequence of distinct, individually-named methods — a new optimizer, a modernized
block, a different attention implementation, new embedding pathways, a different numerical format —
each of which, dropped into this trainer, reached the same 3.28 bar in less wallclock than the method
before it.

## Evaluation settings

Every record is the final line of a reproducible run log: `val_loss` (mean cross-entropy over a fixed
validation-token budget, **lower is better**, must be **≤ 3.28**) and total `train_time` in
milliseconds (**lower is better** — this is the quantity being raced), together with the step count
`num_iterations`. Because a single run has run-to-run variance of a few thousandths in val loss, a
record is additionally validated by running it many times and checking that the mean is statistically
below 3.28 (a one-sided t-test, e.g. `p=0.0001`). Hardware is fixed at 8×H100 (SXM5); the data is the
FineWeb training shards with a fixed held-out validation shard. The numbers quoted on each rung are the
repository's own published record logs, not re-run here.
