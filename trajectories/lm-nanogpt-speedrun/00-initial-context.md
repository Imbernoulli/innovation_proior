## Research question

How fast can 8×H100 GPUs train a GPT-2-scale language model from scratch to a fixed quality bar? The bar is **≤3.28 cross-entropy loss on the FineWeb validation set**. The dataset (FineWeb), the hardware (8×H100), the validation metric (mean token-level cross-entropy on a fixed held-out shard), and the loss threshold are all frozen. The single free variable is **the training algorithm itself** — architecture, optimizer, attention implementation, numerical precision, and schedule. A record is any configuration that reaches ≤3.28 val loss in less total wallclock than the current best, validated to stay below the bar across repeated runs.

## Prior art / Background / Baselines

The starting recipe is the standard small-GPT pipeline inherited from NanoGPT and llm.c.

- **llm.c / NanoGPT trainer.** A 124M-parameter GPT-2 small with 12 transformer blocks, width 768, 12 heads, learned absolute positional embeddings, LayerNorm, GELU MLPs of width 4×, tied input/output embeddings, trained with AdamW under a cosine learning-rate schedule. On 8×H100 it reaches the 3.28 FineWeb val-loss bar in roughly 45 minutes.

- **AdamW optimizer.** It maintains per-coordinate first- and second-moment estimates and rescales each scalar parameter by its own running gradient statistics, with weight decay decoupled from the adaptive step.

- **Original positional / normalization / nonlinearity choices.** Learned absolute position embeddings are added once at the input, LayerNorm uses learned gain and bias, and the MLP uses GELU.

- **Dense causal attention over the full sequence.** Every query attends to every earlier key, giving an O(T²) cost in sequence length for compute and memory.

- **fp32/tf32 logits and tied embeddings.** The language-model head reuses the input embedding matrix and is computed in high precision.

## Fixed substrate / Code framework

The scaffold is a single-file PyTorch trainer descended from llm.c's `train_gpt2.py`. It reads pre-tokenized FineWeb shards, runs a decoder-only transformer under DistributedDataParallel across the 8 GPUs, evaluates cross-entropy on a fixed `val_tokens` budget at intervals, and logs each step as `step:S/N val_loss:L train_time:Tms`. The final logged line of a run is the record: its `val_loss` and total `train_time`.

## Editable interface

The editable parts of the training algorithm are: the architecture (depth, width, heads, embeddings, MLP ratio, tying), the optimizer and its hyperparameters, the attention implementation, the numerical precision of forward/backward/optimizer steps, the learning-rate schedule, and the normalization/activation choices. Any edit is admissible if the run still ends with mean val_loss ≤ 3.28 and lower total train_time than the current record.

## Evaluation settings

A record is the final line of a reproducible run log: `val_loss` (mean cross-entropy over a fixed validation-token budget, lower is better, must be ≤ 3.28) and total `train_time` in milliseconds (lower is better), together with the step count `num_iterations`. Because a single run has run-to-run variance of a few thousandths in val loss, a record is additionally validated by running it many times and checking that the mean is statistically below 3.28 (a one-sided t-test, e.g. `p=0.0001`). Hardware is fixed at 8×H100 (SXM5); the data is the FineWeb training shards with a fixed held-out validation shard. The numbers quoted are the repository's own published record logs.
