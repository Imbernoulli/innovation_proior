## Research question

Can a sequence model learn the bounded hierarchical **Dyck-(k,m)** language — well-nested strings over `k` bracket types with maximum nesting depth `m` — and length-generalize to test strings strictly longer than any seen during training? Dyck-(k,m) is a canonical probe for an *explicit stack*: the set of valid next tokens is fixed by the current stack, so an exact recognizer needs only `O(m log k)` bits of memory. The only design choice is the **model architecture** (`build_model`); the sampler, training loop, optimizer, and evaluation are fixed. The headline metric is next-valid-set token accuracy on the **length-OOD** split.

## Prior art / Background / Baselines

- **Bounded-memory Dyck recognizers (Hewitt, Hahn, Ganguli, Liang & Manning, 2020).** A small RNN with `O(m log k)` hidden units can exactly represent Dyck-(k,m) by encoding the bounded stack in its fixed-width state. Gap: the construction shows representability, not whether gradient descent on a standard language-modeling objective learns the stack-tracking state and length-generalizes.
- **Self-attention on hierarchical languages (Yao, Peng, Papadimitriou & Narasimhan, 2021).** Transformers with `O(log n)` depth can represent bounded-depth Dyck. Gap: with standard absolute positional encodings, learned Transformers fit the training lengths and collapse on longer strings.
- **Stack-augmented recurrent nets (Joulin & Mikolov, 2015; Suzgun et al., 2019).** An RNN augmented with a differentiable external stack whose memory depth grows with the input can length-generalize on Dyck-like tasks where plain LSTMs fail. Gap: existing results rely on an unbounded stack, curriculum training, and test-time discretization; whether a bounded, end-to-end-trained stack inside this harness reaches the same generalization is open.

## Fixed substrate / Code framework

Everything except `build_model` is frozen. The generator samples Dyck-(k,m) strings by a depth-bounded random walk (even lengths, uniform in the per-env training range) and holds 8 000 training strings in memory. Each example is packed as `[BOS, x1, …, xn, EOS]` and padded with EOS as PAD; the model sees `[BOS, x1, …, xn]` and is scored on every non-PAD position. The vocabulary size is `2k + 2` (BOS, EOS/PAD, `k` open brackets, `k` closers). Training is left-to-right language modeling: token-level softmax cross-entropy on non-PAD positions, **AdamW** at `lr = 3e-3`, `weight_decay = 0`, gradient norm clipped to `1.0`, batch size `64`, **100 gradient steps per env**. The harness rejects models with more than **500 000** parameters. Default `config.hidden_dim` is `64`.

## Editable interface

`build_model(config: TaskConfig) -> DyckModel` returns a `torch.nn.Module` subclass of `DyckModel` with `forward(input_ids: LongTensor[B, T]) -> logits: FloatTensor[B, T, vocab]`. The harness provides `vocab_size(config.k)` and the language parameters on `config` (`k`, `m`, `hidden_dim`). The default fill is a 2-layer LSTM language model.

```python
def build_model(config: TaskConfig) -> DyckModel:
    """Construct the sequence model.

    Default baseline: a 2-layer LSTM language model with `config.hidden_dim`
    hidden units.  The agent is free to replace this entirely (e.g. with a
    Transformer, a stack-augmented RNN, or a memory network) as long as the
    returned module implements `DyckModel.forward(input_ids) -> logits`.
    """

    class LSTMModel(DyckModel):
        def __init__(self, vocab: int, hidden: int, num_layers: int):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.rnn = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            h = self.embed(input_ids)
            h, _ = self.rnn(h)
            return self.head(h)

    return LSTMModel(vocab=vocab_size(config.k), hidden=config.hidden_dim, num_layers=2)
```

## Evaluation settings

Three environments, seed 42, one run per seed:

| label             | `k` | `m` | train length | OOD eval length |
|-------------------|-----|-----|--------------|-----------------|
| `dyck-k2-m3`      | 2   | 3   | [8, 64]      | [65, 96]        |
| `dyck-k8-m5`      | 8   | 5   | [16, 96]     | [97, 128]       |
| `dyck-length-ood` | 4   | 4   | [8, 64]      | [128, 256]      |

For each environment the harness reports `id_token_acc` / `ood_token_acc` (next-valid-set token accuracy, in `[0,1]`), `id_string_acc` / `ood_string_acc` (strict per-string accuracy), and `params`. The per-env score is `ood_token_acc`; the task score is the **geometric mean of `ood_token_acc` across the three environments**. `dyck-length-ood` is the hardest probe: training lengths stop at 64, evaluation runs at 128–256.
