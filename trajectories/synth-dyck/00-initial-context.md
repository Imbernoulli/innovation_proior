## Research question

Can a sequence model learn the bounded hierarchical **Dyck-(k,m)** language — well-nested strings over
`k` bracket types with maximum nesting depth `m` — and, crucially, **length-generalize** to test strings
strictly longer than anything seen in training? Dyck-(k,m) is the canonical probe for whether a neural
sequence model implements an *explicit stack*: at every position the set of valid next tokens is fixed
entirely by the stack of currently-open brackets, so an exact recognizer needs only `O(m log k)` bits of
memory. The single thing being designed is the **model architecture** (`build_model`); the data sampler,
training loop, optimizer, and evaluation are all fixed. The headline score is next-valid-set token
accuracy on the **length-OOD** split — strings longer than the training maximum.

## Prior art before the first rung (sequence-model lineage)

The ladder reacts to a body of work on what fixed-width sequence models can and cannot represent, and on
the depth-bounded Dyck recognizers that motivate the task.

- **Bounded-memory Dyck recognizers (Hewitt, Hahn, Ganguli, Liang, Manning, EMNLP 2020).** Prove that a
  recurrent net needs only `O(m log k)` hidden units to recognize Dyck-(k,m) exactly, by encoding the
  depth-`m` stack of `k`-way symbols in a fixed-width state. The construction is the existence proof the
  whole task leans on: a small RNN *can* solve this. Gap: existence of a solution is not learnability —
  whether gradient descent on cross-entropy finds the stack-tracking state, and whether it length-
  generalizes, is exactly what is in question.
- **Self-attention on hierarchical languages (Yao, Peng, Papadimitriou, Narasimhan, ACL 2021).** Show a
  Transformer with `O(log n)` layers/heads *can* process bounded-depth Dyck, but that with learned or
  sinusoidal **absolute** positional encodings it tends to fit in-distribution lengths and then collapse
  on longer test strings. Gap: the representational result does not transfer to length-OOD under the
  standard absolute-position recipe.
- **Stack-augmented recurrent nets (Joulin & Mikolov, NeurIPS 2015; Suzgun et al. 2019).** Bolt a
  differentiable stack with soft PUSH/POP/NO-OP actions onto an RNN controller, so the *depth* of memory
  grows with the input instead of being pinned to a fixed-width vector. Reported to length-generalize on
  generalized Dyck where vanilla LSTMs collapse on the OOD split. Gap: the original constructions use an
  unbounded stack with a curriculum and test-time discretization; the task asks whether a *bounded*,
  end-to-end-trained version inside this harness reaches the same generalization.

## The fixed substrate

Everything except `build_model` is frozen. The data generator samples Dyck-(k,m) strings by a
depth-bounded random walk (lengths even, drawn uniformly in the per-env training range); 8 000 training
strings are held in memory. Each example is packed as `[BOS, x1, …, xn, EOS]` and padded with EOS as
PAD; the model sees `[BOS, x1, …, xn]` and is scored on its prediction at every non-PAD position. The
vocabulary has size `2k + 2`: BOS, EOS/PAD, `k` open brackets, `k` matching closers. Training is a
left-to-right language model: token-level softmax cross-entropy over non-PAD positions, **AdamW** at
`lr = 3e-3`, `weight_decay = 0`, gradient norm clipped to `1.0`, batch size `64`, **100 gradient steps
per env**. The harness rejects any model with more than **500 000** trainable parameters
(`config.max_params`). The default `config.hidden_dim` is `64`.

## The editable interface

`build_model(config: TaskConfig) -> DyckModel` must return a `torch.nn.Module` subclass of `DyckModel`
exposing `forward(input_ids: LongTensor[B, T]) -> logits: FloatTensor[B, T, vocab]`. The harness supplies
the helpers `vocab_size(config.k)` (= `2k + 2`) and the language parameters on `config` (`config.k`,
`config.m`, `config.hidden_dim`). Inside `build_model` you define a model class subclassing `DyckModel`,
construct it from `config`, and return it; the loop handles data, loss, optimization, and the next-valid-
set evaluation. The default fill is a 2-layer LSTM language model.

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

Three environments, each at seed 42, one internal run per seed:

| label             | `k` | `m` | train length | OOD eval length |
|-------------------|-----|-----|--------------|-----------------|
| `dyck-k2-m3`      | 2   | 3   | [8, 64]      | [65, 96]        |
| `dyck-k8-m5`      | 8   | 5   | [16, 96]     | [97, 128]       |
| `dyck-length-ood` | 4   | 4   | [8, 64]      | [128, 256]      |

For each environment the parser reports: `id_token_acc` / `ood_token_acc` (next-valid-set token accuracy
on the in-distribution and length-OOD splits; higher is better, in `[0,1]`), `id_string_acc` /
`ood_string_acc` (strict per-string accuracy — every position correct), and `params`. The per-env score
is `ood_token_acc`; the task score is the **geometric mean of `ood_token_acc` across the three
environments**. `dyck-length-ood` is the hard probe: training tops out at length 64, evaluation runs at
128–256.
