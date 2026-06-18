**Problem.** The absolute-position Transformer fit in-distribution but collapsed on length-OOD
(`ood_string_acc` ≈ 0 on every env): its bracket-matching is anchored to absolute coordinates, which are
untrained noise past the training length. The next rung must process the sequence with a *length-
independent* rule that carries the stack in state, not in position.

**Key idea.** A 2-layer **LSTM** language model. A recurrent net updates its state by the same transition
at every step — no absolute-position table to run off the end of, so order comes for free and the rule
generalizes to any length. The gated memory cell (`c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙
tanh(c_t)`) is a constant-error carousel: a linear state whose backward error is multiplied by the forget
gate, ≈ unit gain across long lags when the gate is open, so the model can hold the depth-`m` stack across
a nested span and read it when the matching closer arrives. The forget gate supplies the per-bracket reset.

**Why it should beat the Transformer.** Dyck-(k,m) has *bounded* depth `m`, so a fixed-width state suffices
(`O(m log k)` units by the bounded-memory result; 64 is generous). The length-independent transition is
exactly the property absolute positions lacked, so `ood_token_acc` should rise most where extrapolation was
largest (`dyck-length-ood`).

**Hyperparameters.** `nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)`, `hidden=config.hidden_dim`
(64), no positional embedding, ~67k params (well under the 500k budget). Two layers so the second can
compose top-of-stack identity for the 8-type `dyck-k8-m5` config.

```python
def build_model(config: TaskConfig) -> DyckModel:
    """Two-layer LSTM language model (Hewitt et al. EMNLP 2020 baseline)."""

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

    return LSTMModel(
        vocab=vocab_size(config.k),
        hidden=config.hidden_dim,
        num_layers=2,
    )
```
