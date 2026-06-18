**Problem.** MQAR asks the mixer to bind `(key, value)` pairs streamed in early and then, at each later
query position, retrieve the value whose key matches the current token — a content-addressed lookup over an
arbitrary earlier position. The mixer slot is the second layer of a fixed 2-layer hybrid (first layer a short
causal conv); it must be causal and map `[B, T, d_model] → [B, T, d_model]` with `d_model=64`.

**Key idea (the recurrent floor).** Use the classical gated recurrent memory cell — a single-layer LSTM over
the sequence axis, hidden size `d_model`, with a linear read-out. Three multiplicative gates wrap a linear
memory state `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)`; the forget gate lets the state
error flow back at unit gain (`ε_s^t = ... + f_{t+1} ⊙ ε_s^{t+1}`), so unlike a vanilla RNN it can train
across long lags. Causality is automatic — a recurrence at step `t` has consumed only inputs `≤ t`.

**Why this is the floor.** Recall is the index problem: answering an *arbitrary* later query means the state
must have retained *all* bindings. A fixed 64-dimensional state cannot hold many distinct key→value pairs
without them colliding, and the recurrence has no native "compare my query against each stored key" operation
— only "fold the current token into a running summary." The gates choose what to keep but cannot add
capacity.

**Hyperparameters.** `num_layers=1`, `hidden_size=d_model=64`, `batch_first=True`; one bias-free
`out_proj`. No extra width or depth — the minimal faithful recurrent baseline.

**What to watch.** Capacity-limited, not optimization-limited: partial-to-poor at `mqar-128` (8 pairs),
collapsing at `mqar-512` (32 pairs), near-chance at `mqar-2048` (128 pairs over a 16× longer sequence). That
collapse is what forces an explicit query-key score — an attention-like mixer — at step 2.

```python
# EDITABLE region of custom_strategy.py — step 1: LSTM (recurrent floor)
class CustomMixer(nn.Module):
    """Single-layer LSTM as the sequence mixer (causal by construction)."""

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.out_proj(h)
```
