**Problem.** The vanilla Transformer re-derives the memory bit by *soft attention* at every read, so it
carries a small irreducible per-read error that **compounds** across the many reads in a sequence (dense
read-error 0.0155 → seq-error 0.77; long_ctx → seq-error 0.998). Scale/data do not remove the glitch, and
learned absolute positions extrapolate badly to `T=1024`. The fix must change the mechanism: stop
re-selecting the bit and start *carrying* it in an explicit state.

**Key idea.** A Mamba-style **selective diagonal state-space model**: a per-channel linear recurrence
`h_t = Ā_t h_{t-1} + B̄_t x_t`, `y_t = C_t h_t`, with `Ā_t = exp(-Δ_t·A)` and `Δ_t, B_t, C_t` *input-
dependent* projections. The state propagation is linear (a gradient highway — a write 200 steps back still
reaches a read), and selecting through the step size `Δ` *is* the write/hold gate: large `Δ` ⇒ `Ā→0`,
overwrite (a write token); small `Δ` ⇒ `Ā→1`, hold (an ignore token). A 1-bit memory is trivially a
recurrent state, so almost no width/depth is needed.

**Grounding in this task (vs. canonical Mamba).** The canonical version recovers speed with a fused
parallel-scan CUDA kernel; that cannot ship in this container and is unnecessary at `T≤1024`. So the
recurrence is a plain Python `for t` loop over the state tensor — mathematically identical, just
sequential — and the model is kept tiny (1 block, `d_model=64`, `d_state=8`, expand 2) to stay inside the
wall-clock budget. Discretisation uses exact-ZOH `Ā=exp(-Δ·A)` with first-order input update `B̄x≈Δ·B·x`;
a short causal depthwise conv + SiLU does local mixing; a SiLU gate branch `z` and per-channel skip `D`
complete the block.

**Why it should win.** Bit carried in state + local read-out ⇒ the compounding per-read error collapses
(dense/long_ctx should recover most — a recurrence has no positional-extrapolation problem at `T=1024`).
Risk: holding one write across 100+ ignores on sparse needs `Δ≈0` precisely on ignores; with `d_state=8`
a small `Ā` leak over a long run could still flip the held bit, so sparse may improve but not reach zero.

**Hyperparameters.** `d_model=64`, `n_layers=1`, `d_state=8`, `expand=2`, conv kernel 3 (causal),
`A` init `1..d_state` per channel (`A=exp(A_log)>0`), softplus `Δ`. Trained by the fixed loop.

```python
# EDITABLE region of custom_strategy.py (lines 191-241) — step 2: selective diagonal SSM (Mamba-style)
class _SelectiveSSMBlock(nn.Module):
    """A single Mamba-style selective diagonal-SSM block.

    Input/output shape: [batch, seq_len, d_model].
    Internally we project to a small d_state=8 width and use a python
    loop to apply the data-dependent recurrence (no CUDA kernel).
    """

    def __init__(self, d_model: int, d_state: int = 8, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=3, padding=2, groups=self.d_inner, bias=True,
        )
        # x_proj produces (delta, B, C) per timestep.
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # Learnable per-channel state-decay matrix (log A_d).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        # x:     [bsz, seq, d_inner]
        # delta: [bsz, seq, d_inner]   (positive)
        # A:     [d_inner, d_state]    (negative; we apply exp(delta*A))
        # B,C:   [bsz, seq, d_state]
        bsz, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        # deltaA: [bsz, seq, d_inner, d_state] = exp(-delta * A)
        deltaA = torch.exp(-(delta.unsqueeze(-1) * A))
        # deltaB_u: [bsz, seq, d_inner, d_state] = delta * B * x
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(bsz, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        # The only sequential loop is over time; d_state is capped at 8 in
        # this baseline so each step remains a small batched tensor update.
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            # y_t = sum_state h * C_t   -> [bsz, d_inner]
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        bsz, seq_len, _ = x.shape
        xz = self.in_proj(x)               # [bsz, seq, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)      # each [bsz, seq, d_inner]
        # Causal depthwise conv along time (right-truncate by kernel-1).
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :seq_len]
        x_conv = F.silu(x_conv.transpose(1, 2))
        proj = self.x_proj(x_conv)
        delta_raw, B, C = torch.split(
            proj, [1, self.d_state, self.d_state], dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # [bsz, seq, d_inner]
        A = torch.exp(self.A_log)                    # [d_inner, d_state]
        y = self._selective_scan(x_conv, delta, A, B, C)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class FlipFlopModel(nn.Module):
    """Mamba-style selective SSM stack.

    We keep the model deliberately small (d_model=64, d_state=8, 1 block)
    so the python-level scan stays inside the per-baseline 30-min budget.
    A 1-bit memory is trivially representable so further width is
    unnecessary.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = 1024):
        super().__init__()
        d_model = 64
        n_layers = 1
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [_SelectiveSSMBlock(d_model=d_model, d_state=8, expand=2)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


def build_model(config: TaskConfig) -> nn.Module:
    """Construct the simplified-Mamba FFLM."""
    return FlipFlopModel(
        vocab_size=VOCAB_SIZE,
        max_len=max(config.train_len, config.eval_long_len),
    )
```
