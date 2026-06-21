The Transformer's numbers tell me exactly what is broken and rule out the easy fixes. On dense the read-error rate is tiny — 0.0155, about one and a half percent of *reads* wrong — yet the sequence-error rate is 0.77 and the score only 0.23: the model gets nearly every individual read right, but each dense sequence has hundreds of reads, so a per-read error of one-and-a-half percent compounds into a near-certainty of *some* slip per sequence, and one slip flags the whole sequence. On long_ctx it collapses entirely — read-error 0.099, sequence-error 0.998 — and learned absolute positions trained at $T=512$ extrapolating to $T=1024$ make it worse. This is the attention glitch I expected: a small *irreducible* per-read error that compounds over reads, because the bit is re-selected from raw history by soft attention every single time and a soft selection occasionally misfires. The cure has to change the *kind* of computation — stop re-selecting the bit and start *carrying* it in an explicit state that is written once and read out without re-deriving it.

What carries an explicit state is a recurrence: a hidden state $h_t$ updated step by step, which *could* be the one-bit register — write a token, the state flips; ignore a token, the state holds; read a token, the state is read out. But a generic nonlinear RNN trained through hundreds of steps suffers vanishing/exploding gradients, and the sparse tail is exactly the regime where BPTT through a saturating nonlinearity fails. So I need a recurrence whose state propagation is *linear* (so the gradient does not collapse) and whose update is *content-dependent* (so it can decide per token whether to overwrite or to hold). That pair of requirements is precisely the selective state-space model, and I propose a Mamba-style **selective diagonal SSM** for this rung.

The derivation runs straight from the continuous linear system underlying structured SSMs: a per-channel scalar state driven by the input, $h'(t) = A\,h(t) + B\,x(t)$, read out as $y(t) = C\,h(t)$, with $A$ a diagonal decay, $B$ the write vector, $C$ the read vector. Discretise with a step $\Delta$ by zero-order hold, and the transition becomes
$$h_t = \bar A\,h_{t-1} + \bar B\,x_t,\qquad \bar A = \exp(\Delta A),\qquad y_t = C\,h_t.$$
Two properties matter. First, the recurrence is *linear* in $h$ — the only thing multiplying $h_{t-1}$ is the number $\bar A$, never a saturating nonlinearity — so the gradient from a read at $t$ back to a write at $t-k$ is a product of $\bar A$ factors, and if $\bar A$ is near 1 that product does not vanish. That is the gradient highway a one-bit memory needs: a write 200 steps back can still reach a read. Second, the original structured SSM keeps $\bar A, \bar B, C$ *constant over time*, which makes the layer a fixed convolution — efficient, but a constant transition cannot tell "this is a write, overwrite" from "this is an ignore, hold," since it ingests $x_t$ the same way either way. Time-invariance is fatal here for the same reason the Transformer's softness was: the model cannot *select*.

So I make the dynamics input-dependent. Let $\Delta_t = \mathrm{softplus}(\mathrm{Linear}(x_t))$ control the step size, $B_t = \mathrm{Linear}(x_t)$ how the input is written, and $C_t = \mathrm{Linear}(x_t)$ how the state is read out; $A$ stays static, since it only ever acts through $\bar A = \exp(\Delta_t A)$ which is already input-dependent through $\Delta_t$, so making $A$ selective too would be redundant. What selecting through $\Delta$ *is* turns out to be the flip-flop solution staring back. Take a leaky integrator ($A=-1$), so $\bar A_t = \exp(-\Delta_t)$. When $\Delta_t$ is large, $\bar A_t \to 0$ and the input drive $\to 1$: the old state is wiped and the current input written — a *write*. When $\Delta_t$ is small, $\bar A_t \to 1$ and the write contribution vanishes: the old state is held — an *ignore*. The data-dependent step size is exactly the write/keep gate the flip-flop language asks for, and it falls out of selective discretisation rather than being bolted on. A write token should learn a large $\Delta$, an ignore a small $\Delta$, and $C_t$ recovers the held bit. A one-bit memory is trivially a recurrent state, so the inductive bias, not capacity, is what was missing — I need very little width or depth.

Grounding this in the task's edit surface, the implementation deliberately differs from the full hardware-aware version in two honest ways, because the harness does not expose the machinery that makes the fast version fast and using it here would only add risk for no benefit at $T\le 1024$. First, canonical selective SSMs recover parallelism by rewriting the linear recurrence as an *associative parallel scan* fused with the discretisation and read-out into one GPU kernel so the expanded state never leaves on-chip memory. I cannot ship a custom CUDA kernel into the container and I do not need to: at $T\le 1024$ with a tiny state, an explicit `for t in range(seq_len)` loop over a `[batch, d_inner, d_state]` state tensor is mathematically identical to the scan and well inside budget — same arithmetic, different memory choreography, no loss of correctness. Second, to keep that loop cheap the model is deliberately small: one block, $d\_model=64$, $d\_state=8$, expansion factor 2. A 1-bit register does not need a 16-wide state or a deep stack.

The block follows the standard selective-SSM design in plain PyTorch. Embed the six tokens to width 64; the block pre-norms its input and projects up to two branches of width $d\_inner = 2\cdot 64$, a main branch $x$ and a gate branch $z$. The main branch passes through a short causal depthwise conv (kernel 3, padded then right-truncated so it stays causal) and a SiLU, giving cheap local mixing — useful because an instruction and its bit are adjacent. From the conv output I project $(\Delta_{\text{raw}}, B, C)$, with $\Delta = \mathrm{softplus}(\text{dt\_proj}(\Delta_{\text{raw}}))$ the per-channel step size and $B, C$ the per-position write/read vectors of width $d\_state$. The static decay is stored as $A\_log$ and used as $A=\exp(A\_log)>0$, so the transition $\bar A = \exp(-\Delta A)$ decays the state — large $\Delta$ decays fast (overwrite), small $\Delta$ holds. The discretised input drive uses the first-order update $\bar B x \approx \Delta\,B\,x$. The selective scan runs $h_t = \bar A_t\,h_{t-1} + (\Delta_t B_t x_t)$, reads out $y_t = (h_t\,C_t).\text{sum(state)}$, adds a per-channel skip $D\cdot x$, and multiplies by $\mathrm{SiLU}(z)$ before projecting back to width 64. A final LayerNorm and bias-free head give the six logits. It is causal by construction (the loop uses only $h_{t-1}$, the conv is right-truncated), handles $T=1024$ (the loop length is just the sequence length), and is far under the parameter cap.

What I expect this to fix, and where the risk is: because the bit is now carried in state and read out locally rather than re-selected by soft attention, the compounding sequence-error should *collapse* — dense should climb from 0.23 toward 1, and long_ctx should recover most dramatically from 0.002, since a recurrence has *no* positional-extrapolation problem at all (the same update at every position, so $T=1024$ is not OOD in any way the model cares about). Sparse is where I am least sure, and the reason is specific to this stripped-down implementation: holding a single write across 100+ ignores demands $\Delta\approx 0$ (hence $\bar A\approx 1$) very precisely on every ignore, and with only $d\_state=8$, one block, and the first-order ZOH approximation, a small systematic leak in $\bar A$ over a long run could still flip the held bit. So I expect sparse to improve over the Transformer but possibly not all the way to zero — and if so, the next rung's lesson is already written: the direction is right, but a more saturating gate that holds a discrete bit *exactly* across arbitrary spans would close the last gap.

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
