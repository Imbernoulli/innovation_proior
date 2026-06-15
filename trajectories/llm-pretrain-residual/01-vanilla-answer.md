**Problem.** Design the residual-flow rule of a 24-layer Pre-LN GPT-2 Medium on FineWeb. Before
designing anything, fix the floor: the strongest *plain* residual rule, so any later redesign is judged
against the honest best-simple-thing, not a strawman.

**Key idea (the fixed floor).** The vanilla additive Pre-LN residual: each block reads a normalized
copy of the stream and writes its branch back with coefficient one, `x = x + attn(LN(x))` then
`x = x + mlp(LN(x))`, a clean identity-plus-addition highway (backward `1` restored, last-layer
gradient `∼1/√L`), with one final LayerNorm before the head. This is the modern default — the
resolution of plain→residual→Post-LN→Pre-LN — so it is the right floor.

**Step-1 edit.** The whole loop is the scaffold's fixed substrate. The only editable residual logic is
left at the **default**: the block loop is the plain `for block in self.transformer.h: x = block(x)`
(an identity replacement to satisfy the one-edit-op requirement). No new parameters, no optimizer
change, no config override.

**Why it is the floor.** The Pre-LN stream is a *fixed unit-weight accumulator*:
`x_l = x_1 + Σ_{i<l} F_i(LN(x_i))` — every branch added with coefficient exactly one, the same for
every token, with no knob on the depth axis. The stream variance climbs with depth (shrinking deep-block
Jacobians toward identity, so the deepest layers go half-dead), and the branch fires full-strength from
init (the block is not identity at step zero, which is why warm-up exists). No mechanism weights,
schedules, or re-routes the depth flow — that rigidity is exactly the seam every later rung pulls on.

**What to watch.** A healthy run: validation loss in the low-2.2s, sensible WikiText-2 / LAMBADA
perplexity, mid-50s ARC-Easy and low-30s HellaSwag. Whatever the numbers, the next rung must keep this
clean highway while giving the depth flow a knob — starting with the cheapest, near-parameter-free
change to *when and how much* each branch writes.

```python
# EDITABLE regions of custom_pretrain.py — step 1: vanilla Pre-LN residual (default)

# Block: scaffold default (unchanged).
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT.__init__ residual region: default (no extra parameters).

# GPT.forward block loop — the only edited region (identity replacement of the loop with itself):
def _gpt_forward_block_loop(self, x):
    # ── Residual stream: iterate through transformer blocks ──
    for block in self.transformer.h:
        x = block(x)
    return x

# GPT.configure_optimizers: default (no new param groups).
# CONFIG_OVERRIDES = {}   (no LR / weight-decay override).
```
