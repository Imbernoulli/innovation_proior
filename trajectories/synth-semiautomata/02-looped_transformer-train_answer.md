The depth-1 numbers came back exactly along the complexity boundary. `memory_unit` is 1.000 — the constant-function semigroup is a single copy-from-last-write head, solved exactly. `grid_world` is 0.882, high but visibly short of 1.0 — the prefix-sum part is easy, but the wall-clamping nonlinearity did not fully close in one mixing step, the small leak I worried about. And `random_dfa` is 0.205 — barely above chance-plus-short-range-structure on the non-solvable environment. The geometric mean is 0.566, and because it is a geometric mean the 0.205 is what pins it: two environments are essentially perfect, the third drags the aggregate down. So the entire problem is now concentrated in one place, and the diagnosis is specific. The 0.205 is not an optimization failure — `memory_unit` hitting 1.0 proves the loop works and the online stream rules out overfitting. It is a *capacity* failure of one kind: simulating a length-$T$ automaton run is inherently a chain of $T$ compositions, and for a non-solvable transition semigroup that chain provably does not collapse below $O(\log T)$ attention-mixing stages (Liu et al. 2022, Thm 4). One layer is one stage. The missing resource is effective depth.

I propose the Looped Transformer: hold a single shared encoder block and apply it `n_loops` times to the embedded sequence, feeding each output back as the next input. The obvious way to add depth is to stack six distinct layers, but I want the depth *without* paying for it in parameters — partly to keep the comparison with the shallow baseline honest, and more importantly because a *shared* block is the structurally correct object here. An automaton applies the *same* transition operator at every step; the computation is not six different transformations but one transformation iterated. A network that ties the same weights across all its sequential stages matches that exactly: it learns one "advance the simulation one round" operator and applies it repeatedly. The depth is the number of loops; the parameter count is that of a one-layer model. There is also a regularization argument for tying over stacking: six independent layers have six times the freedom and could fit the stream with a brittle six-stage pipeline, whereas a shared block forces every stage to be the *same* map — so whatever it learns must be a function that improves the representation when applied once and is safe to apply again, which is precisely the property "apply the transition operator" has. The shared block is therefore not merely cheaper; it is the hypothesis class that *contains* "iterate one operator" and excludes most everything else, and on a task that is operator iteration that is the bias I want.

I am careful to import only the right part of the looping idea. Its strong form is a programmable-computer construction — lay the input out as scratchpad/memory/command regions, give every column a binary address, turn attention into addressed reads and writes so the looped block executes one instruction per pass. This task needs none of that: the harness hands a fixed $[B,T]$ symbol stream and wants $[B,T,\text{num\_states}]$ logits, and the editable contract is just `build_model` returning a module. So the faithful, minimal realization of "depth from looping a shared block" is: embed the tokens once, apply one shared causal self-attention encoder layer `n_loops` times, then a closing norm and the linear state head. No memory layout, no addressing, no temperature-sharpened permutation. The looping supplies the sequential composition stages a non-solvable automaton needs; the weight sharing supplies them at one-layer cost.

Concretely, the embedding is unchanged from the shallow baseline — learned token + absolute position embeddings, added — because the order-injection argument is identical and the length is still 40. The novelty is the body: instead of `nn.TransformerEncoder` with `num_layers=1`, I hold a single `nn.TransformerEncoderLayer` (pre-norm, GELU, $4\times$ MLP, $d_{\text{model}}=128$, 4 heads) and call it in a Python loop, `for _ in range(n_loops)`, threading the causal mask through *every* iteration. That last detail matters: the state at $t$ may only ever depend on the prefix $\le t$, and that must hold at each round of composition, not just the first, so the mask is applied on every pass. After the loops I apply a final $\mathrm{LayerNorm}$ before the head, because pre-norm leaves the residual stream un-normalized at the top of the stack and six residual additions through the *same* block can let the scale drift; one closing norm keeps the head's input on a sane scale. The `forward_logits` wrapper stays the plain `model(input_ids)` — the looping lives inside the module's forward, so the scratchpad-style escape hatch is unneeded.

On the loop count: the depth-vs-difficulty pattern for non-solvable groups grows like $O(\log T)$, with accuracy climbing steeply in depth and a depth-6 model landing high on the milder non-solvable groups but only in the high-20s to low-30s on the harder $S_5$-class regime. With $T=40$, $\log T \approx 5.3$, so six loops is the smallest depth comfortably past the $O(\log T)$ threshold while staying inside the wall-time budget — six passes is roughly $6\times$ the shallow compute, which the budget tolerates. I keep the AdamW recipe at $\text{lr}=3\mathrm{e}{-4}$, $\text{wd}=1\mathrm{e}{-4}$, unchanged from the shallow model, both because a looped shared block's optimization geometry under pre-norm is close enough to a shallow stack and because changing the schedule would confound the depth comparison.

The falsifiable expectations, stated against the depth-1 numbers I am trying to beat: `memory_unit` should stay 1.000 (the extra loops can act as near-identity after the first), and a drop would mean the loop is destabilizing the easy case. `grid_world` is the clean test of the depth hypothesis — if its 0.882 leak was one mixing step being too few for the clamping nonlinearity, six composition stages should close it to $\approx 1.0$. The real target is `random_dfa`: I expect a clear jump above the 0.205 floor because the depth is now provably in the regime where some simulation is possible, but I do *not* expect it solved, because $S_5$ is non-solvable and six loops is only just past $O(\log T)$ — somewhere in the 0.3 range, still the bottleneck. If looping closes the two solvable environments and lifts `random_dfa` only partway, the lesson is that even effective depth 6 is structurally short of what a non-solvable group demands on a length-40 run, and the next move is not more loops but a model that gives up parallel composition stages entirely — an $O(T)$-sequential recurrence that performs one exact update per token.

```python
class CustomSequenceModel(nn.Module):
    """Looped Transformer: one shared encoder layer applied n_loops times."""

    def __init__(self, vocab_size: int, num_states: int, seq_len: int,
                 d_model: int = 128, n_heads: int = 4, n_loops: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_loops = n_loops
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = self.causal_mask[:T, :T]
        for _ in range(self.n_loops):
            x = self.block(x, src_mask=mask, is_causal=True)
        x = self.final_norm(x)
        return self.head(x)


def build_model(env_spec: EnvSpec, config: TaskConfig) -> nn.Module:
    """Looped Transformer: shared block, 6 loops, d_model=128, 4 heads."""
    return CustomSequenceModel(
        vocab_size=env_spec.alphabet_size,
        num_states=env_spec.num_states,
        seq_len=env_spec.seq_len,
        d_model=128,
        n_heads=4,
        n_loops=6,
        dropout=0.0,
    )


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """AdamW (Liu et al. 2022 App. B.3 GPT-2 recipe: lr 3e-4, wd 1e-4)."""
    return {"lr": 3e-4, "wd": 1e-4, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    return model(input_ids)
```
