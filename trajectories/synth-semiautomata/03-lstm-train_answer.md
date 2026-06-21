Looping confirmed the depth diagnosis but left the real problem standing. `memory_unit` stayed 1.000 and `grid_world` jumped from 0.882 to a clean 1.000 — so the shallow model's leak *was* a depth deficit, and six composition stages were enough to fold in the wall-clamping nonlinearity. But `random_dfa` moved only from 0.205 to 0.309. That is a real lift — effective depth 6 is provably past the constant-depth floor — yet nowhere near solved, and the geometric mean (0.676) is still pinned by that third number. The shape of the result is the whole lesson: adding parallel composition stages helps where the chain collapses, and helps only marginally where it does not. Six loops bought $0.10$ on `random_dfa`; the $O(\log T)$ curve and the wall-time budget both say doubling the loops would buy less, chasing an asymptote that for a length-40 run of a non-solvable group sits well below 1.0. "More parallel depth" is the wrong axis. I have to stop trying to *shortcut* the composition and instead *perform* it.

The barrier is worth stating precisely, because the next architecture has to dodge it rather than push against it. Liu et al. 2022 Thm 4 says a non-solvable semiautomaton has no constant-depth attention simulator unless $\mathrm{TC}^0 = \mathrm{NC}^1$, and Thm 1 says the unavoidable cost is $O(\log T)$ depth. Both theorems are about *parallel* models — a fixed number of mixing stages applied to all positions at once — and the looped Transformer, weight-tying notwithstanding, is still such a model: six loops is six parallel stages, and the theorem caps what six can do on $S_5$. The class the theorem says *nothing* about is the one that gives up parallelism in the time axis entirely: a recurrence that reads symbols one at a time and applies an *exact* update per token, $O(T)$ strictly sequential stages. A length-40 run then gets 40 composition stages — one per symbol, which is the *right* number, because the automaton's own definition is "apply the transition operator once per symbol." A model that does one update per token never tries to compress the composition into few stages, so the non-solvability barrier, which is entirely about compression, simply does not apply. The cost is the $O(T)$ sequential depth I worked to avoid at the start — but with $T=40$ and a 30-minute budget, 40 serial steps is cheap. The thing I treated as the enemy in the shallow probe is the correct tool for the hard environment.

I propose a single-layer LSTM whose hidden state *is* the simulated automaton state. The obstacle that historically killed plain recurrence is the through-time gradient: follow one error signal back $q$ steps and it arrives as a product of $q$ factors $f'\!\cdot w$; with the logistic sigmoid, $f'$ peaks at $0.25$, so every factor is below 1 whenever $|w|<4$ and the product decays geometrically — the error from a symbol 40 steps ago is exponentially attenuated. On `random_dfa` that is exactly the regime that matters: the state at position 40 depends on all 40 symbols, including the first. Neither bigger weights (they saturate $f'$ faster than they grow) nor a bigger learning rate (it scales near and far credit equally) touches the exponent. The cure is to make the through-time multiplier *exactly 1*: solve $f'\!\cdot w = 1$, which forces the memory channel to be a *linear* unit with a fixed self-loop of weight one — the constant error carousel. Error rides that self-loop at unit gain for any number of steps, neither vanishing nor exploding.

A bare linear self-loop cannot be wired to the rest of the net, though, because a single incoming weight would have to both *write* the relevant symbol when it arrives and *protect* the stored state when irrelevant symbols come through the same connection — two opposed jobs for one number — and a single outgoing weight has the mirror conflict on the read side. A weight is one number and cannot be context-sensitive; another *unit* can. So I gate the carousel multiplicatively: an input gate $i_t \in [0,1]$ decides how much of the candidate update is written, an output gate $o_t$ decides how much of the state is read out, and — the piece that makes it work on a stream that must reset between runs — a forget gate $f_t$ that multiplies the carried state, recovering the exact carousel when $f_t = 1$ and wiping the cell when $f_t = 0$. Multiplicative, not additive, because protecting the memory means letting through *exactly zero* of an irrelevant input, which only a multiply can do. The cell is

$$i_t = \sigma(W_i[x_t, h_{t-1}]), \quad f_t = \sigma(W_f[x_t, h_{t-1}]), \quad g_t = \tanh(W_g[x_t, h_{t-1}]),$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad o_t = \sigma(W_o[x_t, h_{t-1}]), \quad h_t = o_t \odot \tanh(c_t),$$

and the backward pass carries the state error as $\varepsilon_s^t = o_t \odot \tanh'(c_t) \odot \varepsilon_h^t + f_{t+1} \odot \varepsilon_s^{t+1}$ — unit gain across the lag whenever the forget gate is open. That is precisely the long-range credit assignment `random_dfa` needs: a symbol at step 1 can influence the loss at step 40, and its gradient survives the 40-step trip. And the cell *is* a learned finite-state register — $c_t$ holds the running state, the gates implement a learned $\delta$ that reads the current symbol and the current state and produces the next state. This matches the automaton's structure exactly: one exact update per token.

Grounding this in the task's contract changes the shape from a generic recurrent regressor. The harness wants per-position logits $[B,T,\text{num\_states}]$ — a prediction at *every* step, not a single readout at the end — so I read out *all* of the LSTM's hidden states, not just the last: `out, _ = lstm(embed(input_ids))` gives $[B,T,\text{hidden}]$, and a linear head maps every position's hidden vector to $\text{num\_states}$ logits. This is the natural fit for full state supervision: the LSTM produces a hidden state after every symbol, exactly when a state prediction is due, so the per-position output and the per-token loss line up with no masking or reshaping, and there is no single-vector bottleneck. The input is a learned token embedding of width 64 (the gates and candidate read both $x_t$ and $h_{t-1}$, so the embedding only has to carry symbol identity), the hidden dimension is 128 — the state register — and a single layer suffices, because one layer already gives one exact composition per token; stacking more would re-add parallel-style depth on top of the sequential depth that actually solves the task. There is no causal mask to manage: a forward recurrence is causal by construction, $h_t$ depending only on $x_{1:t}$, so the prefix-only-dependence the Transformers enforced with a mask comes for free.

The optimizer needs a different touch, for structural reasons. The carousel keeps the gradient well-scaled across the lag rather than vanishing, which makes the loss surface forgiving of a larger step, so I run AdamW at $\text{lr}=1\mathrm{e}{-3}$, an order above the Transformers' $3\mathrm{e}{-4}$ — the recurrent recipe that converges fast on these online streams. And I set weight decay to essentially zero, $\text{wd}=1\mathrm{e}{-9}$: the gates' biases and the carousel's near-identity dynamics are delicate, and shrinking the recurrent weights fights the unit-gain memory channel the cell is built around — decaying the forget-gate weights would bias the cell toward forgetting, the wrong prior for a task that must remember 40 symbols. The harness's gradient clipping at 1.0 handles the exploding side (a bad batch can still spike a gradient even with the carousel), so I add no clip of my own, and `forward_logits` stays the plain `model(input_ids)`.

The falsifiable expectations against the looped Transformer's numbers: `memory_unit` should be 1.000 (the LSTM is the canonical memory-cell solver — its forget/input gates *are* the noop/write logic), and `grid_world` should be 1.000 (the recurrence computes the clamped prefix sum exactly, one step at a time, with no nonlinearity-folding approximation). The decisive test is `random_dfa`: because the LSTM performs one exact update per symbol and its gradient survives the full 40-step lag, I expect it to clear the looped Transformer's 0.309 by a clear margin and post the highest geometric mean of the three architectures. I am *not* claiming it reaches 1.0 there — 60 states, a random table, 40 steps, and 12000 online steps may leave residual error on rarely-visited states — only that it is the strongest, and that the strength comes precisely from trading parallel depth for sequential exactness.

```python
class CustomSequenceModel(nn.Module):
    """Single-layer LSTM (Liu et al. 2022 App. B.3: emb=64, hidden=128)."""

    def __init__(self, vocab_size: int, num_states: int, seq_len: int,
                 emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_states)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        h, _ = self.lstm(x)
        return self.head(h)


def build_model(env_spec: EnvSpec, config: TaskConfig) -> nn.Module:
    """Single-layer LSTM, emb=64, hidden=128."""
    return CustomSequenceModel(
        vocab_size=env_spec.alphabet_size,
        num_states=env_spec.num_states,
        seq_len=env_spec.seq_len,
        emb_dim=64,
        hidden_dim=128,
        num_layers=1,
    )


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """AdamW (Liu et al. 2022 App. B.3: LSTM uses lr 1e-3, wd 1e-9)."""
    return {"lr": 1e-3, "wd": 1e-9, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    return model(input_ids)
```
