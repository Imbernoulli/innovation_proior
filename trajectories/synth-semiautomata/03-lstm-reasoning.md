Looping closed the two solvable environments and confirmed the depth diagnosis, but it left the real
problem standing. `memory_unit` stayed 1.000 and `grid_world` jumped from 0.882 to a clean 1.000 — so the
shallow model's `grid_world` leak *was* a depth deficit, and six composition stages were enough to fold in
the wall-clamping nonlinearity, exactly as I predicted. But `random_dfa` moved only from 0.205 to 0.309.
That is a real lift — effective depth 6 is provably past the constant-depth floor where nothing is possible
— but it is nowhere near solved, and the geometric mean (0.676) is still pinned by that third number. The
shape of this result is the whole lesson: adding parallel composition stages helps on the solvable
environments where the chain collapses, and helps only marginally on the non-solvable one where it does not.
Six loops bought 0.10 of accuracy on `random_dfa`; the `O(log T)` curve and the wall-time budget both say
that doubling the loops again would buy less, and I would still be chasing an asymptote that, for a length-40
run of a non-solvable group, sits well below 1.0. So "more parallel depth" is the wrong axis. I have to stop
trying to *shortcut* the composition and instead *perform* it.

Let me state the barrier precisely, because the next architecture has to dodge it rather than push against
it. Liu et al. 2022 Thm 4 says a non-solvable semiautomaton has no constant-depth attention simulator unless
`TC^0 = NC^1`, and Thm 1 says the unavoidable cost is `O(log T)` depth. Both theorems are about *parallel*
models — models that apply a fixed number of mixing stages to all positions at once. The looped Transformer,
even with weight tying, is still such a model: six loops is six parallel stages, and the theorem caps what
six stages can do on `S_5`. The class of models the theorem says *nothing* about is the one that gives up
parallelism in the time axis entirely: a recurrence that reads the symbols one at a time and applies an
*exact* update per token, `O(T)` strictly sequential stages. A length-40 run then gets 40 composition stages
— one per symbol — which is far more than `O(log T)` and, crucially, is the *right* number, because the
automaton's own definition is exactly "apply the transition operator once per symbol." A model that does one
update per token never tries to compress the composition into few stages, so the non-solvability barrier,
which is entirely about compression, simply does not apply. The cost is the `O(T)` sequential depth I worked
to avoid at the start of the climb — but the harness budgets ~30 minutes per environment and the sequences
are only 40 long, so 40 serial steps is cheap. The thing I treated as the enemy in the shallow probe is the
correct tool for the hard environment.

So I want a recurrent net whose hidden state *is* the simulated automaton state. The obstacle that killed
plain recurrence historically is the through-time gradient. Follow one error signal back `q` steps and it
arrives as a product of `q` factors `f'·w`; with the logistic sigmoid `f'` peaks at 0.25, so every factor is
below 1 whenever `|w| < 4`, and the product decays geometrically — the error from a symbol 40 steps ago is
exponentially attenuated, and the net cannot learn long-range transitions. On `random_dfa` that is exactly
the regime that matters: the state at position 40 depends on all 40 symbols, including the first, so I need
gradient to survive the full lag. Neither bigger weights (they saturate `f'` faster than they grow) nor a
bigger learning rate (it scales near and far credit equally) touches the exponent. The cure is to make the
through-time multiplier exactly 1: solve `f'·w = 1`, which forces the memory channel to be a *linear* unit
with a fixed self-loop of weight one — the constant error carousel. Error rides that self-loop at unit gain
for any number of steps, neither vanishing nor exploding.

A bare linear self-loop cannot be wired to the rest of the net, though, because a single incoming weight has
to both *write* the relevant symbol when it arrives and *protect* the stored state when irrelevant symbols
come through the same connection — two opposed jobs for one number — and a single outgoing weight has the
mirror conflict on the read side. A weight is one number and cannot be context-sensitive; another *unit*
can. So gate the carousel multiplicatively: an input gate `i_t` in `[0,1]` decides how much of the candidate
update is written, an output gate `o_t` decides how much of the state is read out, and — the piece that
makes it work on a stream that must reset between runs — a forget gate `f_t` that multiplies the carried
state, recovering the exact carousel when `f_t=1` and wiping the cell when `f_t=0`. Multiplicative, not
additive, because protecting the memory means letting through *exactly zero* of an irrelevant input, which
only a multiply can do. The cell is then

`i_t = σ(W_i[x_t,h_{t-1}])`, `f_t = σ(W_f[x_t,h_{t-1}])`, `g_t = tanh(W_g[x_t,h_{t-1}])`,
`c_t = f_t⊙c_{t-1} + i_t⊙g_t`, `o_t = σ(W_o[x_t,h_{t-1}])`, `h_t = o_t⊙tanh(c_t)`,

and the backward pass carries the state error as `ε_s^t = o_t⊙tanh'(c_t)⊙ε_h^t + f_{t+1}⊙ε_s^{t+1}` — unit
gain across the lag when the forget gate is open. That is precisely the long-range credit assignment
`random_dfa` needs: a symbol at step 1 can influence the loss at step 40, and its gradient survives the
40-step trip. And the cell *is* a learned finite-state register: `c_t` holds the running state, the gates
implement a learned `δ` that reads the current symbol and the current state and produces the next state.
This is the architecture that matches the automaton's structure exactly — one exact update per token.

Now I have to ground this in *this task's* contract, and the shape is different from a generic recurrent
regressor. The harness wants per-position state logits `[B, T, num_states]` — a prediction at *every* step,
not a single readout at the end. So the LSTM must run over the whole symbol sequence and I read out *all* of
its hidden states, not just the last: `out, _ = lstm(embed(input_ids))` gives `[B, T, hidden]`, and a linear
head maps every position's hidden vector to `num_states` logits. This is the natural fit for full state
supervision — the LSTM produces a hidden state after every symbol, which is exactly when a state prediction
is due, so the per-position output and the per-token loss line up with no masking or reshaping. There is no
single-vector bottleneck and no last-step readout here; the recurrence emits a full trajectory. The input is
a single learned token embedding of width 64 (the gates and candidate read both `x_t` and `h_{t-1}`, so the
embedding only has to carry the symbol identity), the hidden dim is 128, and a single LSTM layer suffices —
one layer already gives one exact composition per token, which is the whole point; stacking more would add
parallel-style depth on top of the sequential depth, but the sequential depth is what solves the task, so I
keep it minimal. There is no causal mask to manage: a forward recurrence is inherently causal — `h_t`
depends only on `x_{1:t}` by construction — so the prefix-only-dependence property the Transformers had to
enforce with a mask comes for free here.

The optimizer needs a different touch than the Transformers, and the reason is structural, not cosmetic. The
LSTM's carousel makes the loss surface much more forgiving of a larger step — the gradient is well-scaled
across the lag rather than vanishing — so I run AdamW at `lr=1e-3`, an order of magnitude above the
Transformers' `3e-4`, which is the recurrent recipe that converges fast on these online streams. And I set
weight decay to essentially zero (`wd=1e-9`): the gates' biases and the carousel's near-identity dynamics are
delicate, and shrinking the recurrent weights toward zero fights the very unit-gain memory channel I built
the cell around — decaying the forget-gate weights would bias the cell toward forgetting, exactly the wrong
prior for a task that must remember 40 symbols. The harness's gradient clipping at 1.0 handles the exploding
side of the through-time analysis (a bad batch can still spike a gradient even with the carousel), so I do
not need to add my own clip. `forward_logits` stays the plain `model(input_ids)` — the recurrence is inside
the module's forward.

The falsifiable expectations against the looped Transformer's numbers. `memory_unit` should be 1.000 — the
LSTM is the canonical solver of a memory cell (its forget/input gates *are* the noop/write logic), and if it
is not 1.0 something is broken. `grid_world` should be 1.000 — the recurrence computes the clamped prefix sum
exactly, one step at a time, with no nonlinearity-folding approximation. The decisive test is `random_dfa`:
because the LSTM performs one *exact* state update per symbol and its gradient survives the full 40-step lag,
I expect it to clear `random_dfa` by a clear margin over the looped Transformer's 0.309 — this is the
environment where the `O(T)`-sequential model is supposed to beat every parallel one, and if it does not, the
whole "perform the composition instead of shortcutting it" thesis is wrong. I expect the LSTM to post the
highest `random_dfa` of the three architectures and therefore the highest geometric mean. What I am *not*
claiming is that it reaches 1.0 on `random_dfa` — 60 states, a random table, 40 steps, and 12000 online
steps may leave the cell with residual error on rarely-visited states — only that it is the strongest, and
that the strength comes precisely from trading parallel depth for sequential exactness. The full scaffold
module is in the answer.
