Looping closed the two solvable environments and confirmed the depth diagnosis, but it left the real
problem standing. `memory_unit` stayed 1.000 and `grid_world` jumped from 0.882 to a clean 1.000 — so the
shallow model's `grid_world` leak *was* a depth deficit, and six composition stages were enough to fold in
the wall-clamping nonlinearity, exactly as I predicted. But `random_dfa` moved only from 0.205 to 0.309.
Let me quantify what that costs and buys, because the arithmetic is the whole argument for changing axes. The
aggregate is the geometric mean `(1.000 · 1.000 · 0.309)^{1/3} = 0.309^{1/3} = 0.676`, which is what comes
back — and note that with the two solvable environments now pinned at exactly 1.0, the geometric mean has
collapsed to the *cube root of the single `random_dfa` number*. There is nowhere left to hide: from here the
aggregate is `random_dfa^{1/3}` and literally nothing else moves it. So the entire task is now one number.

Now weigh what depth bought. Going from one stage to six lifted `random_dfa` by `0.309 − 0.205 = 0.104`,
i.e. about `0.021` of accuracy per extra composition stage averaged over the five I added — and that is the
*early, steep* part of the depth curve, where each stage should help most. The `O(log T)` story says the
curve flattens past `log_2 T ≈ 5.3`, so the sixth-to-twelfth loops would each buy less than `0.021`, and
doubling to twelve loops (also doubling wall time from the `293` s `random_dfa` run toward `~590` s) might
plausibly reach only the high-0.3s or low-0.4s — still nowhere near the 1.0 the solvable environments sit at,
and still the sole bottleneck. So "more parallel depth" is chasing a flattening asymptote that, for a
length-40 run of a non-solvable group, provably sits below 1.0. The shape of this result is the whole lesson:
adding parallel composition stages helps decisively on the solvable environments where the chain collapses,
and helps only marginally on the non-solvable one where it does not. I have to stop trying to *shortcut* the
composition and instead *perform* it.

Let me state the barrier precisely, because the next architecture has to dodge it rather than push against
it. Liu et al. 2022 says a non-solvable semiautomaton has no constant-depth attention simulator unless
`TC^0 = NC^1`, and that the unavoidable cost is `O(log T)` depth. Both statements are about *parallel*
models — models that apply a fixed number of mixing stages to all positions at once. The looped Transformer,
even with weight tying, is still such a model: six loops is six parallel prefix-scan stages, and the theorem
caps what six stages can do on `S_5`. Recall *why* six was the magic number last rung: an associative scan
doubles its reach per stage, so `d` stages cover `2^d` positions and `⌈log_2 40⌉ = 6` covers the whole
prefix. But covering the prefix in the reachability sense is not the same as *computing* the composition
accurately when the group is non-solvable — the scan can route the operators together, yet the theorem says a
bounded-depth circuit cannot actually multiply them out for an `S_5`-class table. The class of models the
theorem says *nothing* about is the one that gives up parallelism in the time axis entirely: a recurrence
that reads the symbols one at a time and applies an *exact* update per token, `O(T)` strictly sequential
stages. A length-40 run then gets 40 composition stages — one per symbol — which is far more than `O(log T)`
and, crucially, is the *right* number, because the automaton's own definition is exactly "apply the
transition operator once per symbol." A model that does one update per token never tries to compress the
composition into few stages, so the non-solvability barrier, which is entirely about compression, simply does
not apply. The cost is the `O(T)` sequential depth I worked to avoid at the start of the climb — but the
harness budgets ~30 minutes per environment and the sequences are only 40 long, so 40 serial steps is cheap:
the looped model already spent `293` seconds on `random_dfa` doing six full attention passes over 40
positions, and an `O(T)` recurrence over 40 short steps is comparable, comfortably inside the `~1800`-second
cap. The thing I treated as the enemy in the shallow probe is the correct tool for the hard environment.

So I want a recurrent net whose hidden state *is* the simulated automaton state. Before I reach for a
specific cell, let me weigh the recurrent options, because "use a recurrence" is not yet a design. The plain
Elman RNN `h_t = tanh(W[x_t, h_{t-1}])` is the minimal one; a GRU folds the gating into two gates; the LSTM
uses three plus a separate linear cell channel. The distinguishing question is which of these actually
carries a state error across a 40-step lag, because that is what `random_dfa` demands — the state at position
40 depends on all 40 symbols including the first. So the obstacle that decides between them is the
through-time gradient. Follow one error signal back `q` steps and it arrives as a product of `q` factors
`f'·w`; with the logistic sigmoid `f'` peaks at 0.25, so every factor is below 1 whenever `|w| < 4`, and the
product decays geometrically. Concretely, even a favorable per-step factor of `0.9` over 40 steps gives
`0.9^{40} ≈ 0.015`, and a more typical `0.5` gives `0.5^{40} ≈ 9·10^{-13}` — the error from a symbol 40 steps
ago is annihilated, and the net cannot learn the long-range transitions. The plain RNN sits squarely in this
regime, which is exactly why it fails on long dependencies. On `random_dfa` that decay is fatal: the state at
position 40 depends on all 40 symbols, so I need gradient to survive the full lag. Neither bigger weights
(they saturate `f'` faster than they grow, pushing the factor back below 1) nor a bigger learning rate (it
scales near and far credit equally, so the *exponent* is untouched) fixes the geometric decay. The cure is to
make the through-time multiplier exactly 1: solve `f'·w = 1`, which forces the memory channel to be a
*linear* unit with a fixed self-loop of weight one — the constant error carousel. Error rides that self-loop
at unit gain for any number of steps, neither vanishing nor exploding. That requirement — a linear
unit-gain memory channel — is precisely what the LSTM's cell provides and the plain RNN and (to a lesser
degree) the GRU do not have in as clean a form, so the LSTM is the recurrence the barrier is pointing me at.

One more sequential family before committing: a selective state-space model, a diagonal *linear* recurrence
`c_t = A⊙c_{t-1} + B⊙x_t` with input-dependent `A, B`. It shares the LSTM's `O(T)` structure, but its per-step
state map stays *linear* between the gates — a scaled add — whereas a non-solvable group's `δ` is a genuinely
nonlinear permutation-composition on 60 states. The LSTM's `h_t = o_t⊙tanh(c_t)` with gates depending
nonlinearly on `[x_t, h_{t-1}]` makes the effective per-token map a full nonlinear function of state and
symbol, exactly the shape of `δ`; on `T=40` the SSM's linear-scan speed buys little, so the more expressive
gated recurrence is the right call. That leaves the LSTM, and I should say why its specific gate structure is
forced rather than chosen.

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

and the backward pass carries the state error as `ε_s^t = o_t⊙tanh'(c_t)⊙ε_h^t + f_{t+1}⊙ε_s^{t+1}` — the
per-step multiplier is now `f_{t+1}` rather than `f'·w`, so with the forget gate saturated open the 40-step
product is `≈ 1` instead of the `9·10^{-13}` from the plain RNN. The gate can *choose* unit gain on the
channels that must remember and decay on the channels that must reset, per coordinate and per step — precisely
the long-range credit assignment `random_dfa` needs: a symbol at step 1 can influence the loss at step 40, and
its gradient survives the 40-step trip. And the cell *is* a
learned finite-state register: `c_t` holds the running state, the gates implement a learned `δ` that reads the
current symbol and the current state and produces the next state. This is the architecture that matches the
automaton's structure exactly — one exact update per token.

Now I ground this in *this task's* contract. The per-position target `[B, T, num_states]` wants a prediction
at *every* step, so I read out *all* the hidden states, not the last: `out, _ = lstm(embed(input_ids))` gives
`[B, T, hidden]`, and a linear head maps every position's hidden vector to `num_states` logits — the LSTM
emits a hidden state after every symbol, exactly when a state prediction is due, so the output and the
per-token loss line up with no bottleneck or reshaping. The token embedding is width 64 (the gates read both
`x_t` and `h_{t-1}`, so it only has to carry symbol identity, and eight symbols need nowhere near 64
dimensions), the hidden dim is 128, and a single layer suffices — one layer already gives one exact
composition per token; stacking more would re-add parallel-style depth, but sequential depth is what solves the
task. The whole model is on the order of 0.1M parameters, actually smaller than the Transformers. And there is
no causal mask to manage: a forward recurrence is inherently causal — `h_t` depends only on `x_{1:t}` — so the
prefix-only dependence the Transformers enforced with a mask comes for free here.

The optimizer needs a different touch than the Transformers, and the reason is structural, not cosmetic. The
LSTM's carousel makes the loss surface much more forgiving of a larger step — the gradient is well-scaled
across the lag rather than vanishing — so I run AdamW at `lr=1e-3`, an order of magnitude above the
Transformers' `3e-4`, which is the recurrent recipe that converges fast on these online streams. The
Transformers needed the smaller step because a deep or looped attention stack has a sharper, more curved loss
surface where `1e-3` risks overshoot; the LSTM's flatter through-time gradient tolerates the bigger step and
uses it to reach the early-stop threshold sooner. And I set weight decay to essentially zero (`wd=1e-9`): the
gates' biases and the carousel's near-identity dynamics are delicate, and shrinking the recurrent weights
toward zero fights the very unit-gain memory channel I built the cell around — decaying the forget-gate
weights would pull `f_t` off its saturated-open value toward `σ(0)=0.5`, turning the per-step multiplier from
`≈1` into `≈0.5` and reintroducing the `0.5^{40}` decay I just eliminated, exactly the wrong prior for a task
that must remember 40 symbols. The harness's gradient clipping at 1.0 handles the exploding side of the
through-time analysis (a bad batch can still spike a gradient even with the carousel, since unit gain cuts
both ways), so I do not need to add my own clip. `forward_logits` stays the plain `model(input_ids)` — the
recurrence is inside the module's forward.

The falsifiable expectations against the looped Transformer's numbers. `memory_unit` should be 1.000 — the
LSTM is the canonical solver of a memory cell, its forget/input gates *being* the noop/write logic (`f_t=1,
i_t=0` holds the cell on a `noop`; `f_t=0, i_t=1` overwrites it with `j` on a `write(j)`). `grid_world` should
be 1.000 — a sequential cell holding the position and applying `pos ← clamp(pos±1, 0, 8)` per symbol re-derives
the state at every step, so it separates the pathological pair `[R,R,R,R,R,L]`/`[R,R,R,R,L,R]` (both net
displacement `+4`) that a permutation-symmetric single stage could not, with no summary-statistic
reconstruction and none of the running-extremum trouble that made a single parallel stage leak. The decisive
test is `random_dfa`: because the LSTM performs one *exact* state update per symbol and its gradient survives
the full 40-step lag, I expect it to clear the looped Transformer's 0.309 by a clear margin — this is the
environment where the `O(T)`-sequential model is supposed to beat every parallel one, and if it does not, the
whole "perform the composition instead of shortcutting it" thesis is wrong. What I am *not* claiming is 1.0: 60
states, a random table, and 40-step walks may leave residual error on rarely-visited states. The table has
`60·8 = 480` transitions to learn and the online stream delivers `12000 × 64 × 40 ≈ 3·10^7` symbol
observations, so under uniform visitation each transition would be seen `~6·10^4` times — abundant. But
visitation on a random 60-state table is not uniform: a length-40 walk rarely reaches the deep tail of states,
so their outgoing transitions get far below-average signal. That data-coverage tail — not a capacity or
gradient failure, which the carousel rules out — is what should keep `random_dfa` short of 1.0. So the honest
claim is bounded: the LSTM should be the *strongest* of the three, its strength coming precisely from trading
parallel depth for sequential exactness, with the last stretch to 1.0 gated by how often the stream exercises
the rare transitions.
