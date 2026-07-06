Looping closed the two solvable environments and confirmed the depth diagnosis, but it left the real
problem standing. `memory_unit` stayed 1.000 and `grid_world` jumped from 0.882 to a clean 1.000 — so the
shallow model's `grid_world` leak *was* a depth deficit, and six composition stages were enough to fold in
the wall-clamping nonlinearity, exactly as I predicted. But `random_dfa` moved only from 0.205 to 0.309.
Let me quantify what that costs and buys, because the arithmetic is the whole argument for changing axes. The
aggregate is the geometric mean `(1.000 · 1.000 · 0.309)^{1/3} = 0.309^{1/3} = 0.676`, which the leaderboard
confirms — and note that with the two solvable environments now pinned at exactly 1.0, the geometric mean has
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

There is one more sequential family I should weigh before committing, because it is the fashionable one: a
selective state-space model, a diagonal *linear* recurrence `c_t = A⊙c_{t-1} + B⊙x_t` with input-dependent
`A, B`. It shares the LSTM's `O(T)` sequential structure and its parallel-scan trainability, so on the
"perform the composition" axis it is a sibling. But the crux for `random_dfa` is that the state update must
represent an arbitrary transition function `δ` over 60 states, and a diagonal-linear recurrence keeps the
state evolution *linear* between the input-dependent gates — its expressive per-step map is a scaled add,
whereas a non-solvable group's `δ` is a genuinely nonlinear permutation-composition on the state. The LSTM's
`c_t = f_t⊙c_{t-1} + i_t⊙g_t` with `h_t = o_t⊙tanh(c_t)` wraps the carousel in a `tanh` readout and lets the
gates depend nonlinearly on `[x_t, h_{t-1}]`, so the *effective* per-token state map is a full nonlinear
function of the current state and symbol — which is exactly the shape of `δ`. On short `T=40` sequences the
linear-scan speed advantage of an SSM is worth little (40 steps is cheap either way), and its representational
economy is the wrong economy here, so the more expensive but strictly more expressive gated nonlinear
recurrence is the right call. That leaves the LSTM as the target, and I should now say why its specific gate
structure is forced rather than chosen.

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
gain across the lag when the forget gate is open. Let me sanity-check that recurrence against the vanishing
arithmetic that killed the plain RNN: the carried term is `f_{t+1}⊙ε_s^{t+1}`, so the multiplier per step is
`f_{t+1}` rather than `f'·w`, and with the forget gate saturated open (`f ≈ 1`) the 40-step product is
`≈ 1^{40} = 1` instead of the `9·10^{-13}` I computed above. That is the whole point: the gate can *choose*
unit gain on the channels that must remember, and choose decay on the channels that must reset, per
coordinate and per step. That is precisely the long-range credit assignment `random_dfa` needs: a symbol at
step 1 can influence the loss at step 40, and its gradient survives the 40-step trip. And the cell *is* a
learned finite-state register: `c_t` holds the running state, the gates implement a learned `δ` that reads the
current symbol and the current state and produces the next state. This is the architecture that matches the
automaton's structure exactly — one exact update per token.

Now I have to ground this in *this task's* contract, and the shape is different from a generic recurrent
regressor. The harness wants per-position state logits `[B, T, num_states]` — a prediction at *every* step,
not a single readout at the end. So the LSTM must run over the whole symbol sequence and I read out *all* of
its hidden states, not just the last: `out, _ = lstm(embed(input_ids))` gives `[B, T, hidden]`, and a linear
head maps every position's hidden vector to `num_states` logits. Let me verify the shapes so the readout
lines up with the loss: `input_ids` is `[B, 40]`; the embedding gives `[B, 40, 64]`; `nn.LSTM` with
`batch_first=True` returns `out` of shape `[B, 40, 128]` (one hidden vector per input step); the head maps
the last axis `128 → num_states`, giving `[B, 40, num_states]` — exactly the per-token target the
cross-entropy expects, with no masking, slicing, or reshaping. This is the natural fit for full state
supervision — the LSTM produces a hidden state after every symbol, which is exactly when a state prediction
is due, so the per-position output and the per-token loss line up with no bottleneck. There is no
single-vector bottleneck and no last-step readout here; the recurrence emits a full trajectory. The input is
a single learned token embedding of width 64 (the gates and candidate read both `x_t` and `h_{t-1}`, so the
embedding only has to carry the symbol identity, and the eight symbols of `random_dfa` need nowhere near 64
dimensions to separate), the hidden dim is 128, and a single LSTM layer suffices — one layer already gives
one exact composition per token, which is the whole point; stacking more would add parallel-style depth on
top of the sequential depth, but the sequential depth is what solves the task, so I keep it minimal. Let me
count the parameters to confirm this is not a bloated model: each of the four gate/candidate transforms is a
linear over `[x_t, h_{t-1}]` of width `64 + 128 = 192` into 128, so `4 · (192·128 + 128) ≈ 4 · 24.7k ≈ 99k`,
plus a `vocab·64` embedding and a `128·num_states` head — on the order of `0.1M`, actually *smaller* than the
`~0.2M` Transformers, so the sequential win does not come at a parameter premium. There is no causal mask to
manage: a forward recurrence is inherently causal — `h_t` depends only on `x_{1:t}` by construction — so the
prefix-only-dependence property the Transformers had to enforce with a mask comes for free here.

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
LSTM is the canonical solver of a memory cell (its forget/input gates *are* the noop/write logic: on a
`noop` set `f_t=1, i_t=0` to hold the cell, on a `write(j)` set `f_t=0, i_t=1` to overwrite it with `j`), and
if it is not 1.0 something is broken. `grid_world` should be 1.000 — the recurrence computes the clamped
prefix sum exactly, one step at a time: it can increment or decrement the stored position per `R`/`L` and
saturate at the walls, with no nonlinearity-folding approximation and none of the running-extremum trouble
that made a single parallel stage leak. Trace the pathological pair that broke the depth-1 model:
`[R,R,R,R,R,L]` versus `[R,R,R,R,L,R]`, both net displacement `+4` from the middle state 4. A sequential
cell holding the current position and applying `pos ← clamp(pos±1, 0, 8)` per symbol walks the first to
`4,5,6,7,8,8,7` and the second to `4,5,6,7,8,7,8`, correctly separating the final states 7 and 8 — because it
never has to reconstruct the clamp from a summary statistic; it re-derives the position at every step and the
wall saturation is just a per-step `clamp` the cell can implement with its gates. The order-dependence that a
permutation-symmetric single stage could not resolve is trivial for a machine that reads the symbols in
order, which is the whole reason the sequential model closes `grid_world` exactly rather than approximately. The decisive test is `random_dfa`: because the LSTM performs one
*exact* state update per symbol and its gradient survives the full 40-step lag, I expect it to clear
`random_dfa` by a clear margin over the looped Transformer's 0.309 — this is the environment where the
`O(T)`-sequential model is supposed to beat every parallel one, and if it does not, the whole "perform the
composition instead of shortcutting it" thesis is wrong. I expect the LSTM to post the highest `random_dfa`
of the three architectures and therefore the highest geometric mean, since with the two solvable
environments already at 1.0 the aggregate is now exactly `random_dfa^{1/3}` and every point on that number
flows straight to the top line. What I am *not* claiming is that it reaches 1.0 on `random_dfa` — 60 states,
a random table, 40 steps, and 12000 online steps may leave the cell with residual error on rarely-visited
states. Let me reason about whether coverage could be the limiter: the table has `|Q|·|Σ| = 60·8 = 480`
distinct transitions to learn, and the online stream delivers `12000` steps `× 64` batch `× 40` positions
`≈ 3·10^7` symbol observations, so *if* states were visited uniformly each transition would be seen `~6·10^4`
times — abundant. But visitation on a random 60-state table is not uniform: a length-40 walk cannot reach the
deep tail of rarely-entered states often, so those states' outgoing transitions are seen far below the
average, and the cell has correspondingly less signal to fit their `δ`. That is the mechanism that should
keep `random_dfa` short of 1.0 even for an exact sequential model — not a capacity or gradient failure, which
the carousel rules out, but a data-coverage tail. So the honest claim is bounded: the LSTM should be the
*strongest* of the three and post the highest geometric mean, with the strength coming precisely from trading
parallel depth for sequential exactness, while the last stretch to 1.0 is gated by how often the stream
exercises the table's rare transitions. The full scaffold module is in the answer.
