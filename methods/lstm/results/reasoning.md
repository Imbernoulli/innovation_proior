Let me start from the thing that actually fails. I have a recurrent net, and the whole reason I reach for a recurrent net rather than a feedforward one is that its feedback connections can hold information across time: the activation pattern now can encode an input from many steps ago, and the network can learn to use it. That is the promise — short-term memory in the activations. And in practice it half works: if the relevant input and the moment its information is needed are only a few steps apart, BPTT or RTRL trains fine. But the moment the lag stretches out — tens, hundreds, a thousand steps — training either stalls completely or crawls so slowly it never gets anywhere. I want to understand *why*, precisely, because "recurrent nets are hard to train on long dependencies" is a symptom, not a cause, and I cannot fix a symptom.

So let me follow a single error signal as it flows backward through time and watch what happens to it. Conventional recurrent backprop: an output unit `k` at time `t` has error `δ_k(t) = f_k'(net_k(t))(d_k(t) - y_k(t))`, and any non-output unit `j` gets its error by collecting what flows back from the next step, `δ_j(t) = f_j'(net_j(t)) Σ_i w_ij δ_i(t+1)`, where `net_i(t) = Σ_j w_ij y_j(t-1)` and `f_i` is the unit's squashing function. Now take the error that lands on some unit `u` at time `t`, and ask how much of it reaches a unit `v` after being propagated `q` steps back into the past. For `q = 1` it is just `f_v'(net_v(t-1)) w_uv`. For `q > 1` I have to recurse: the error at `v`, `t-q`, is `f_v'(net_v(t-q))` times the sum over intermediate units `l` of `w_lv` times the error that reached `l` at `t-q+1` on its way from `u`. Spell that recursion all the way out and it telescopes into a product over the path:

  ∂δ_v(t-q)/∂δ_u(t) = Σ_{l_1} … Σ_{l_{q-1}} Π_{m=1}^q f'_{l_m}(net_{l_m}(t-m)) w_{l_m l_{m-1}},

with `l_0 = u` and `l_q = v`. I can check this by induction — it is right for `q=1`, and the recursion clearly appends one more `f'·w` factor and one more summation index each step. So the error that survives `q` steps is a sum over all `n^{q-1}` paths from `u` to `v`, and each path contributes a *product of `q` factors*, each factor being some `f'(net) · w`.

Now stare at that product. A product of `q` numbers, where `q` is the lag. If every factor on a path has magnitude greater than 1 — which can happen, e.g. with a near-linear unit and a weight above 1 — then the product grows like (something>1)^q: the error *explodes* exponentially in the lag, and arriving error signals start to oscillate and bifurcate; learning is unstable. And if every factor has magnitude below 1 — the ordinary case — the product *shrinks* like (something<1)^q: the error *vanishes* exponentially. Either way the lag `q` sits in the exponent. There is no middle setting of the weights that makes a generic product of `q` factors stay `O(1)` for large `q`; a product of many numbers each a bit under one is tiny, each a bit over one is huge. Exponential by construction.

Let me pin the vanishing case down quantitatively for the squashing function I actually use, the logistic sigmoid, because I want to know how bad it is. The sigmoid's derivative peaks at `0.25`. So a single factor `|f'(net) w|` is at most `0.25 |w|`, which is below 1 whenever `|w| < 4.0`. Any reasonable weight is below 4, so each factor is below 1, so the product over `q` steps decays geometrically — the contribution of a long-ago input to today's loss is exponentially small. Let me put real numbers on "exponentially small," because "small" could mean anything. Take a fairly *large* weight, `|w| = 2`, so the per-step factor is `0.25·2 = 0.5` — already generous, this is not even the worst case. Over a lag of `q = 10` the surviving factor is `0.5^10 ≈ 9.8e-4`; over `q = 50` it is `0.5^50 ≈ 8.9e-16`; over `q = 100`, `0.5^100 ≈ 7.9e-31`. At `q = 50` the error from an input fifty steps ago arrives at machine-epsilon scale relative to a recent one — it is *gone*, drowned in floating-point noise, long before I reach the thousand-step lags I actually want. And `|w|=2` flatters the situation; the typical small initial weight makes it far worse (at `|w|=0.5` the factor is `0.125`, and `0.125^10 ≈ 9.3e-10` already). So this is not a slow leak I can outrun with more epochs; the signal is annihilated. I can bound the whole thing including the number of units `n`. With the matrix norm `||W||_A := max_r Σ_s |w_rs|` and vector norm `||x|| := max_r |x_r|`, the per-step factor in operator form is at most `f'_max ||W||_A` with `f'_max = 0.25`, and `|∂δ_v(t-q)/∂δ_u(t)| ≤ n (f'_max ||W||_A)^q`. If I keep `|w_ij| ≤ w_max < 4/n`, then `||W||_A ≤ n w_max < 4`, and setting `τ := n w_max / 4 < 1` gives `|∂δ_v(t-q)/∂δ_u(t)| ≤ n τ^q` — exponential decay in the lag, full stop. (The other branch is no better: a near-linear unit with `w = 1.2` gives `1.2^50 ≈ 9.1e3`, a factor of ten thousand — the error doesn't just survive, it blows up and oscillates. There is no comfortable middle.)

And here is the part that kills the obvious escape routes. Maybe I just use bigger weights to push the factor above one? No — a bigger `w` drives the unit toward saturation, where `f'` collapses toward zero *faster* than `w` grows, so the product gets *worse*, not better. Maybe a bigger learning rate? No — that scales the long-range error and the short-range error by the same factor, so the *ratio* of long-range to short-range credit is unchanged; the recent inputs still dominate every update and the distant one is still drowned out. There is also a sign subtlety: those `n^{q-1}` path products can have different signs, so just adding more units does not reliably increase the surviving flow — the terms can cancel. So the difficulty is structural: gradient flow through a recurrence is exponentially attenuated in the lag, and none of the usual knobs touch the exponent.

OK. If the product of factors is the disease, then the cure has to make the product *not* shrink and *not* explode — the product over `q` steps should be exactly 1, for any `q`. Constant. Let me see what that demands, in the simplest possible setting: a single unit `j` with a single self-connection of weight `w_jj`. Its local backward flow over one step is `δ_j(t) = f_j'(net_j(t)) δ_j(t+1) w_jj`. For the error to be *unchanged* as it passes through `j`, I need the multiplier to be one:

  f_j'(net_j(t)) · w_jj = 1.0.

That is a constraint I can actually solve. Read it as a differential equation for `f_j`: `f_j'(net) = 1/w_jj`, a constant, so integrating, `f_j(net_j(t)) = net_j(t) / w_jj`. The squashing function must be *linear* — which is a striking conclusion, because the whole field's instinct is that squashing nonlinearities are what give a net its power, and here the analysis is telling me the memory channel must have *no* squashing at all. If I take the cleanest choice, the identity `f_j(x) = x` with `w_jj = 1.0`, then the unit's forward dynamics become `y_j(t+1) = f_j(net_j(t+1)) = f_j(w_jj y_j(t)) = y_j(t)` (when only the self-connection feeds it) — the activation just *persists*, unchanged, step after step, and the backpropagated error riding through it is multiplied by exactly 1 each step. A linear unit with a fixed self-loop of weight one: the error neither vanishes nor explodes no matter how long it has to travel, because `1^q = 1` for every `q`. Error goes round and round the self-loop at unit gain — I'll call it a constant error carousel. So I have a place where gradient can survive an arbitrarily long lag; the question is whether I can connect it to the rest of the net without destroying that property.

But of course unit `j` cannot be an island connected only to itself; it has to receive inputs from the rest of the net and send its content onward. The moment I wire it up, two conflicts appear, and they are not minor — they are exactly why a plain linear self-loop is not enough. Take a single incoming weight `w_ji` carrying some input `i`. That one weight has to do two contradictory jobs across the sequence. At the moment the relevant information arrives, `w_ji` must let it *in* — switch `j` on, write to the memory. But at all the *other* moments, when irrelevant inputs are coming through the same connection `i`, `w_ji` must *not* let them in — it must protect what is already stored from being overwritten. As long as `i` is nonzero, the same weight receives conflicting update signals: some pushing it to participate in storing, others pushing it to protect. The learning signal is at war with itself. Call it the input weight conflict. And there is a mirror image on the output side. A single outgoing weight `w_kj` must, at the moment the stored content is needed, let it *out* to the downstream unit `k` — read the memory. But at all the other moments, when `j`'s content is irrelevant to `k`, the same `w_kj` must protect `k` from being perturbed by `j`. Again one weight, two opposed jobs, conflicting updates. The output weight conflict. Both conflicts get *worse* precisely in the long-lag regime I care about: the longer information must be held, the longer it must be protected from overwriting, and — especially late in training — the more already-correct outputs there are that must be protected from disturbance.

So what I really need is not a static weight but a *context-sensitive* control: something that, depending on the current situation, decides whether to write to `j` and whether to read from `j`. A weight cannot be context-sensitive — it is one number. But another *unit* can be: a unit whose activation depends on the whole rest of the network at this step can output "open" in one context and "closed" in another. And I need that control to be able to *block* completely, not merely bias — because protecting the memory means the irrelevant input contributes *nothing*, exactly zero, not "a bit less." A multiplicative interaction does exactly that. If I gate the input to `j` by multiplying it by a control signal in `[0,1]`, then a control of 0 zeroes the input entirely (perfect protection) and a control of 1 lets it through (write enabled); anything between is a soft valve. An additive control could never zero a signal — it can only shift it. So the control must be *multiplicative*. That is the move: wrap the carousel in multiplicative gates that are themselves learned sigmoid units reading the rest of the net.

Concretely, give the cell two gate units. An input gate `in_j` with activation `y^{in_j}(t) = f_{in_j}(net_{in_j}(t))`, where `net_{in_j}(t) = Σ_u w_{in_j u} y^u(t-1)` collects from the rest of the net, and `f_{in_j}` is a logistic sigmoid so the gate sits in `[0,1]`. Its job is to protect the carousel from irrelevant inputs — it controls writing, resolving the input weight conflict. And an output gate `out_j` with `y^{out_j}(t) = f_{out_j}(net_{out_j}(t))`, sigmoid, whose job is to protect the rest of the net from the cell's irrelevant contents — it controls reading, resolving the output weight conflict. The two conflicts are genuinely distinct (one is about what gets in, the other about what gets out), so they need two separate gates.

Now the cell itself. The carousel is the linear self-loop with weight one. The new input to the cell is squashed through some function `g` and *multiplied by the input gate* before being added in; the internal state just accumulates:

  s_{c_j}(0) = 0,  s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) · g(net_{c_j}(t)),

where `net_{c_j}(t) = Σ_u w_{c_j u} y^u(t-1)` is the cell's net input. Notice the `s(t-1)` term sits there with an implicit weight of exactly 1 — that *is* the CEC, untouched, so error riding the state still gets multiplied by 1 per step. The input gate `y^{in_j}` decides how much of the new squashed input `g(net_{c_j})` is allowed to perturb the stored state; when it is near zero the state is held frozen, protected. Then the cell's *output* is the state, scaled through a squashing `h` and *multiplied by the output gate*:

  y^{c_j}(t) = y^{out_j}(t) · h(s_{c_j}(t)).

When the output gate is near zero, the cell reveals nothing to the rest of the net; when near one, it exposes `h(s)`. Why squash the state through `h` at all rather than emit `s` directly? Because `s` is a linear accumulator and can grow large, and I want the cell's output to live on the same bounded scale as the ordinary units it feeds, so that downstream weights see a comparable signal — `h` with range `[-1,1]` does that. And `g` squashes the candidate input similarly so a single huge input cannot slam the state; in the experiments `g` has range `[-2,2]` and `h` range `[-1,1]`, with the gates the logistic in `[0,1]` — `g(x)=4σ(x)-2`, `h(x)=2σ(x)-1`, `f(x)=1/(1+e^{-x})`. The asymmetry (wider `g`, narrower `h`) is just so the cell can be driven over a useful range while its exposed output stays tightly bounded.

Step back and look at what I have built: a linear memory state with a unit self-loop (gradient highway), an input gate that learns *when to write*, and an output gate that learns *when to read*. The gates are sigmoid units conditioned on the whole network, so the read/write decisions are context-sensitive — exactly the thing a bare weight could not be. This is a memory cell.

But I have to be careful, because I have now added connections, and connections are how error leaks. The whole point was a self-loop with multiplier 1. If I let standard backprop run unrestricted, error that flows *out* of the cell through the output gate and back into the network could come *back into* the cell through the input gate at an earlier step, around a loop, and then the factor along that loop is some product of `f'·w` again — the very thing that vanishes or explodes. The carousel only stays a carousel if error cannot sneak around it. So I am going to *truncate* the backprop: when an error signal arrives at the cell's net input, or at either gate's net input, I do *not* propagate it further back in time through the connections that brought it. Formally, I set the relevant cross-time derivatives to zero: `∂net_{in_j}(t)/∂y^u(t-1) ≈ 0`, `∂net_{out_j}(t)/∂y^u(t-1) ≈ 0`, `∂net_{c_j}(t)/∂y^u(t-1) ≈ 0`. The *only* error path I keep alive across multiple time steps is the one inside the cell, riding the state `s` along the CEC. Within the cell, error is propagated back through previous internal states — that is the long-range channel — but once it would leave through a gate or the cell input, it is cut off after changing the incoming weights at that step.

Does truncation actually preserve constant flow? Let me check the error scaling on the state directly rather than trust the slogan. The state recursion is `s(t) = s(t-1) + y^{in}(t) g(net_c(t))`. With truncation killing the back-in-time terms through the gates and cell input, the only surviving piece of `∂s(t)/∂s(t-1)` is the explicit `s(t-1)` term, whose derivative is 1. So `∂δ_{s_c}(t)/∂δ_{s_c}(t+1) = ∂s_c(t+1)/∂s_c(t) ≈ 1`. The error on the internal state should be multiplied by *exactly one* per step, for any number of steps.

I don't fully trust a one-line derivative argument for a claim the whole design rests on, so let me actually run a small example end to end and read off the numbers. Take one cell over a window of five steps. Say the relevant information arrives at the first step: input gate open and candidate `+1` at `t=0` (`i_0=1, g_0=1`), then the input gate shut for the rest (`i=0`), and — using the original ungated carousel — the state carried forward at weight 1. The output gate stays shut until the last step, where I open it to read (`o_4=1`, all earlier `o=0`). Forward, the state is `s = 1` written at `t=0` and then held: `s = [1, 1, 1, 1, 1]`. The hidden outputs are `h = [0,0,0,0, tanh(1)] = [0,0,0,0, 0.7616]` — nothing exposed until the read. Now suppose the loss only touches the final hidden, `∂L/∂h_4 = 1`, and run the backward state recursion `ε_s(t) = o_t tanh'(s_t) ε_h(t) + ε_s(t+1)` (the `+ε_s(t+1)` is the unit-gain carry). At `t=4` the entry term is `o_4 tanh'(1)·1 = 1·0.41997·1 = 0.41997` and there is nothing after it, so `ε_s(4) = 0.41997`. At `t=3,2,1,0` the entry term is zero (output gate shut) and the carry passes the value straight back, so `ε_s = [0.41997, 0.41997, 0.41997, 0.41997, 0.41997]` — identical at every step. The ratio `ε_s(0)/ε_s(4) = 1.0` exactly: the error from the write five steps ago arrives at the same magnitude as it had at the read. For contrast, a plain recurrent unit with the sigmoid's best-case factor `0.25` would have multiplied it by `0.25^4 ≈ 0.0039` over those four intervening steps — already down by two and a half orders of magnitude over a lag of *four*. So the carousel does what I demanded: the error gets scaled only twice — once when it enters the cell (by the output gate and `h'`, here `0.41997`) and once when it leaves to change the input weights (by the input gate and `g'`) — and in between, across the lag, it is scaled by 1. The surviving long-range flow does not decay with `q`. A protected linear channel that backprop is forbidden to route error around.

And truncation buys me efficiency on top of correctness. Because I never carry error back in time through the gates and cell inputs, I do not need to store the whole unrolled history of activations; the only derivatives that have to persist from one step to the next are the cell's and input gate's `∂s(t-1)/∂w` — the long-lived ones that ride the CEC. The update cost per weight per time step is `O(1)`, so the whole net is `O(W)` per step, same as BPTT for a fully recurrent net and far cheaper than RTRL's `O(W^2)`; and it is local in both space and time, so it runs online on arbitrarily long sequences. The thing I wanted from the start — bridge >1000 steps at `O(1)` per weight per step — falls out of the carousel-plus-truncation design.

Let me actually write the gradient so I am not gesturing at it. The forward pass for one memory cell block, with the squashing functions fixed as above: hidden unit `i` has `net_i(t)=Σ_u w_{iu} y^u(t-1)`, `y^i=f_i(net_i)`; the input gate `net_{in_j}=Σ_u w_{in_j u} y^u(t-1)`, `y^{in_j}=f_{in_j}(net_{in_j})`; the output gate likewise `net_{out_j}=Σ_u w_{out_j u} y^u(t-1)`, `y^{out_j}=f_{out_j}(net_{out_j})`; the cell `net_{c_j}=Σ_u w_{c_j u} y^u(t-1)`, then `s_{c_j}(t)=s_{c_j}(t-1)+y^{in_j}(t)g(net_{c_j}(t))`, `y^{c_j}(t)=y^{out_j}(t)h(s_{c_j}(t))`; and the output `net_k=Σ_u w_{ku} y^u(t-1)`, `y^k=f_k(net_k)`. I take the squared error at time `t` as `E(t)=1/2 Σ_k (t^k(t)-y^k(t))^2`, so the harmless factor of two is not floating around, and the weight update at learning rate `α` is `Δw_lm(t) = -α ∂E(t)/∂w_lm`. Define each unit's error `e_l(t) := -∂E(t)/∂net_l(t)`. The conventional units come out by standard backprop: for an output unit `e_k(t) = f_k'(net_k(t))(t^k(t)-y^k(t))`; for a hidden unit `e_i(t) = f_i'(net_i(t)) Σ_k w_ki e_k(t)`; and for the output gate, since it scales `Σ_v h(s_{c_j^v})` worth of cell outputs that feed downstream, `e_{out_j}(t) = f_{out_j}'(net_{out_j}(t)) (Σ_{v=1}^{S_j} h(s_{c_j^v}(t)) Σ_k w_{k c_j^v} e_k(t))`, summing over the `S_j` cells in the block. For all of these, `Δw_lm(t)=α e_l(t) y^m(t-1)`.

The interesting ones are the input gate and the cell, because they ride the state. Define the internal state's error `e_{s_{c_j^v}} := -∂E(t)/∂s_{c_j^v}(t)`. Tracing the path from `s` out through `h`, the output gate, and the downstream weights, `e_{s_{c_j^v}}(t) = f_{out_j}(net_{out_j}(t)) h'(s_{c_j^v}(t)) Σ_k w_{k c_j^v} e_k(t)`. Now the weights into the input gate and into the cell get their update through the *derivative of the state with respect to those weights*, accumulated along the CEC. Because `s(t)=s(t-1)+y^{in}g(net_c)`, differentiating with respect to an input-gate weight `w_{in_j m}` gives a recursion `∂s(t)/∂w_{in_j m} = ∂s(t-1)/∂w_{in_j m} + g(net_{c_j}(t)) f_{in_j}'(net_{in_j}(t)) y^m(t-1)` — the old sensitivity is carried forward by the unit self-loop (there is the CEC again, now in the *forward sensitivity*), plus a new contribution from this step. And differentiating with respect to a cell weight `w_{c_j m}` gives `∂s(t)/∂w_{c_j m} = ∂s(t-1)/∂w_{c_j m} + g'(net_{c_j}(t)) f_{in_j}(net_{in_j}(t)) y^m(t-1)`. So the updates are `Δw_{in_j m}(t) = α Σ_v e_{s_{c_j^v}}(t) ∂s_{c_j^v}(t)/∂w_{in_j m}` and `Δw_{c_j^v m}(t) = α e_{s_{c_j^v}}(t) ∂s_{c_j^v}(t)/∂w_{c_j^v m}`. The only things I keep between steps are exactly these `∂s/∂w` sensitivities for the cell and input gate — the long-lived quantities that the carousel preserves.

The five-step trace is one instance; let me write the general factor symbolically so I know the `1.0` ratio wasn't an artifact of the particular numbers I picked. Suppose an error `δ_j(t)` arrives at the cell output at time `t` and I follow it `q` steps back until it reaches the cell input `g(net_c)` or the input gate. Under truncation, the per-step state-to-state factor is `∂s_{c_j}(t-k)/∂s_{c_j}(t-k-1) ≈ 1` for every intermediate `k` — that is eq. (35) above, the carousel. So the only scalings are at the two ends. Expanding the chain, the total factor from cell output at `t` back to the cell input or input gate at `t-q` is `y^{out_j}(t) h'(s_{c_j}(t))` (entering) times a product of `q` ones (the CEC) times, at the far end, `g'(net_{c_j}(t-q)) y^{in_j}(t-q)` if it lands on the cell input, or `g(net_{c_j}(t-q)) f_{in_j}'(net_{in_j}(t-q))` if it lands on the input gate. The two end factors are exactly the `0.41997` entry term and the (unit) exit term I read off the example; the `q` ones in the middle are why the ratio came out 1 regardless of `q`. No factor of the form (something)^q. The flow may be scaled *once* on entry and *once* on exit, but it does not decay with the length of the lag in between. That is qualitatively different from the bare recurrent net, where the `q` ones would have been `q` copies of some number below one. The output gate's entry factor `y^{out_j}(t)` is itself meaningful: it lets the cell *scale down* errors it should not trap (errors that ought to be fixed cheaply by short-term units), and the input gate's exit factor lets it scale which errors actually change the stored content — the gates learn which errors to trap in the CEC and which to release.

Now, having a working memory cell, I should ask what one cell still cannot do, because that is where the next piece comes from. The internal state `s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) g(net_{c_j}(t))` can only ever *add* to itself; with the self-loop pinned at weight 1 it never decays. On a task that is neatly segmented — a sequence, reset, the next sequence — that is fine, I just start each sequence from `s=0`. But on a continual stream that is never reset, an input gate that occasionally opens keeps piling contributions onto `s`, and `s` grows without bound. Let me check what that actually does to the cell, because "saturates" is exactly the kind of word I just caught myself hand-waving with. The exposed output is `h(s) = tanh(s)` and what carries gradient back into the state is `h'(s) = tanh'(s) = 1 - tanh^2(s)`. At `s = 1`, `tanh'(1) = 0.420` — healthy. At `s = 3`, `tanh'(3) = 9.9e-3`. At `s = 5`, `1.8e-4`. At `s = 10`, `8.2e-9`. So once a few writes have driven `s` past about 5, the output is pinned near `±1` and the derivative that lets error enter the cell has collapsed by four to eight orders of magnitude — the carousel is intact internally, but it has become unreadable and unwritable through a derivative that is effectively zero. A dead, pegged accumulator. The cell has no way to *release* what it is holding when the information becomes stale. What I want is for the cell to learn *when to forget* — to reset itself at the right moments — without my having to tell it where the sequence boundaries are.

The fixed self-weight of 1 is exactly what stands in the way, so make it adaptive — but adaptive in the same context-sensitive, multiplicative way the other two gates already are. Introduce a third gate, a forget gate `y^{φ_j}(t) = f_{φ_j}(net_{φ_j}(t))` in `[0,1]`, and let it multiply the carried-over state instead of the fixed 1:

  s_{c_j}(t) = y^{φ_j}(t) · s_{c_j}(t-1) + y^{in_j}(t) · g(net_{c_j}(t)).

When `y^{φ_j} = 1` this is the original CEC, exactly — the constant carousel is recovered as a special case, so I have not lost the long-lag highway. When `y^{φ_j} = 0` the cell wipes its state and starts fresh. In between it can let the memory decay at a learned rate. Let me sanity-check that "decay at a learned rate" is really the behavior: if the forget gate sits at a constant `y^φ` across a lag, the state error picks up a factor `y^φ` per step, so over `q` steps it is `(y^φ)^q`. At `y^φ = 1`, `1^{100} = 1` — undamped, as required. At `y^φ = 0.99`, `0.99^{100} ≈ 0.37` — a gentle, deliberate forgetting over a hundred steps. At `y^φ = 0.9`, `0.9^{50} ≈ 5.2e-3` — fast forgetting. So the single gate value `y^φ` is precisely a knob from "remember forever" to "forget immediately," and it slides smoothly between them. Crucially the forget gate is learned from context, so the cell discovers its *own* reset points and its own decay rate instead of relying on an externally chosen sequence boundary or a fixed time constant.

That gives me the modern memory cell: three gates — input (when to write), forget (when to reset), output (when to read) — wrapped around a linear state that, when the forget gate is open, carries gradient at unit gain across arbitrarily long lags. Let me write it in the compact vectorized form I would actually compute, for a whole layer of cells at once, with `x_t` the input at step `t`, `h_{t-1}` the previous cell outputs fed back, `σ` the logistic sigmoid for the gates, and `tanh` for the input squashing `g` and the output squashing `h` (the standard bounded, zero-centered choice):

  i_t = σ(W_ix x_t + W_ih h_{t-1} + b_i)        # input gate: when to write
  f_t = σ(W_fx x_t + W_fh h_{t-1} + b_f)        # forget gate: when to reset
  g_t = tanh(W_gx x_t + W_gh h_{t-1} + b_g)      # candidate cell input
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t               # the CEC, now gated by forget/input
  o_t = σ(W_ox x_t + W_oh h_{t-1} + b_o)        # output gate: when to read
  h_t = o_t ⊙ tanh(c_t)                          # exposed cell output

Each gate is a sigmoid unit reading both the new input and the recurrent feedback; the cell state `c_t` is the protected linear memory; `h_t` is what the rest of the net sees. To make a deeper hidden representation I can stack these — feed the hidden sequence `h_t` of one layer as the input sequence to another LSTM layer — with the same per-step update at each layer.

Let me also write the backward pass for this vectorized cell, because the carousel claim must hold in this form too. Let `ε_h^t := ∂L/∂h_t` be the error arriving at the exposed hidden output, including the read-out at time `t` and the next step's use of `h_t` in its gate and candidate affine maps. Let `ε_s^t := ∂L/∂c_t` be the error on the state. The output gate, which feeds the loss through `h_t = o_t ⊙ tanh(c_t)`, gets `δ_o^t = σ'(a_o^t) ⊙ tanh(c_t) ⊙ ε_h^t`. The state error collects the part coming back through `h_t` (via `o_t ⊙ tanh'(c_t)`) plus — and this is the carousel — the part carried back from the next step's state, which because `c_{t+1} = f_{t+1} ⊙ c_t + …` arrives multiplied by `f_{t+1}`:

  ε_s^t = o_t ⊙ tanh'(c_t) ⊙ ε_h^t + f_{t+1} ⊙ ε_s^{t+1}.

There it is in gradient form: the state error at `t` inherits the state error at `t+1` scaled by the forget gate. When `f_{t+1} ≈ 1`, that scaling is unit gain — the error flows back across the lag undamped, the constant error carousel as a statement about gradients. When the forget gate is closed, the cell deliberately drops the gradient too, because it has chosen to forget. The remaining updates fall out by differentiating each multiplication in the forward pass: the candidate gets `δ_g^t = i_t ⊙ tanh'(a_g^t) ⊙ ε_s^t`; the forget gate gets `δ_f^t = σ'(a_f^t) ⊙ c_{t-1} ⊙ ε_s^t`; the input gate gets `δ_i^t = σ'(a_i^t) ⊙ g_t ⊙ ε_s^t`. The recurrent contribution sent to `h_{t-1}` is the sum of the four recurrent weight-transpose products, `W_ih^T δ_i^t + W_fh^T δ_f^t + W_gh^T δ_g^t + W_oh^T δ_o^t`, plus any other path already collected in `ε_h^{t-1}`. Every gate's error is its own derivative times the quantity it multiplies in the forward pass times the state error — clean, and all of it has the right sign because the `δ` terms are gradients of `L` with respect to preactivations.

So I have the architecture, the learning rule, and the proof that gradient survives the lag. Now I want to put it to work on a concrete regression: predict a forward return for each instrument from a window of recent observations — six base price/volume ratios over sixty trading days. That is a sequence-to-one problem: a sixty-step, six-feature window in, one score out. The LSTM is exactly the right shape for it — run the cell over the sixty steps, and the protected memory lets a feature from sixty days ago still influence the prediction without its gradient having vanished. The flat feature vector arrives as `[N, 360]`; I reshape it back into the time axis the cell needs, `[N, 6, 60]` then transpose to `[N, 60, 6]`, run the LSTM, and read out the *last* step's hidden state — the hidden after sixty steps is the cell's summary of the whole window — through a linear layer to a single number. I'll train it by masked mean-squared error over the finite targets, with Adam at `1e-3`; I'll clip gradients by value at 3.0, because even with the carousel the *forward* analysis warned me about the exploding side — a bad batch can still spike a gradient — and value-clipping caps parameter updates without changing the cell equations; and I'll early-stop on a validation score, restoring the best parameters, so I take the iterate that generalizes rather than the last noisy one.

```python
import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


class LSTMModel(nn.Module):
    """A stack of LSTM memory-cell layers + a linear read-out.

    Each LSTM layer runs the gated memory cell over the time axis. The forget/input/output
    gates control reset/write/read; the cell state c_t is the constant-error carousel that
    carries gradient across the 60-step window at unit gain when the forget gate is open.
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        # nn.LSTM is the gated memory cell derived above: sigmoid gates, tanh g & h,
        # c_t = f_t*c_{t-1} + i_t*g_t (the CEC), h_t = o_t*tanh(c_t). Stacked num_layers deep.
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)   # read out the final hidden -> one score
        self.d_feat = d_feat

    def forward(self, x):
        # x arrives flat as [N, 6*60]; recover the time axis the recurrence needs
        x = x.reshape(len(x), self.d_feat, -1)    # [N, 6, 60]
        x = x.permute(0, 2, 1)                     # [N, 60, 6]  (time, then features)
        out, _ = self.rnn(x)                       # out: [N, 60, hidden]
        # last step's hidden = the cell's summary of the whole 60-day window
        return self.fc_out(out[:, -1, :]).squeeze()


class LSTM(Model):
    """Sequence-to-one LSTM regressor over a fixed feature window."""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=800,
        early_stop=20,
        loss="mse",
        GPU=0,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop          # patience on validation score
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

        self.lstm_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        return torch.mean((pred - label) ** 2)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)                 # ignore missing targets
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_values = x_train.values
        y_values = np.squeeze(y_train.values)
        self.lstm_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            # cap the exploding side of the error-flow analysis
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.lstm_model.eval()
        scores, losses = [], []
        indices = np.arange(len(x_values))
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.lstm_model(feature)
            losses.append(self.loss_fn(pred, label).item())
            scores.append(self.metric_fn(pred, label).item())
        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps, best_score, best_epoch, best_param = 0, -np.inf, 0, None
        self.fitted = True
        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))
            if val_score > best_score:                 # keep the best-validating iterate
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break
        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []
        for begin in range(sample_num)[:: self.batch_size]:
            end = sample_num if sample_num - begin < self.batch_size else begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=index)
```
