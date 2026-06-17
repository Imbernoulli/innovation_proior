# Context: learning long-range temporal dependencies with recurrent networks (early/mid 1990s)

## Research question

A recurrent network can, in principle, do something a feedforward network cannot: feed its own
activations back to itself, so the activation pattern at one time step carries information about
inputs seen many steps earlier. This is "short-term memory" stored in activations, as opposed to
the "long-term memory" stored in slowly-changing weights. It is exactly what tasks with temporal
structure need — speech, music, control, any setting where the right output at time `t` depends
on an input seen at some earlier time `t - q`.

The trouble is *learning* what to store. The gradient-based learning rules for recurrent nets
(Back-Propagation Through Time, Real-Time Recurrent Learning) do work when the lag `q` between
the informative input and the moment its information is needed is small — a handful of steps. But
as `q` grows, training either fails outright or takes a prohibitively long time. Empirically,
existing recurrent learning algorithms do not reliably bridge minimal time lags beyond roughly
ten steps. The precise goal is a recurrent architecture *and* a matching learning rule that can
learn to bridge minimal time lags well in excess of 1000 steps, even when the inputs are noisy
and incompressible (so there is no shortcut through local predictability); that does not lose the
ability to handle short lags at the same time; and whose per-weight, per-time-step update cost
stays `O(1)` — cheap and ideally local in both space and time, so it is usable online and on long
sequences. Closing the gap between "recurrent nets can in principle remember" and "we can
actually train them to remember over long lags" is the problem.

## Background

The dominant way to train a recurrent net on time-varying inputs is to compute the exact gradient
of a per-time-step squared error through the unrolled recurrence. Two standard algorithms do
this. Back-Propagation Through Time (BPTT; Werbos 1988; Williams and Zipser 1992) unrolls the
network over the sequence and backpropagates error from each output back through time. Real-Time
Recurrent Learning (RTRL; Robinson and Fallside 1987) instead carries forward the sensitivities
of every unit's activation to every weight, so it can update online but at `O(W^2)` cost per step
for `W` weights. Many related gradient methods (Elman 1988; Pearlmutter 1989; Schmidhuber 1992a)
share the same gradient and the same behavior on long lags.

The load-bearing diagnostic finding — knowable purely from the BPTT/RTRL recurrence, and the
thing that frames everything — is *how the backpropagated error scales as it flows back in time*.
Take the standard recurrent backprop signals: an output unit `k`'s error is
`δ_k(t) = f_k'(net_k(t)) (d_k(t) - y_k(t))`, and a non-output unit `j`'s error is
`δ_j(t) = f_j'(net_j(t)) Σ_i w_ij δ_i(t+1)`, where `f_i` is unit `i`'s differentiable activation,
`net_i(t) = Σ_j w_ij y_j(t-1)`, and `w_ij` is the weight from `j` to `i`. Now follow the error
that occurs at a unit `u` at time `t` as it is propagated `q` steps back to a unit `v`. By
induction over `q`,

```
∂δ_v(t-q) / ∂δ_u(t) = Σ_{l_1=1}^n ... Σ_{l_{q-1}=1}^n  Π_{m=1}^q  f'_{l_m}(net_{l_m}(t-m)) · w_{l_m l_{m-1}}
```

with `l_0 = u`, `l_q = v`. Each of the `n^{q-1}` paths from `u` to `v` contributes a product of
`q` factors of the form `f'(net) · w`. The behavior of that product as `q` grows decides
everything:

- If `|f'_{l_m}(net_{l_m}(t-m)) · w_{l_m l_{m-1}}| > 1` for all `m` along a path (which can happen,
  e.g. with a linear unit), the largest product *grows exponentially* in `q`. The error blows up;
  arriving error signals oscillate and learning is unstable (Pineda 1988; Baldi and Pineda 1991;
  Doya 1992).
- If `|f'_{l_m}(net_{l_m}(t-m)) · w_{l_m l_{m-1}}| < 1` for all `m`, the largest product *decays
  exponentially* in `q`. The error vanishes; the contribution of a long-ago input to the present
  loss is exponentially small, so nothing can be learned over long lags in acceptable time.

For the logistic sigmoid, `max f' = 0.25`, and `|f'(net) w| < 1.0` whenever `|w| < 4.0`. So with
ordinary weight magnitudes the decaying case is the rule: error to long-past inputs vanishes. A
slightly extended bound that also tracks the number of units `n` is
`|∂δ_v(t-q)/∂δ_u(t)| ≤ n (f'_max ||W||_A)^q`; choosing the matrix/vector norms
`||W||_A := max_r Σ_s |w_rs|` and `||x|| := max_r |x_r|` gives `f'_max = 0.25` for the sigmoid,
and if `|w_ij| ≤ w_max < 4/n` then `||W||_A ≤ n w_max < 4`, so with `τ := n w_max / 4 < 1` the
factor is bounded by `n τ^q` — exponential decay in the lag. Two corollaries matter: increasing
the weights does *not* rescue long-range flow (a larger `w` drives the unit into saturation where
`f'` collapses even faster), and increasing the learning rate does *not* either (it scales the
long-range and short-range error identically, so the *ratio* that starves long-range credit
assignment is unchanged). The lag itself, through the exponent `q`, is the obstacle.

This diagnostic fact frames the problem: gradient flow over long recurrences is exponentially
attenuated (or exponentially amplified) by construction. Note also the sign structure: because
the `n^{q-1}` path products can have different signs, simply adding units does not necessarily
increase the total flow — the cancellation can make things worse, not better.

A handful of partial ideas were in the air. Time-Delay Neural Networks (Lang, Waibel and Hinton
1990) and Plate's holographic method (1993) handle a fixed short window of past activations via
explicit delays, but only for short lags. Mozer (1992) used adjustable *time constants* that slow
down a unit's activation change, which lengthens its effective memory — but the time constants
need external fine tuning, and Sun, Chen and Lee's (1993) related "add scaled old activation to
current input" update tends to let the net input perturb the stored information, making long-term
storage impractical. Schmidhuber's hierarchical sequence chunkers (1992b, 1993) can bridge
arbitrary lags, but only when the subsequences across the lag are *locally predictable*; their
performance degrades as the inputs get noisier and less compressible. Watrous and Kuhn (1992)
used multiplicative units in second-order nets for finite-state induction, but with fully
connected second-order sigma-pi units at `O(W^2)` cost per step and no mechanism enforcing
constant error flow. None of these closes the long-lag, noisy-input gap cheaply.

## Baselines

These are the prior methods a new approach would be measured against and would react to.

**BPTT / RTRL (Werbos 1988; Williams and Zipser 1992; Robinson and Fallside 1987).** The exact
recurrent gradient. BPTT unrolls the net over the whole sequence and backpropagates; RTRL carries
the forward sensitivities so it can update online. Both implement the recurrence and error signals
above. *Gap:* the backpropagated error over `q` steps is the product of `q` factors `f'·w`, which
is forced to decay (or explode) exponentially in `q` (see Background). On long lags BPTT's
credit assignment is dominated by recent inputs and the long-range signal is lost; RTRL pays a
further `O(W^2)` per-step cost. Larger weights and larger learning rates do not change the
long-vs-short ratio. So these methods learn short-lag structure and stall on long-lag structure.

**Time constants and time delays (Mozer 1992; Lang, Waibel and Hinton 1990; Plate 1993; Sun,
Chen and Lee 1993).** Stretch a unit's effective memory either by a slow first-order time
constant on its activation, or by feeding in explicit delayed copies of past activations. *Gap:*
delays only cover a fixed, short window; time constants must be tuned by hand and are appropriate
to a particular lag scale; and the additive "old activation + current input" form lets new input
keep perturbing what is stored, so the stored quantity drifts rather than being held.

**Hierarchical sequence chunkers (Schmidhuber 1992b, 1993).** Compress a sequence by predicting
it at multiple time scales, so a higher level sees a shorter, chunked sequence and only the
*unpredictable* events are passed up; this can bridge long lags. *Gap:* it relies on local
predictability across the lag — when the sequence is noisy and incompressible there is nothing to
chunk, and performance deteriorates as the noise level rises.

**Multiplicative-unit second-order nets (Watrous and Kuhn 1992).** Use multiplicative
(sigma-pi) units to gate signals in a recurrent net for finite-state language induction. *Gap:*
the architecture does not enforce constant error flow and is not designed for long lags; with
fully connected second-order units it costs `O(W^2)` per step.

**Non-gradient and discrete-error approaches (Bengio, Simard and Frasconi 1994; Bengio and
Frasconi 1994; Puskorius and Feldkamp 1994).** Bengio et al. analyze the same long-term-dependency
difficulty and try simulated annealing, multi-grid random search, time-weighted pseudo-Newton, and
discrete error propagation; their EM "state network" can bridge long lags but only represents `n`
discrete states. Kalman-filter-trained recurrent nets impose a derivative discount factor that
decays past dynamics exponentially. *Gap:* the discrete-state approaches need an unacceptable
number of states for continuous problems, and the discount-factor methods explicitly decay the
very long-range dependencies they would need to keep.

## Evaluation settings

The natural yardsticks for long-range temporal credit assignment:

- **Artificial long-time-lag tasks with controllable lag `q`.** Sequences where a relevant input
  appears at some step and the required output (a classification or a stored real value) is due
  many steps later, with the lag `q` swept up into the hundreds and beyond, to measure directly
  how far back a method can assign credit. Variants use local vs. distributed input encodings,
  real-valued vs. symbolic inputs, and added noise / distractor symbols so the task cannot be
  solved by short-window shortcuts.
- **The "adding problem" and similar continuous storage tasks**, where the network must carry and
  combine real-valued quantities across a long lag — a continuous problem that finite-state
  methods cannot represent compactly.
- **Noisy, incompressible input streams**, included specifically to defeat methods that rely on
  local predictability or compressibility of the sequence.
- Metrics: whether the task is solved at all (many prior methods simply cannot), the number of
  successful runs out of many random seeds, and training speed. Protocol: per-time-step squared
  error against a target signal, gradient updates, repeated over random initializations.

For a real-world regression deployment, the natural pipeline is a cross-sectional predictor over
a fixed feature panel: a per-instrument window of recent observations (here six base
price/volume ratios over sixty trading days), a scalar regression target (a forward return), a
fixed train/validation/test split, and a downstream ranking/portfolio backtest. The model is
asked to map each instrument's sixty-step window to one score, trained by mean-squared error.

## Code framework

The architecture plugs into the standard recurrent-net training harness that already exists: a
sequence module that consumes a `[batch, time, features]` tensor and emits a per-step hidden
representation, a linear read-out, an optimizer, a squared-error loss, and a minibatch loop with
early stopping on a validation score. What is *not* yet settled is the recurrent unit itself —
the thing inside the sequence module that decides, at each step, how the previous hidden state and
the new input combine into the next hidden state. That recurrent cell is exactly the slot to be
designed; the substrate below is generic.

```python
import torch
import torch.nn as nn


class RecurrentCell(nn.Module):
    """One step of a recurrent hidden layer. Given the previous recurrent state and the
    current input, produce the next state. The exact internal computation -- how the past
    state and the new input are combined -- is the thing to be designed."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # TODO: the parameters and per-step update rule we will design.

    def forward(self, x_t, state_prev):
        # x_t: [batch, input_size]; state_prev: the carried recurrent state.
        # TODO: combine x_t and state_prev into the next state and the step output.
        raise NotImplementedError


class SequenceRegressor(nn.Module):
    """Generic sequence-to-one regressor: run a recurrent cell over the time axis,
    read out the final hidden state with a linear layer to one score."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = RecurrentCell(input_size, hidden_size)
        self.read_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, features*time] flattened; recover the [batch, time, features] view
        x = x.reshape(len(x), self.cell.input_size, -1)   # [batch, features, time]
        x = x.permute(0, 2, 1)                            # [batch, time, features]
        state = None                                      # initialized inside the cell
        last_hidden = None
        for t in range(x.shape[1]):
            last_hidden, state = self.cell(x[:, t, :], state)
        return self.read_out(last_hidden).squeeze()       # one score per instrument


# existing minibatch training loop the cell plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)         # masked MSE over finite targets
        loss.backward()
        optimizer.step()
```

The loop, the read-out, the loss, and the optimizer are fixed. The single empty slot is the
recurrent cell's per-step update: how the previous state and the current input become the next
state across the sequence.
