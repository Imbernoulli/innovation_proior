# Context: delayed generalization on algorithmic datasets (circa 2022-2024)

## Research question

On small algorithmic datasets — for example predicting `c = a · b (mod p)` from the pair
`(a, b)` — a modest neural network reaches near-perfect training accuracy very quickly, within
on the order of `10^3` optimizer steps, yet its validation accuracy stays near chance for a very
long time and only climbs to comparable levels after on the order of `10^5`-`10^6` steps. The
generalization is *delayed* by one to three orders of magnitude past the point of overfitting.
For a practitioner this delay is the whole problem: the network clearly *can* generalize on this
task, but waiting tens of thousands of extra steps to see it is prohibitively expensive and makes
the behavior almost useless as a practical training regime.

The precise goal is to **shrink that delay** — make the late jump in validation accuracy happen
far sooner in wall-clock and step count — under tight constraints. The dataset, the loss
(cross-entropy), the train/test split, and the step budget are fixed; the architecture is a
standard small network. The intervention must therefore live in the *optimization process*: it
should be cheap (a handful of lines), add little memory, attach to whatever first-order optimizer
is already in use (SGD, Adam, AdamW) rather than replace it, and not depend on the train/test
split. A solution that needs a bespoke optimizer, a second forward pass, or memory that scales
with the dataset would defeat the point.

## Background

By this time the delayed-generalization behavior on algorithmic tasks is an established,
reproducible phenomenon. Power et al. (2022, "Grokking: Generalization Beyond Overfitting on
Small Algorithmic Datasets", arXiv:2201.02177) reported it on a two-layer Transformer trained on
binary modular-arithmetic tables: the train curve saturates early, the validation curve lags by
a large factor, and crucially the *weight decay* of an AdamW optimizer dramatically reduces the
lag — turning weight decay from a generic regularizer into a control knob over *when* the network
generalizes. Liu et al. ("Omnigrok", arXiv:2210.01117) then showed the same train-fast /
generalize-late signature appears far beyond algorithmic data — on images, language, and graphs —
once the model is put in the right regime, and analyzed it through the loss landscape and the
norm of the weights, again finding weight decay to be a critical determinant. Nanda et al.
(arXiv:2301.05217) reverse-engineered a one-layer Transformer on a modular-addition table and
found that the network is not idle during the long plateau: an internal, slowly-rising progress
measure tracks the gradual formation of a structured (Fourier-style) algorithm, so the eventual
jump in validation accuracy is the visible end of a *continuous* internal process rather than a
sudden event.

So the field's picture, established before any fix exists, is: (i) overfitting and generalization
proceed on **two very different timescales** — the training loss collapses fast, the validation
loss moves slowly; (ii) something is changing *gradually* in the weights throughout the long
plateau, even while the validation metric looks flat; and (iii) regularization choices,
especially weight decay, strongly modulate the length of the plateau.

The load-bearing tools available for acting on this picture come from two directions. From
optimization: first-order methods all build their update out of the gradient stream, and the
standard accelerators keep a *running average* of past gradients — classical heavy-ball momentum
`m(t) = μ m(t-1) + (1-τ) g(t)`, its Nesterov look-ahead variant, and the exponential moving
averages inside RMSProp/Adam. Each is a recursion that smooths the raw gradient over time, with a
decay factor `β` whose effective memory is roughly `1/(1-β)` steps. From signal processing: a
sequence indexed by time can be analyzed in the frequency domain via the discrete-time Fourier
transform `F{f}(ω) = Σ_t f(t) e^{-iωt}`; a *linear time-invariant* system acts on a signal by
convolution `h * x` in time, which becomes pointwise multiplication `H(ω) X(ω)` in frequency,
where `H(ω) = F{h}` is the system's transfer function; a **low-pass filter** is one whose `H(ω)`
keeps low frequencies and attenuates high ones, and a windowed moving average and a single-pole
exponential moving average are the two textbook low-pass filters (one finite-impulse-response,
one infinite-impulse-response). These two vocabularies — "running average of gradients" and
"low-pass filter of a time signal" — describe the same kind of object from different fields.

## Baselines

These are the existing ways of attacking the delay, and the optimizer machinery a new method
would attach to or react against.

**AdamW weight decay as the grokking knob (Power et al. 2022; Liu et al. 2022).** The dominant
recipe for *inducing and shortening* the delay is decoupled weight decay: train the small
Transformer with `AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay≈1)`, and larger weight decay
generally makes the late generalization arrive sooner. Core idea: shrink the weight norm so the
optimizer is pushed toward the lower-norm, generalizing solution. *Gap:* it is a global
regularization pressure, not a targeted acceleration — it changes which solution is favored but
still leaves a long plateau, the right amount is task-dependent and brittle (too much makes
training unstable), and it gives no separate handle on the slow-versus-fast dynamics themselves.

**Heavy-ball / Nesterov momentum (Polyak; Sutskever et al. 2013).** Keep an exponentially decaying
running average of the gradient and step along it: `m(t) = μ m(t-1) + (1-τ) g(t)`, with update
`u(t) = -η m(t)` (heavy-ball) or `u(t) = -η(g(t) + μ m(t))` (Nesterov look-ahead). The average
suppresses per-step noise and builds speed along persistent directions. *Gap:* momentum is built
to *speed convergence of the training loss* and is consumed as the update itself with a single
fixed decay; it is not aimed at the validation-side slow dynamics, and there is no separate
control to *emphasize* a slow component relative to the fast one — the smoothed signal *replaces*
the step rather than being dialed up against the raw step.

**Adaptive EMAs inside RMSProp / Adam / AdamW.** These maintain exponential moving averages of the
gradient and of the squared gradient to set per-coordinate step sizes, again with a fixed decay.
*Gap:* the moving average is an internal mechanism of one specific optimizer used to *precondition*
the step; it is not exposed as a tunable, optimizer-independent operation on the gradient stream,
and it is not used to selectively amplify the slowly-varying part of the gradients.

**Staged / regime-aware training and other regularizers (Nanda et al. 2023; mechanistic studies).**
Because the dynamics look different before versus after overfitting, one line treats training as
passing through distinct regimes and considers applying different pressure in each, and other work
notes that L2-norm and dropout (but, reportedly, not L1) can induce the phenomenon. *Gap:* these
identify *that* regime structure and regularization matter and characterize the slow internal
progress, but stop at description — they do not turn "a slow component is quietly accumulating
through the plateau" into a concrete, cheap, optimizer-agnostic operation that acts on it.

## Evaluation settings

The natural yardsticks already in use, all predating any fix:

- **Algorithmic modular arithmetic** — the original grokking setting: a two-layer decoder-only
  Transformer trained to predict `a · b (mod 97)` (and modular addition `(a + b) mod p` for
  small primes such as `p = 59, 97, 113`), full-batch or large-batch gradient descent on a fixed
  fraction of the `p × p` table, cross-entropy loss, AdamW with the Power et al. hyperparameters.
  The reported quantity is the *delay*: the ratio (or gap) between the step at which training
  accuracy saturates and the step at which validation accuracy reaches a chosen threshold (e.g.
  95% of train accuracy, or a fixed `0.99`).
- **MNIST under the grokking regime** — a small ReLU-MLP placed in the regime where delayed
  generalization appears; train/validation accuracy versus steps.
- **Molecular property regression on QM9** — a graph convolutional network; since there is no
  accuracy, validation-loss convergence speed is the yardstick.
- **IMDb sentiment** — a small LSTM exhibiting the same delayed-generalization signature; best
  validation loss/accuracy and convergence speed.
- Protocol: identical initialization and fixed data split across runs; any optimizer-side
  hyperparameters chosen by a small grid search over a gain factor and a memory/decay parameter;
  comparisons read off the time/steps to reach a fixed validation threshold.

## Code framework

The intervention plugs into the ordinary first-order training loop already used for every
baseline above. Backpropagation has filled in `p.grad` for each parameter, and the optimizer
turns those gradients into a parameter update. The one place where something *new* may act is the
window between the gradient becoming available and the optimizer consuming it: a hook that runs
after `loss.backward()` and before `optimizer.step()`, free to read and overwrite each `p.grad`.
What that hook should do is exactly the open question — so the scaffold leaves it as a single
empty slot, with whatever per-parameter state it might need also unspecified.

```python
import torch


def build_optimizer(model, config):
    """The existing first-order optimizer the baselines already use
    (e.g. AdamW with the standard grokking hyperparameters). Unchanged."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


class GradHook:
    """Runs once per step, AFTER loss.backward() (so every p.grad is set)
    and BEFORE optimizer.step(). It may read and overwrite each p.grad,
    and may keep any per-parameter state across steps."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        # TODO: any per-parameter state the operation we will design needs.
        self.state = {}

    @torch.no_grad()
    def post_grad(self, step):
        for name, p in self.model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad                    # this step's raw gradient for this parameter
            # TODO: the per-parameter operation on the gradient stream we will design.
            #       Given the sequence of gradients g seen at this parameter over
            #       steps (and any state we choose to keep), modify p.grad in place
            #       before the optimizer reads it.
            pass


# existing full-/large-batch training loop the hook plugs into
def train(model, loss_fn, data, optimizer, hook):
    for step, (inputs, targets) in enumerate(data):
        optimizer.zero_grad()
        logits = model(inputs)            # forward through the existing model
        loss = loss_fn(logits, targets)   # existing cross-entropy loss
        loss.backward()                   # backprop fills p.grad for every parameter
        hook.post_grad(step)              # the one new slot: may modify p.grad
        optimizer.step()                  # existing optimizer consumes (possibly modified) grads
```

The training loop, the model, the loss, and the optimizer are all fixed; the only degree of
freedom is what `post_grad` does to the gradients before the optimizer sees them.
