## Research question

We are handed a function `F: R^d -> R` that we want to drive as low as possible, but we never
see `F` directly. All we have is a stochastic oracle: we can draw sample functions `f(., xi)`
where `xi` is a random sample (a training example, or a minibatch index) such that
`E[f(., xi)] = F(.)`. In supervised learning `x` is the model parameters, `xi` an example,
`f(x, xi)` the loss on that example, and `F` the population (or training) loss. We do not
assume `F` is convex, so finding a true global minimum is in general intractable; the relaxed
goal is to find a *critical point*, a point with `||nabla F(x)|| = 0`, using only stochastic
gradients (no Hessians, no second-order oracle).

The hard, quantitative version of the problem is: how few stochastic-gradient queries does it
take to reach a point with `E[||nabla F(x)||] <= epsilon`? Plain SGD reaches
`E[||nabla F(x_t)||] <= O(T^{-1/4})` after `T` steps, i.e. `O(epsilon^{-4})` queries. A family
of newer methods provably does better on this nonconvex stationarity measure — but each buys
the improvement with operational machinery (giant batches, carefully balanced and non-adaptive
learning rates, a periodic full-pass over the data) that is awkward and brittle in practice, to
the point where the faster methods are reported to underperform plain SGD on real deep-learning
workloads. The problem a solution must solve is to attain the better convergence guarantee
*while* keeping the per-step cost and the tuning burden of ordinary single-sample SGD: no large
batches, learning rates that adapt to the problem on their own, and no need to know the noise
level in advance.

## Background

**Stochastic gradient descent and its nonconvex rate.** The standard algorithm produces
iterates by `x_{t+1} = x_t - eta_t g_t` with `g_t = nabla f(x_t, xi_t)` an unbiased single-sample
gradient and `eta_t` a hand-tuned learning-rate sequence. With properly chosen step sizes, a
randomly selected iterate satisfies `E[||nabla F(x_t)||] <= O(T^{-1/4})` for smooth nonconvex
`F` (Ghadimi & Lan 2013). The single source of slowness is *gradient variance*: even arbitrarily
close to a critical point, where `nabla F(x_t) -> 0`, the sampled gradient `g_t` keeps a
variance floor `E[||g_t - nabla F(x_t)||^2] <= sigma^2` that does not vanish, so the iterate
keeps rattling and progress stalls.

**Variance reduction as a way past the SGD floor.** The idea, introduced independently by three
groups at the same conference (Johnson & Zhang 2013; Zhang, Mahdavi & Jin 2013; Wang, Chen,
Smola & Xing 2013) for finite sums, is to replace `g_t` by a lower-variance estimate of
`nabla F(x_t)` built from auxiliary information, while keeping the same `x_{t+1} = x_t - eta_t g_t`
update shape. Allen-Zhu & Hazan (2016) brought this to nonconvex SGD; subsequent work pushed the
nonconvex stationarity rate from `O(T^{-1/4})` down to `O(T^{-3/10})` (Allen-Zhu & Hazan 2016;
Reddi, Hefny, Sra, Poczos & Smola 2016) and then to `O(T^{-1/3})` (Fang, Li, Lin & Zhang 2018;
Zhou, Xu & Gu 2018). A central tool throughout is the *control variate built from smoothness*:
if the same sample `xi` is evaluated at two nearby points `x` and `x'`, the difference
`nabla f(x, xi) - nabla f(x', xi)` is, by `L`-smoothness, at most `L ||x - x'||` in norm, so it
is a *small, low-variance* random quantity whenever the two points are close — exactly the regime
an optimizer that is converging lives in. This is the lever every variance-reduced method pulls.

**The cost these methods pay — the diagnostic facts.** Two operational facts about the existing
variance-reduced methods set up the problem. First, to construct their low-variance estimate they
must periodically *stop producing iterates* and compute a "checkpoint" gradient
`(1/N) sum_{i=1}^N nabla f(v, xi_i)` at an anchor point `v` over a fresh batch of `N` samples;
depending on the method `N` can be as large as `O(T)` and is typically no smaller than `O(T^{2/3})`.
A full pass spends many samples learning about a single point `v` while making zero progress.
Second, their analyses use *non-adaptive*, carefully scheduled learning rates whose optimal
setting balances several unknown problem constants (the smoothness `L`, the noise `sigma`, the
batch size, the checkpoint frequency) against each other — so the constants must be known or
swept, and a mis-set schedule loses the guarantee. On top of this, it has been reported that
SVRG-style variance reduction is, empirically, *ineffective* for deep learning relative to plain
SGD (Defazio & Bottou 2019) — the theoretical speedup does not materialize on real nonconvex
workloads. The lower-bound side of the picture (Arjevani, Carmon, Duchi, Foster, Srebro &
Woodworth 2019) clarifies what is actually possible: under a mean-squared-smoothness oracle model
the query complexity to reach an `epsilon`-stationary point is `Theta(epsilon^{-3})`, so the
`O(T^{-1/3})` rate is the best attainable, and any method matching it is optimal.

**Adaptive learning rates.** A parallel line removes hand-tuning of `eta_t` by setting it from
the observed gradients. AdaGrad (Duchi, Hazan & Singer 2011) divides the step by the square root
of the running sum of squared gradients, `eta_t = k / (sum_{i<=t} g_i^2)^{1/2}`; RMSProp
(Tieleman & Hinton 2012) and Adam (Kingma & Ba 2015) use exponential moving averages of the same
quantity, and have become the practical default. In the nonconvex setting adaptive step sizes can
be shown to improve SGD's rate to `O(T^{-1/2} + (sigma^2/T)^{1/4})` (Li & Orabona 2019; Ward, Wu
& Bottou 2018; Reddi, Kale & Kumar 2018) — much faster when the noise is small — *without*
knowing `sigma`. These analyses lean on a Lipschitz (`||nabla f|| <= G`) assumption that the
variance-reduction analyses above do not use. The two threads had stayed almost entirely
separate: the only variance-reduced method with adaptive learning rates that existed
(Cutkosky & Busa-Fekete 2018) applied only to *convex* losses.

**Momentum, and a standing puzzle.** SGD with (heavy-ball) momentum keeps
`d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t)` (an exponential moving average of past gradients,
with `a` small, e.g. `0.1`) and steps `x_{t+1} = x_t - eta d_t`. It is ubiquitous and effective
in practice. Yet it is well documented that in the *stochastic* (noisy-gradient) setting the
noise can nullify momentum's theoretical advantage (Yuan, Ying & Sayed 2016): unlike the
noiseless/accelerated case, there is no general result showing momentum improves the
*convergence rate* of stochastic gradient methods. So momentum is a heuristic whose practical
success lacks a stochastic-setting explanation.

## Baselines

**SGD with momentum (heavy ball).** `d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t)`,
`x_{t+1} = x_t - eta d_t`. The moving average smooths minibatch noise and builds speed along
consistent directions. **Gap:** the smoothing introduces a lag/bias and, in the presence of
gradient noise, no analysis shows the averaged direction `d_t` is a *lower-variance* estimate of
`nabla F(x_t)` than a single sample; the theoretical gain known in the noiseless case is not known
to survive noise, so it does not by itself break the `O(T^{-1/4})` SGD barrier on nonconvex
problems.

**SVRG (Johnson & Zhang 2013).** Periodically fix a snapshot point `w_0` and compute its full
gradient `nabla F(w_0)` (a full pass), then in an inner loop use the control-variate estimate
`v_t = nabla f_i(w_t) - nabla f_i(w_0) + nabla F(w_0)`. This is *unbiased* for `nabla F(w_t)` and
its variance shrinks as the iterates approach the snapshot. **Gap:** it needs the full-gradient
snapshot (a checkpoint over the whole dataset / a mega-batch), and it must re-anchor the snapshot
periodically; the snapshot frequency and the inner-loop length are non-adaptive knobs that must be
balanced against `L` and `sigma`, and the method is reported to be ineffective on deep-learning
problems.

**SARAH (Nguyen, Liu, Scheinberg & Takac 2017).** Replace SVRG's fixed-anchor control variate by
a *recursive* one that re-anchors at the previous iterate every step:
`v_t = nabla f_i(w_t) - nabla f_i(w_{t-1}) + v_{t-1}`, initialized by a full gradient
`v_0 = (1/n) sum_i nabla f_i(w_0)`. Each step's correction term uses the *same* sample `i` at the
current and previous points, so by smoothness it is small; the estimate is *biased*
(`E[v_t | F_t] = nabla P(w_t) - nabla P(w_{t-1}) + v_{t-1} != nabla P(w_t)`), but accumulates
information across the inner loop and its step variance drives to zero within the loop, unlike
SVRG. **Gap:** it still opens each outer loop with a full-gradient checkpoint and restarts; its
learning rate is a fixed constant tuned against `L`, and (as published) it requires knowledge of
`sigma` and does not improve over plain SGD's nonconvex rate.

**SPIDER / SNVRG (Fang, Li, Lin & Zhang 2018; Zhou, Xu & Gu 2018).** Use the SARAH-style biased
recursive estimator together with carefully sized mega-batches to attain the optimal nonconvex
rate `O(T^{-1/3})`. **Gap:** the optimality is bought with large batches (`N` up to `O(T)`,
typically `>= O(T^{2/3})`) and a non-adaptive step size whose optimal value depends on knowing
`L` and `sigma`; the normalized variant `||v_t||`-clipped step is sensitive to these constants.

**AdaGrad / Adam (Duchi, Hazan & Singer 2011; Kingma & Ba 2015).** Set the step from observed
gradients, `eta_t propto (accumulated g^2)^{-1/2}`, removing learning-rate tuning and adapting to
the noise level (in nonconvex analyses, to `O(T^{-1/2} + (sigma^2/T)^{1/4})`). **Gap:** these are
*not* variance-reduced — `g_t` is still a single-sample gradient with the full `sigma^2` floor —
so the rate, while noise-adaptive, does not reach the variance-reduced `O(T^{-1/3})`; the
adaptivity and the variance reduction had not been combined for nonconvex losses.

## Evaluation settings

The natural yardsticks at the time. The headline metric for nonconvex stochastic optimization is
the gradient-norm stationarity measure `E[||nabla F(x_hat)||]` (or `E[(1/T) sum_t ||nabla F(x_t)||]`)
as a function of iterations / gradient queries `T`, since global optimality is intractable.
Algorithms are also compared on per-iteration and per-effective-pass cost (an "effective pass"
costing one full-gradient evaluation), which is what penalizes the mega-batch methods. The
empirical testbed is image classification on CIFAR-10 (Krizhevsky 2009) with a ResNet
architecture (He, Zhang, Ren & Sun 2016), as packaged in the Tensor2Tensor framework on top of
TensorFlow (Abadi et al. 2015), recording training cross-entropy loss and train/test accuracy
versus iterations; a smaller dataset such as MNIST (LeCun, Bottou, Bengio & Haffner 1998) is the
natural place to pick default hyperparameters on a logarithmic grid before transferring them.
Baselines are AdaGrad and Adam, with their learning rates swept over a logarithmically spaced
grid so that each method is shown at its best setting.

## Code framework

A new optimizer drops into the same minibatch training harness as the baselines: an optimizer
object owns per-parameter state and exposes a `step()` that consumes freshly computed gradients
and writes a small update into the parameters; an outer loop draws a sample / minibatch, runs the
existing model and loss, backpropagates to fill each parameter's gradient, and calls `step()`.
What is *not* yet decided is the gradient-estimate-and-step rule itself — that rule is exactly
what is to be designed — so the substrate exposes only the generic stochastic-optimization
machinery plus whatever per-parameter buffers a rule might want. The single empty slot is the
update rule.

```python
import tensorflow.compat.v1 as tf
from tensorflow.contrib import graph_editor as contrib_graph_editor
from tensorflow.contrib.optimizer_v2 import optimizer_v2

GATE_OP = 1


class StreamingOptimizer(optimizer_v2.OptimizerV2):
    """Generic stochastic optimizer slotting into a TensorFlow training graph."""

    def __init__(self, output_summaries=False, use_locking=False, name="StreamingOptimizer"):
        super(StreamingOptimizer, self).__init__(use_locking, name)
        self.output_summaries = output_summaries

    def _find_read_tensors(self, outputs, target):
        # TODO: identify the graph reads that an update rule may need to substitute.
        pass

    def _make_replace_dict(self, state, grads, var_list):
        # TODO: build any graph substitutions the update rule needs.
        pass

    def _recompute_gradients(self, state):
        # TODO: optionally recompute gradients under the rule's graph substitutions.
        pass

    def _create_slot_with_value(self, state, var, value, name):
        state.create_slot(
            var, tf.constant(value, shape=var.shape, dtype=var.dtype.base_dtype), name)

    def _create_vars(self, var_list, state):
        for var in var_list:
            # TODO: allocate the per-variable buffers the update rule needs.
            pass

    def _prepare(self, state):
        self.grads = []
        self.vars = []

    def _resource_apply_dense(self, grad, var, state):
        return self._apply_dense(grad, var, state)

    def _apply_dense(self, grad, var, state):
        self.grads.append(grad)
        self.vars.append(var)
        return tf.no_op()

    def _finish(self, state):
        update_ops = []
        auxiliary_grads = self._recompute_gradients(state)
        for var, grad, auxiliary_grad in zip(self.vars, self.grads, auxiliary_grads):
            # TODO: form the update from the current stochastic gradient
            # and whatever state or auxiliary gradient the rule keeps,
            # then apply it to `var`.
            update_ops.append(tf.no_op())
        return tf.group(*update_ops)

    def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                          aggregation_method=None, grad_loss=None,
                          stop_gradients=None, colocate_gradients_with_ops=False,
                          scale_loss_by_num_replicas=None):
        return super(StreamingOptimizer, self).compute_gradients(
            loss, var_list, gate_gradients, aggregation_method, grad_loss,
            stop_gradients, scale_loss_by_num_replicas)
```

The outer loop supplies stochastic gradients through the standard optimizer interface; `_finish()`
is where the rule will live.
