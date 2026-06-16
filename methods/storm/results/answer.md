# STORM, distilled

STORM (STochastic Recursive Momentum) is a variance-reduced optimizer for smooth nonconvex
stochastic optimization that reaches the optimal `O(T^{-1/3})` stationarity rate **without any
checkpoint / mega-batch and without knowing the noise level**, using a momentum recursion that
folds in a SARAH-style same-sample correction, plus an AdaGrad-style adaptive step. It uses one
fresh sample per step (evaluated at the current and previous iterates), a small set of per-variable
slots, and a TensorFlow training setup where defaults for `k` and `w` leave `c` as the swept knob.

## Problem it solves

Minimize a smooth, possibly nonconvex `F(x)` accessed only through a stochastic oracle
`f(x, xi)`, `E[f] = F`, with gradient variance `<= sigma^2`. Goal: find a critical point,
`E[||nabla F(x)||]` small, in as few gradient queries as possible. SGD gets `O(T^{-1/4})`;
variance-reduced methods reach the optimal `O(T^{-1/3})` (Arjevani et al. 2019 show
`Theta(epsilon^{-3})` is a lower bound) but pay for it with full-gradient checkpoints / large
batches and non-adaptive, hard-to-tune learning rates. STORM keeps the optimal rate with neither.

## Key idea

Track the **error in the update direction**, `epsilon_t := d_t - nabla F(x_t)`, and make it
*shrink* instead of sitting at SGD's `sigma^2` floor. Build `d_t` by grafting SARAH's recursive
same-sample correction onto heavy-ball momentum:

```
d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t) + (1-a)(nabla f(x_t, xi_t) - nabla f(x_{t-1}, xi_t))
    = nabla f(x_t, xi_t) + (1-a)(d_{t-1} - nabla f(x_{t-1}, xi_t))          (equivalent form)
x_{t+1} = x_t - eta d_t
```

The only change from plain momentum is the term `(1-a)(nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t))`:
the **same** sample `xi_t` evaluated at the current and previous points, so by `L`-smoothness it is
`O(||x_t - x_{t-1}||)` — a low-variance control variate, small exactly when the iterates are close.
Subtracting `nabla F(x_t)`:

```
epsilon_t = (1-a) epsilon_{t-1}
          + a(nabla f(x_t,xi_t) - nabla F(x_t))                                   # ~ a*sigma noise, small via small a
          + (1-a)(nabla f - nabla f' - (nabla F(x_t) - nabla F(x_{t-1})))         # ~ O(eta ||d||) by smoothness
```

a contraction `(1-a)` of the old error plus two small inputs, so `||epsilon_t||` settles at `~ Z/a`;
the design is to make `Z/a` tiny. **Limiting cases:** drop the correction term -> plain SGD
momentum; set `a = 0` -> SARAH's recursive estimator (which needs a full-gradient seed); STORM is
the interior, no checkpoint needed because the `(1-a)` contraction + smoothness term do the variance
reduction that SARAH bought with `v_0 = full gradient`.

## Adaptive schedule (Algorithm)

```
Input: k, w, c, x_1
G_1 = ||nabla f(x_1, xi_1)||;  d_1 = nabla f(x_1, xi_1);  eta_0 = k / w^{1/3}
for t = 1 .. T:
    eta_t   = k / (w + sum_{i=1}^{t} G_i^2)^{1/3}        # AdaGrad-style, but CUBE root
    x_{t+1} = x_t - eta_t d_t
    a_{t+1} = c eta_t^2                                  # momentum decays like t^{-2/3}
    sample xi_{t+1};  G_{t+1} = ||nabla f(x_{t+1}, xi_{t+1})||
    d_{t+1} = nabla f(x_{t+1}, xi_{t+1}) + (1 - a_{t+1})(d_t - nabla f(x_t, xi_{t+1}))
return x_hat ~ Uniform{x_1..x_T}   (in practice x_T)
```

Why these powers: `sum G^2 = Theta(t)` (variance keeps `||g||^2 > 0` even at critical points), so
`eta_t = Theta(t^{-1/3})`, `a_t = Theta(t^{-2/3})`. The error recursion then equilibrates at
`||epsilon_t||^2 = O(t^{-2/3} + ||nabla F||^2)`, halting progress at `||nabla F||^2 = O(t^{-2/3})`,
i.e. `||nabla F|| = O(T^{-1/3})`. AdaGrad's `1/2` power would give the easy-case rate, not the
variance-reduced one; the `1/3` is tuned to the equilibrium. `a = c eta^2` makes the noise term
`a^2 = c^2 eta^4` summable and `a` decay at the rate the equilibrium needs.

## Theorem (optimal, noise-adaptive)

Assume `f(.,xi)` is `L`-smooth and `G`-Lipschitz w.p. 1, `E[||nabla f - nabla F||^2] <= sigma^2`,
`F^* > -infinity`. For any `b > 0` set
`k = b G^{2/3}/L`, `c = 28L^2 + G^2/(7Lk^3) = L^2(28 + 1/(7b^3))`,
`w = max((4Lk)^3, 2G^2, (ck/4L)^3) = G^2 max((4b)^3, 2, (28b + 1/(7b^2))^3/64)`. Then

```
E[||nabla F(x_hat)||] = E[(1/T) sum_t ||nabla F(x_t)||]
   <= (w^{1/6} sqrt(2M) + 2 M^{3/4}) / sqrt(T)  +  2 sigma^{1/3} / T^{1/3},
M = (8/k)(F(x_1) - F^*) + w^{1/3} sigma^2 / (4 L^2 k^2) + (k^2 c^2 / (2L^2)) ln(T+2).
```

`sigma = 0` -> `O(ln T / sqrt T)`; `sigma != 0` -> `2 sigma^{1/3}/T^{1/3}`, the optimal nonconvex
rate, achieved **without knowing `sigma`** (it falls out of the adaptive `eta_t`). `M` is morally
an `O(log T)` problem-hardness constant; `G -> 0` is benign (`F(x_1)-F^* = O(G)`, `sigma = O(G)`),
`L -> 0` correctly diverges (no critical point exists when all gradients are equal).

## Proof sketch (Lyapunov)

Potential `Phi_t = F(x_t) + ||epsilon_t||^2 / (32 L^2 eta_{t-1})` — the **time-varying** weight
`z_t propto 1/eta_{t-1}` is what removes the need for a checkpoint (constant `z_t` would require one).

1. **Biased descent lemma** (`eta_t <= 1/4L`):
   `E[F(x_{t+1}) - F(x_t)] <= E[-(eta_t/4)||nabla F(x_t)||^2 + (3 eta_t/4)||epsilon_t||^2]`.
   (Smoothness + Young + `||a+b||^2 <= 2||a||^2 + 2||b||^2`; the `3/4` vs `1/4` is the bias tax.)
2. **Error recursion:**
   `E[||epsilon_t||^2/eta_{t-1}] <= E[2c^2 eta_{t-1}^3 G_t^2 + (1-a_t)^2(1+4L^2 eta_{t-1}^2)||epsilon_{t-1}||^2/eta_{t-1} + 4(1-a_t)^2 L^2 eta_{t-1}||nabla F(x_{t-1})||^2]`.
   (Expand `epsilon_t = P + Q + R`; `P,Q` are conditionally mean-zero in `xi_t`, so cross terms with
   `R = (1-a_t)epsilon_{t-1}` vanish — the unbiasedness substitute for a biased `d_t`; in expectation,
   `E[eta_{t-1}^3 ||nabla f - nabla F||^2] <= E[eta_{t-1}^3 G_t^2]`, and smoothness gives
   `||nabla f(x_t,xi) - nabla f(x_{t-1},xi)||^2 <= L^2||x_t - x_{t-1}||^2 = L^2 eta_{t-1}^2 ||d_{t-1}||^2`.)
3. **Telescope.** `sum A_t = sum 2c^2 eta_t^3 G_{t+1}^2 <= 2k^3 c^2 ln(T+2)` via the log lemma
   `sum a_t/(a_0 + sum_{i<=t} a_i) <= ln(1 + sum a_i/a_0)` (noise is only *logarithmic* — the payoff).
   `B_t <= (eta_t^{-1} - eta_{t-1}^{-1} + eta_t(4L^2 - c))||epsilon_t||^2`; concavity of `x^{1/3}`
   gives `eta_t^{-1} - eta_{t-1}^{-1} <= (G^2/(7Lk^3)) eta_t`, and `c = 28L^2 + G^2/(7Lk^3)` makes
   `B_t <= -24 L^2 eta_t ||epsilon_t||^2`. Scaling by `1/(32L^2)`, this `-24L^2 eta_t` becomes
   `-(3 eta_t/4)||epsilon_t||^2`, which **exactly cancels** the descent lemma's `+(3 eta_t/4)`.
   Summed: `E[sum eta_t ||nabla F||^2] <= 8(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2 k) + (k^3 c^2/2L^2)ln(T+2)`,
   the seed `E[||epsilon_1||^2] <= sigma^2` replacing a checkpoint.
4. **Strip the weights.** `eta_t` decreasing + Cauchy-Schwarz `E[A^2]E[B^2] >= E[AB]^2` ->
   `E[X]^2 <= M(w + 2T sigma^2)^{1/3} + 2^{1/3} M E[X]^{2/3}`, `X = sqrt(sum ||nabla F||^2)`; the
   two cases give `E[X] <= sqrt(2M)(w + 2Tsigma^2)^{1/6} + 2M^{3/4}`; final Cauchy-Schwarz
   `(1/T) sum ||nabla F|| <= X/sqrt T` yields the theorem.

**No-Lipschitz variant:** if `sigma` is known, drop `G`-Lipschitz by using the deterministic
schedule `eta_t = k/(w + sigma^2 t)^{1/3}` (replace `G`, `G_t` by `sigma`); then `eta_t` is
gradient-independent, the final Cauchy-Schwarz is unneeded, and
`(1/T) E[sum ||nabla F||^2] <= M_det w^{1/3}/(kT) + M_det sigma^{2/3}/(k T^{2/3})`, with
`M_det = 8(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2 k) + (k^3 c^2/2L^2)ln(T+2)`, trading
`sigma`-adaptivity for removing the Lipschitz assumption.

## Working code

Faithful implementation shape: TensorFlow slots hold the previous iterate, the running estimate,
the elementwise squared-gradient accumulator, a scalar running max-gradient norm, and a diagnostic
estimate accumulator. Names: `lr = k`, `eta = w`, `momentum = c`, `g_max` initializes the gradient
scale. The theorem uses a scalar accumulator; the implementation uses the same cube-root schedule
elementwise, then caps `a` with `min(1, c eta^2)`, clips the estimate by the scalar gradient scale,
records optional summaries, and keeps the surrounding training framework's `compute_gradients`
signature.

```python
import tensorflow.compat.v1 as tf
from tensorflow.contrib import graph_editor as contrib_graph_editor
from tensorflow.contrib.optimizer_v2 import optimizer_v2

GATE_OP = 1

PREVIOUS_ITERATE = "previous_iterate"
GRAD_ESTIMATE = "grad_estimate"
SUM_GRAD_SQUARED = "sum_grad_squared"
MAXIMUM_GRADIENT = "maximum_gradient"
SUM_ESTIMATES_SQUARED = "sum_estimates_squared"


class StormOptimizer(optimizer_v2.OptimizerV2):
  def __init__(self, lr=1.0, g_max=0.01, momentum=100.0, eta=10.0,
               output_summaries=False, use_locking=False,
               name="StormOptimizer"):
    super(StormOptimizer, self).__init__(use_locking, name)
    self.lr = lr
    self.g_max = g_max
    self.momentum = momentum
    self.eta = eta
    self.output_summaries = output_summaries

  def _find_read_tensors(self, outputs, target):
    read_tensors, visited = set(), set()

    def dfs(parent):
      for x in parent.op.inputs:
        if x.name not in visited:
          if x.name == target.name:
            read_tensors.add(parent)
          visited.add(x.name)
          dfs(x)

    for output in outputs:
      dfs(output)
    return read_tensors

  def _make_replace_dict(self, state, grads, var_list):
    replace_dict = {}
    for var in var_list:
      previous_iterate = tf.convert_to_tensor(state.get_slot(var, PREVIOUS_ITERATE))
      for tensor in self._find_read_tensors(grads, var):
        replace_dict[tensor] = previous_iterate
    return replace_dict

  def _recompute_gradients(self, state):
    return contrib_graph_editor.graph_replace(
        self.grads, self._make_replace_dict(state, self.grads, self.vars))

  def _create_slot_with_value(self, state, var, value, name):
    state.create_slot(
        var, tf.constant(value, shape=var.shape, dtype=var.dtype.base_dtype), name)

  def _create_vars(self, var_list, state):
    for var in var_list:
      state.create_slot(var, var.initialized_value(), PREVIOUS_ITERATE)
      self._create_slot_with_value(state, var, self.g_max ** 3, SUM_GRAD_SQUARED)
      state.zeros_slot(var, GRAD_ESTIMATE)
      state.create_slot(
          var, tf.constant(self.g_max, dtype=var.dtype.base_dtype), MAXIMUM_GRADIENT)
      self._create_slot_with_value(state, var, 0.01, SUM_ESTIMATES_SQUARED)

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
    grads_at_prev_iterate = self._recompute_gradients(state)

    for var, grad, grad_at_prev_iterate in zip(
        self.vars, self.grads, grads_at_prev_iterate):
      sum_grad_squared = state.get_slot(var, SUM_GRAD_SQUARED)
      previous_iterate = state.get_slot(var, PREVIOUS_ITERATE)
      maximum_gradient = state.get_slot(var, MAXIMUM_GRADIENT)
      grad_estimate = state.get_slot(var, GRAD_ESTIMATE)
      sum_estimates_squared = state.get_slot(var, SUM_ESTIMATES_SQUARED)

      maximum_gradient_updated = tf.assign(
          maximum_gradient, tf.maximum(maximum_gradient, tf.norm(grad)))
      update_ops.append(maximum_gradient_updated)

      sum_grad_squared_updated = tf.assign_add(
          sum_grad_squared, tf.pow(tf.abs(grad), 2.0))
      update_ops.append(sum_grad_squared_updated)

      smoothness = tf.norm(grad - grad_at_prev_iterate) / (
          0.0001 + tf.norm(var - previous_iterate))
      eta = self.lr * tf.pow(self.eta + sum_grad_squared_updated, -1.0 / 3.0)
      beta = tf.minimum(1.0, self.momentum * tf.square(eta))

      new_grad_estimate = grad + (1.0 - beta) * (
          grad_estimate - grad_at_prev_iterate)
      new_grad_estimate = tf.clip_by_value(
          new_grad_estimate, -maximum_gradient_updated, maximum_gradient_updated)

      if self.output_summaries:
        tf.summary.scalar(self._name + "/smoothness/" + var.name, smoothness)
        tf.summary.scalar(self._name + "/max_grad/" + var.name,
                          maximum_gradient_updated)
        tf.summary.scalar(self._name + "/average_beta/" + var.name,
                          tf.reduce_mean(beta))
        tf.summary.scalar(self._name + "/iterate_diff/" + var.name,
                          tf.norm(var - previous_iterate))
        tf.summary.scalar(self._name + "/grad_diff/" + var.name,
                          tf.norm(grad - grad_at_prev_iterate))
        tf.summary.scalar(self._name + "/vr_grad_estimate_norm/" + var.name,
                          tf.norm(new_grad_estimate))
        tf.summary.scalar(self._name + "/grad_norm/" + var.name, tf.norm(grad))

      grad_estimate_updated = tf.assign(grad_estimate, new_grad_estimate)
      update_ops.append(grad_estimate_updated)

      update_ops.append(tf.assign_add(
          sum_estimates_squared, tf.square(new_grad_estimate)))

      with tf.control_dependencies([grad_at_prev_iterate]):
        previous_iterate_updated = tf.assign(previous_iterate, var)
        update_ops.append(previous_iterate_updated)

      with tf.control_dependencies([previous_iterate_updated]):
        update_ops.append(tf.assign_add(var, -eta * grad_estimate_updated))

    return tf.group(*update_ops)

  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None, grad_loss=None,
                        stop_gradients=None, colocate_gradients_with_ops=False,
                        scale_loss_by_num_replicas=None):
    return super(StormOptimizer, self).compute_gradients(
        loss, var_list, gate_gradients, aggregation_method, grad_loss,
        stop_gradients, scale_loss_by_num_replicas)
```

In the experimental setup, `k = w = 0.1` are selected on a smaller dataset and only `c` is swept on
a logarithmic grid for the target benchmark, leaving `c` as the effective knob in that setup.
