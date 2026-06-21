We are handed a smooth, possibly nonconvex function $F:\mathbb{R}^d\to\mathbb{R}$ that we can touch only through a stochastic oracle $f(x,\xi)$ with $\mathbb{E}[f(x,\xi)]=F(x)$, and we want to drive it toward a critical point, $\|\nabla F(x)\|=0$, using as few gradient queries as possible. Plain SGD, $x_{t+1}=x_t-\eta_t g_t$ with $g_t=\nabla f(x_t,\xi_t)$, reaches $\mathbb{E}[\|\nabla F(x_t)\|]\le O(T^{-1/4})$, and the single thing strangling that rate is gradient variance: even as $\nabla F(x_t)\to 0$ near a critical point, the sampled gradient keeps a noise floor $\mathbb{E}[\|g_t-\nabla F(x_t)\|^2]\le\sigma^2$ that does not vanish, so the iterate rattles and progress stalls. The variance-reduction family beats this floor by building a lower-noise estimate of $\nabla F(x_t)$ out of a smoothness control variate: evaluating the *same* sample $\xi$ at two nearby points $x,x'$ gives a difference $\nabla f(x,\xi)-\nabla f(x',\xi)$ that is, by $L$-smoothness, at most $L\|x-x'\|$ in norm and small *in variance* precisely when the points are close — exactly the regime a converging optimizer lives in. SVRG anchors this control variate at a frozen snapshot whose full gradient $\nabla F(w_0)$ it pays for with a whole pass over the data; SARAH re-anchors recursively at the previous iterate, $v_t=\nabla f_i(w_t)-\nabla f_i(w_{t-1})+v_{t-1}$, seeded by a full gradient $v_0$; SPIDER and SNVRG run the SARAH-style biased recursive estimator with carefully sized mega-batches and reach the optimal nonconvex rate $O(T^{-1/3})$, which Arjevani et al. (2019) show is the best attainable. But all of them share two costs I cannot stomach: a periodic checkpoint that burns a fresh batch of $N$ samples (as large as $O(T)$, never smaller than $O(T^{2/3})$) to learn about a single anchor point while making zero progress, and a non-adaptive learning rate that must be set by balancing $L$, $\sigma$, the batch size, and the checkpoint frequency against each other, so the constants must be known or swept. Empirically the whole family is reported to underperform plain SGD on real deep nets. The separate adaptive-step line — AdaGrad's $\eta_t=k/(\sum_{i\le t}g_i^2)^{1/2}$, RMSProp, Adam — removes the tuning and adapts to the noise level but is *not* variance-reduced, so it cannot reach $O(T^{-1/3})$. We want the optimal rate with neither a checkpoint nor a known $\sigma$.

I propose STORM, for STochastic Recursive Momentum. The starting observation is that SARAH's recursive estimator, written as $v_t=\nabla f_i(w_t)+(v_{t-1}-\nabla f_i(w_{t-1}))$, and heavy-ball momentum, $d_t=(1-a)d_{t-1}+a\,\nabla f(x_t,\xi_t)$, are both "carry the running estimate forward and fold in the new gradient" — the only reason SARAH needs a checkpoint is its full-gradient seed $v_0$, while momentum's $(1-a)$ averaging is a contraction that could supply the missing variance reduction. So I graft the SARAH same-sample correction directly onto the momentum recursion, weighting the correction by $(1-a)$ to match the "old" part of $d$:
$$d_t=(1-a)d_{t-1}+a\,\nabla f(x_t,\xi_t)+(1-a)\big(\nabla f(x_t,\xi_t)-\nabla f(x_{t-1},\xi_t)\big),\qquad x_{t+1}=x_t-\eta\,d_t.$$
Collecting the two $\nabla f(x_t,\xi_t)$ pieces ($a g+(1-a)g=g$) gives the equivalent compact form $d_t=\nabla f(x_t,\xi_t)+(1-a)(d_{t-1}-\nabla f(x_{t-1},\xi_t))$, which is exactly SARAH's recursion with a $(1-a)$ weight on the carry-forward instead of a hard $1$. This two-parameter object contains both ancestors as corners: dropping the correction term is plain momentum SGD, and $a=0$ is SARAH's pure carry-forward; the interesting region is the interior, $a$ small but nonzero. Notice immediately that if the algorithm is converging, $x_t\approx x_{t-1}$, the correction term vanishes and $d_t$ collapses to plain momentum — it is not an exotic object, just momentum with an extra correction that matters most early when steps are large, and no checkpoint anywhere.

What makes it work is that the error in the update direction shrinks instead of sitting at SGD's floor. Define $\epsilon_t:=d_t-\nabla F(x_t)$. Subtracting $\nabla F(x_t)$ from the recursion (adding and subtracting $(1-a)\nabla F(x_{t-1})$, and splitting the leading gradient) decomposes the error into three readable terms:
$$\epsilon_t=(1-a)\epsilon_{t-1}+a\big(\nabla f(x_t,\xi_t)-\nabla F(x_t)\big)+(1-a)\Big(\nabla f(x_t,\xi_t)-\nabla f(x_{t-1},\xi_t)-\big(\nabla F(x_t)-\nabla F(x_{t-1})\big)\Big).$$
The first is a *contraction* of the previous error by $(1-a)$ — the momentum averaging doing its job. The second is fresh single-sample noise of magnitude $\sim\sigma$, but multiplied by the small $a$, so I can shrink it by shrinking $a$. The third is the centered same-sample two-point difference, $O(\|x_t-x_{t-1}\|)=O(\eta\|d_{t-1}\|)$ by smoothness, small whenever the step is small. A contraction-plus-small-input recursion settles at the fixed point $\|\epsilon\|\sim Z/a$, so the entire design problem is to make $Z/a$ tiny — drive the numerator down without letting the denominator $a$ collapse.

I refuse to hand-tune the step size and momentum coefficient, so I set them adaptively from observed gradients and let the convergence analysis pin the exponents rather than assuming AdaGrad's $1/2$ power. Writing $\eta_t=k/(w+\sum_{i=1}^t G_i^2)^{p}$ with $G_t=\|\nabla f(x_t,\xi_t)\|$, $w$ a small offset, and $a_{t+1}=c\,\eta_t^2$ — coupling $a$ to $\eta^2$ because then the squared noise term scales as $a^2=c^2\eta^4$, which is summable — the equilibrium heuristic fixes $p$. Since the variance floor keeps $G_t^2>0$ even at a critical point, $\sum G_i^2=\Theta(t)$, so $\eta_t\sim t^{-p}$ and $a_t\sim t^{-2p}$. Balancing the error recursion at equilibrium gives $\|\epsilon\|^2\sim t^{-2p}+\|\nabla F\|^2$, and the iterate stops improving when $\|\nabla F\|^2$ falls to the floor $t^{-2p}$, i.e. $\|\nabla F\|\sim T^{-p}$. To hit the optimal $T^{-1/3}$ I need $p=1/3$: the *cube* root of accumulated squared gradients, not AdaGrad's square root, which would deliver only the easy-case $T^{-1/2}$ and never the variance-reduced rate in the noisy regime. With $p=1/3$, $a_t\sim t^{-2/3}$, decaying exactly as wanted. The schedule is therefore $\eta_t=k/(w+\sum_{i=1}^t G_i^2)^{1/3}$, $a_{t+1}=c\,\eta_t^2$, $d_{t+1}=\nabla f(x_{t+1},\xi_{t+1})+(1-a_{t+1})(d_t-\nabla f(x_t,\xi_{t+1}))$, $x_{t+1}=x_t-\eta_t d_t$, seeded by a single sample $d_1=\nabla f(x_1,\xi_1)$, $\eta_0=k/w^{1/3}$.

The proof that this actually achieves the rate rests on a *time-varying* Lyapunov potential $\Phi_t=F(x_t)+z_t\|\epsilon_t\|^2$ with $z_t=1/(32L^2\eta_{t-1})$ — the $1/\eta_{t-1}$ weight is precisely what removes the checkpoint, since a constant weight would not track the shrinking step and would need a low-noise anchor to seed. A biased descent lemma (smoothness, Young's inequality on the cross term $-\eta_t\nabla F\cdot\epsilon_t$, and $\|a+b\|^2\le 2\|a\|^2+2\|b\|^2$) gives, for $\eta_t\le 1/(4L)$,
$$\mathbb{E}[F(x_{t+1})-F(x_t)]\le\mathbb{E}\big[-(\eta_t/4)\|\nabla F(x_t)\|^2+(3\eta_t/4)\|\epsilon_t\|^2\big],$$
where the $3/4$-versus-$1/4$ asymmetry is the tax for using a biased direction. The error recursion, expanding $\epsilon_t=P+Q+R$ with $P,Q$ conditionally mean-zero in the fresh sample $\xi_t$ (so their cross terms with $R=(1-a_t)\epsilon_{t-1}$ vanish — the unbiasedness substitute for a biased $d_t$), yields $\mathbb{E}[\|\epsilon_t\|^2/\eta_{t-1}]\le\mathbb{E}[2c^2\eta_{t-1}^3 G_t^2+(1-a_t)^2(1+4L^2\eta_{t-1}^2)\|\epsilon_{t-1}\|^2/\eta_{t-1}+4(1-a_t)^2 L^2\eta_{t-1}\|\nabla F(x_{t-1})\|^2]$, using $\|\nabla f(x_t,\xi)-\nabla f(x_{t-1},\xi)\|^2\le L^2\eta_{t-1}^2\|d_{t-1}\|^2$. Telescoping is where the constants are forced: the noise term sums to only $\sum_t 2c^2\eta_t^3 G_{t+1}^2\le 2k^3c^2\ln(T+2)$ via the log lemma $\sum_t a_t/(a_0+\sum_{i\le t}a_i)\le\ln(1+\sum a_i/a_0)$ — *logarithmic*, where SGD pays linearly in $T$, and that is the variance-reduction payoff in the math. The contraction term $B_t\le(\eta_t^{-1}-\eta_{t-1}^{-1}+\eta_t(4L^2-c))\|\epsilon_t\|^2$ must come out negative; concavity of $x^{1/3}$ bounds the step-size increment $\eta_t^{-1}-\eta_{t-1}^{-1}\le(G^2/(7Lk^3))\eta_t$, so choosing $c=28L^2+G^2/(7Lk^3)$ — the $28L^2$ producing a surplus, the second piece exactly eating the increment — gives $B_t\le-24L^2\eta_t\|\epsilon_t\|^2$. Scaled by $1/(32L^2)$, this $-24L^2\eta_t$ becomes $-(3\eta_t/4)\|\epsilon_t\|^2$, which *exactly cancels* the descent lemma's $+(3\eta_t/4)$ — the cancellation that the $z_t\propto1/\eta$ weight and the $28L^2$ were tuned to land. Summed, $\mathbb{E}[\sum_t\eta_t\|\nabla F(x_t)\|^2]\le 8(F(x_1)-F^*)+w^{1/3}\sigma^2/(4L^2k)+(k^3c^2/2L^2)\ln(T+2)$, the single seed $\mathbb{E}[\|\epsilon_1\|^2]\le\sigma^2$ standing in for the checkpoint. Stripping the weights with $\eta_t$ decreasing and Cauchy–Schwarz yields the theorem: with $k=bG^{2/3}/L$, $c=L^2(28+1/(7b^3))$, $w=G^2\max((4b)^3,2,(28b+1/(7b^2))^3/64)$,
$$\mathbb{E}[\|\nabla F(\hat x)\|]\le\frac{w^{1/6}\sqrt{2M}+2M^{3/4}}{\sqrt T}+\frac{2\sigma^{1/3}}{T^{1/3}},\quad M=\tfrac{8}{k}(F(x_1)-F^*)+\tfrac{w^{1/3}\sigma^2}{4L^2k^2}+\tfrac{k^2c^2}{2L^2}\ln(T+2).$$
When $\sigma=0$ this is $O(\ln T/\sqrt T)$; when $\sigma\ne 0$ the dominant term is $2\sigma^{1/3}/T^{1/3}$, the optimal nonconvex rate, achieved *without ever knowing $\sigma$* — it falls out of the adaptive $\eta_t$. If one is willing to be told $\sigma$ instead, the $G$-Lipschitz assumption can be dropped by replacing $G,G_t$ with $\sigma$ to make $\eta_t=k/(w+\sigma^2 t)^{1/3}$ a deterministic schedule, removing the final Cauchy–Schwarz and giving the same $O(T^{-1/3})$-flavored guarantee.

The implementation hides a few decisions the clean rule does not. There is one sample per step but two parameter sets in play, the current $x_t$ and previous $x_{t-1}$, and I need the *same* sample's gradient at *both* — in a static TensorFlow graph I keep a slot holding the previous iterate and use graph replacement to recompute the current loss gradient with each variable read swapped to that slot. The per-variable slots are the previous iterate, the running estimate $d$, an elementwise squared-gradient accumulator, a scalar running maximum gradient norm for clipping, and a diagnostic estimate accumulator. The theorem's scalar $\eta_t$ becomes the same cube-root law applied elementwise (`sum_grad_squared += grad^2`, the diagonal preconditioning practitioners expect); $a$ becomes `beta = min(1, momentum * eta^2)`, the cap being the numerical version of the proof's $a\le 1$ constraint; the update is the compact formula `grad + (1-beta)(grad_estimate - grad_at_prev_iterate)`, scalar-norm-clipped to the running gradient scale, with the current variable saved as the next previous iterate before applying `var += -eta * new_grad_estimate`. Mapping names: `lr` is $k$, `eta` is the offset $w$, `momentum` is $c$, `g_max` initializes the gradient scale; in the training setup $k=w=0.1$ are picked on a smaller dataset and only $c$ is swept, leaving $c$ as the effective knob.

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
