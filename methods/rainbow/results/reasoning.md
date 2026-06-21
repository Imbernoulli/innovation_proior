I start with the base DQN update because that tells me where every later change has to attach. The
online network supplies the behavior policy and receives gradients, while the target network is a
frozen copy used inside the bootstrap. The scalar target is
$R_{t+1}+\gamma_{t+1}\max_{a'}q_{\bar\theta}(S_{t+1},a')$. That target has one action-selection
step, one evaluation step, one scalar error, and one network head producing one value per action.
If I combine the available improvements carelessly, I will redefine the same pieces several times,
so I need to choose an order that exposes the conflicts rather than hiding them.

The most invasive change is the categorical distributional one, because it changes the object being
learned. Once the network predicts a probability vector over atoms for each action, the scalar
$q(s,a)$ becomes the mean $z^\top p(s,a)$, and the native loss is no longer a squared TD error. So I
make the return distribution the spine of the construction. The atom support is fixed:
$z^i=v_{\min}+(i-1)(v_{\max}-v_{\min})/(N_{\text{atoms}}-1)$, with 51 atoms and support
$[-10,10]$ in the Atari setting. The target distribution can move off that support after reward
shifting and discounting, so it must be projected back with the categorical projection
$\Phi_z$ before comparison.

Now I fold in multi-step learning. In the scalar case, I would replace the one reward with
$R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1}$ and bootstrap from $S_{t+n}$ with
$\gamma_t^{(n)}$. In distribution space the same idea is to take the distribution at $S_{t+n}$,
contract its atom locations by the cumulative discount, and shift by the truncated return:
$d_t^{(n)}=(R_t^{(n)}+\gamma_t^{(n)}z,\ p_{\bar\theta}(S_{t+n},a^*_{t+n}))$. If the sampled
sequence terminates before the $n$th next state, the nonterminal mask makes the bootstrap term zero
and the projected target collapses to the observed truncated return. The signs here matter: reward
shifts the support additively, discount multiplies the future atoms, and the clamp is to
$[v_{\min},v_{\max}]$ after the shift and contraction.

Double Q-learning has only one place to attach after that: the choice of $a^*_{t+n}$. The
distributional baseline can choose the next action using target-network means, but the double-Q
version must decouple selection from evaluation. Therefore I select with the online network mean,
$a^*_{t+n}=\arg\max_a z^\top p_\theta(S_{t+n},a)$, and evaluate by reading the whole probability
vector from the target network, $p_{\bar\theta}(S_{t+n},a^*_{t+n})$. I do not mix the online
probabilities into the target distribution; the online network only selects the action.

Prioritized replay is the next collision. My first scalar priority candidate is the old absolute TD
error computed from means, because it is available. But that is the wrong native error for this
agent: the optimization target is the projected distribution, and a stochastic transition can keep a
large sampled TD error even when the predicted return distribution is already correct. So the
priority should be the per-sample categorical loss itself. In math this is written as
$p_t\propto(D_{\mathrm{KL}}(\Phi_zd_t^{(n)}\|d_t))^\omega$. In code, the minimized quantity is the
cross-entropy $-\sum_i m_i\log p_i$ because the projected target distribution $m=\Phi_zd_t^{(n)}$
is fixed under the gradient, so it differs from the KL only by the target entropy. The replay buffer
stores the raw per-sample loss and applies the exponent $\omega$ in its priority update; the
importance-sampling weights multiply the loss for the gradient but are not part of the new priority.

The categorical projection has two edge cases that I have to keep explicit. For each target atom
location $Tz_j$, I compute $b=(Tz_j-v_{\min})/\Delta z$ and send probability mass to
$\lfloor b\rfloor$ and $\lceil b\rceil$ with weights $u-b$ and $b-l$. If $b$ is exactly an integer,
then $l=u$ and both weights are zero, which would drop all probability mass. The implementation fixes
that by decrementing $l$ when possible, or incrementing $u$ when possible, so the two index-add terms
still assign the full mass to the exact atom. The terminal case is separate: the nonterminal mask
removes the discounted atom term before clamping and projection, so terminal targets do not leak
future target-network values.

Now the dueling head has to be reinterpreted. The scalar version combines a value scalar and
advantage scalars, subtracting the mean advantage to pin the decomposition. If I aggregate after
taking expectations, I lose the categorical object. The correct move is to make the value stream
produce one logit per atom and the advantage stream produce one logit per action per atom. For each
atom I compute $v^i+a^i-\bar a^i$, then I apply a softmax over atoms independently for each action.
The mean subtraction is therefore per atom and before normalization, not on already-normalized
probabilities and not on scalar action means.

Noisy layers are simpler but still have a faithfulness trap. The rule is to replace linear
layers with noisy linear equivalents and act greedily with $\epsilon=0$ when noise is used. In the
PyTorch reference I am matching, the convolutional torso remains deterministic and all fully
connected layers in the value and advantage streams are `NoisyLinear`. The factorized noise is
$f(x)=\operatorname{sign}(x)\sqrt{|x|}$, the mean parameters are initialized uniformly in
$[-1/\sqrt{p},1/\sqrt{p}]$ for input size $p$, the weight scale is
$\sigma_0/\sqrt{p}$, and the bias scale is $\sigma_0/\sqrt{q}$ for output size $q$. The noise
sample has to be allocated on the parameter device; otherwise a CUDA model fails when resetting
noise. The reference also samples online-network noise in the outer training loop before acting and
updating, while `learn` resamples the target-network noise just before building the bootstrap
distribution.

The hyperparameters now line up with the integrated object: learning starts after 80K frames rather
than DQN's 200K because prioritized replay makes early samples more useful; Adam uses learning rate
$6.25\times10^{-5}$ and epsilon $1.5\times10^{-4}$; target copies occur every 32K frames; the
priority exponent is $\omega=0.5$ and the importance-sampling exponent anneals from 0.4 to 1.0;
$n=3$ is the selected multi-step horizon; the categorical support has 51 atoms in $[-10,10]$; and
$\epsilon=0$ because exploration comes from the noisy weights.

The ablation evidence should be stated carefully. Prioritization and multi-step learning are the
largest contributors in the reported combination, distributional learning becomes important later in
training, and noisy layers help in median performance. Dueling and double Q-learning are still part
of the coherent agent, but their median ablation effects are smaller and mixed by game; for double
Q-learning the bounded categorical support can already push estimates downward when true clipped
returns exceed 10, which partly counteracts classic overestimation. So the final answer is not that
every component contributes equally. The answer is the exact integrated update: multi-step
categorical target, online-selected and target-evaluated bootstrap distribution, KL/cross-entropy
priority, per-atom dueling logits, and noisy fully connected layers with greedy action selection.
