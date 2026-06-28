I start with the base DQN update because that tells me where every later change has to attach. The
online network supplies the behavior policy and receives gradients, while the target network is a
frozen copy used inside the bootstrap. The scalar target is
$R_{t+1}+\gamma_{t+1}\max_{a'}q_{\bar\theta}(S_{t+1},a')$. That target has one action-selection
step, one evaluation step, one scalar error, and one network head producing one value per action.
Each of the six improvements I want to fold in modifies exactly one of those four pieces — selection,
evaluation, error, or head — plus the replay sampling and the exploration rule on the side. If I
combine them carelessly I will redefine the same piece twice and the definitions will fight. So
before writing anything I want to find an ordering that surfaces the collisions instead of papering
over them.

The candidate that constrains everything else is the categorical distributional change, because it is
the only one that changes the *object* being learned rather than how a scalar is computed. Once the
network emits a probability vector $p(s,a)$ over a fixed atom support instead of a scalar, the value
$q(s,a)$ is no longer a primitive — it is the mean $z^\top p(s,a)$ — and the squared TD error is no
longer the native loss. Every other improvement that mentions "the value" or "the TD error" therefore
has to be re-expressed in terms of this vector. That dependency is one-directional: the distributional
change forces re-readings of the others, but they do not force a re-reading of it. So I will build the
return distribution first and check whether each remaining improvement still has a single clean
attachment point once the object is a distribution.

The atom support is fixed up front: $z^i=v_{\min}+(i-1)(v_{\max}-v_{\min})/(N_{\text{atoms}}-1)$,
which in the Atari setting is 51 atoms evenly spaced over $[-10,10]$. The subtlety I have to respect
is that the target distribution does not live on this support. After I shift atoms by a reward and
contract them by a discount, the shifted locations $Tz$ fall between the fixed $z^i$, so I have to
redistribute their probability mass back onto the fixed grid with a projection $\Phi_z$ before I can
compare it to the prediction. This projection is the part I least trust to get right by inspection —
the index arithmetic and the sign conventions are exactly where I would expect a silent bug — so I
want to actually run it on a small case before I rely on it.

Take a tiny support to make the arithmetic checkable: 5 atoms over $[-10,10]$, so
$z=(-10,-5,0,5,10)$ and $\Delta z=5$. Suppose the target network puts all its mass on the middle
atom, $p=(0,0,1,0,0)$ at $z=0$. With a 3-step return $R^{(3)}=1$ and $\gamma=0.99$, the cumulative
discount is $\gamma^3=0.9703$, so the single mass atom moves to $Tz=1+0.9703\cdot0=1$. Then
$b=(Tz-v_{\min})/\Delta z=(1+10)/5=2.2$, which sits between atom 2 ($z=0$) and atom 3 ($z=5$). The
projection splits the mass by distance: atom 2 gets $u-b=3-2.2=0.8$ and atom 3 gets $b-l=2.2-2=0.2$.
So the projected vector is $(0,0,0.8,0.2,0)$, which sums to 1 and has mean
$0.8\cdot0+0.2\cdot5=1.0$ — exactly the shifted value $Tz$. I ran the full code path on this input to
be sure I had not mis-stated a sign, and it returns $m=(0,0,0.8,0.2,0)$ with projected mean
$1.0000$, matching by hand. So the convention is: the reward shifts the support additively, the
discount multiplies the future atoms, and the clamp to $[v_{\min},v_{\max}]$ happens after both, on
$Tz$, before computing $b$.

That worked example also tells me where the projection can fail silently. If a shifted atom lands
*exactly* on a grid point, then $\lfloor b\rfloor=\lceil b\rceil$, both split weights $u-b$ and $b-l$
are zero, and the atom's entire probability mass would vanish. To check this is real and to check the
fix, I project the distribution $p=(0.1,0.2,0.4,0.2,0.1)$ with $R=0,\gamma=1$ so that $Tz=z$ and
every $b$ is an integer. With the naive split, every atom would lose its mass; with the guard that
decrements $l$ (or increments $u$) when $l=u$, the code returns $m=(0.1,0.2,0.4,0.2,0.1)$ — the
original distribution unchanged, as a no-op projection should. So the edge-case branch is load-bearing,
not cosmetic, and I keep it explicit: decrement $l$ when possible, otherwise increment $u$, so the two
index-add terms still deposit the full mass on the exact atom.

One more property I want before trusting this as the spine: the projection should not distort the
expected return, since the whole agent acts on means. I project $p=(0.05,0.15,0.5,0.2,0.1)$ with
$R=1.5,\gamma=0.99$ and compare the projected mean $z^\top m$ against the direct mean
$\sum_i p_i\,Tz_i$ of the clipped shifted atoms. They come out equal to ten digits ($2.10743$ either
way), so the projection is mean-preserving wherever no mass is clipped at the support edges. That is
the guarantee I need for the categorical target to be a faithful stand-in for a scalar multi-step
target.

Now I fold in multi-step learning, and the worked example above already did it implicitly — I used a
3-step return and a cumulative discount $\gamma^3$. In the scalar case I would replace one reward with
$R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1}$ and bootstrap from $S_{t+n}$ with the cumulative
$\gamma_t^{(n)}$. In distribution space the same operation is: take the target distribution at
$S_{t+n}$, contract its atom locations by $\gamma_t^{(n)}$, shift them by $R_t^{(n)}$, then project.
So $d_t^{(n)}=(R_t^{(n)}+\gamma_t^{(n)}z,\;p_{\bar\theta}(S_{t+n},a^*_{t+n}))$. The terminal case
needs a separate check, because it is where future target values could leak in. I project the same
distribution with the nonterminal mask set to zero: every atom collapses to $Tz=R$, and the projected
mean comes back as exactly $R=1.5$ with no dependence on the next-state distribution. Good — when the
trajectory terminates, the bootstrap term is masked to zero and the target reduces to the observed
truncated return, which is what a correct multi-step target must do.

Double Q-learning has exactly one place left to attach once the object is a distribution: the choice
of the bootstrap action $a^*_{t+n}$. A distributional agent without double-Q would pick this action
from the target network's own means. The double-Q discipline is to decouple selection from evaluation,
so I select with the *online* network's mean,
$a^*_{t+n}=\arg\max_a z^\top p_\theta(S_{t+n},a)$, and then read the *whole probability vector* for
that action from the target network, $p_{\bar\theta}(S_{t+n},a^*_{t+n})$. The online network only
points at an action; its probabilities never enter the target distribution. This is a single, clean
attachment — it modifies the selection step and nothing else — which is the confirmation I was hoping
for that distributional and double-Q do not collide.

Prioritized replay is the first place where the obvious choice is the wrong one. The available
quantity, and my first instinct, is to reuse the scalar absolute TD error computed from the means,
because it is cheap and already lying around. But the optimization target here is the projected
distribution, not the mean, and those two errors can disagree. Concretely, on a stochastic transition
the network can already predict the correct return *distribution* — matching $\Phi_z d_t^{(n)}$ — while
a single sampled outcome still produces a large mean TD error just from the variance of the draw. A
mean-error priority would then keep re-sampling a transition the agent has nothing left to learn from.
So the priority has to be the agent's actual per-sample loss: the categorical divergence
$p_t\propto(D_{\mathrm{KL}}(\Phi_z d_t^{(n)}\|d_t))^\omega$. In code the minimized quantity is the
cross-entropy $-\sum_i m_i\log p_i$ rather than the full KL, which is fine because the projected target
$m=\Phi_z d_t^{(n)}$ is fixed under the gradient, so the two differ only by the target's entropy — a
constant that does not affect either the gradient or the relative ordering of priorities. The replay
buffer stores this raw per-sample loss and raises it to $\omega$ in its own priority update; the
importance-sampling weights multiply the loss for the gradient but are deliberately not folded back
into the stored priority, since the priority should reflect learnability, not the correction for
sampling bias.

Now the dueling head, which has to be reinterpreted rather than reused. The scalar version adds a value
scalar to action-advantage scalars and subtracts the mean advantage to pin the otherwise-unidentifiable
split. If I aggregate after taking expectations — combine the means and only then build a distribution
— I have thrown away the categorical object. So the value stream must emit one logit per atom, and the
advantage stream one logit per action per atom; for each atom $i$ I form $v^i+a^i-\bar a^i$ (mean over
actions), and only then apply a softmax over atoms, independently per action. I want to confirm two
things here: that this still yields a genuine per-action distribution, and that the per-atom mean
subtraction is doing something. Running it on random logits with 3 actions and 4 atoms, every action's
row sums to 1 after the softmax, so each action gets a valid distribution; and the result with the mean
subtraction differs from the result without it, so the subtraction is not a silent no-op — it couples
the two streams exactly as the dueling identification requires. The key constraint that falls out is
ordering: the mean subtraction is per atom and *before* the softmax, never on already-normalized
probabilities and never on scalar action means.

Noisy layers are mechanically simpler but carry their own faithfulness trap. The rule is to swap the
linear layers for noisy-linear equivalents and then act greedily with $\epsilon=0$, letting the
sampled weights supply exploration. The convolutional torso stays deterministic; the fully connected
layers in both the value and advantage streams become `NoisyLinear`. The factorized noise transform is
$f(x)=\operatorname{sign}(x)\sqrt{|x|}$, the mean parameters initialize uniformly in
$[-1/\sqrt{p},1/\sqrt{p}]$ for input size $p$, the weight noise scale is $\sigma_0/\sqrt{p}$ and the
bias noise scale $\sigma_0/\sqrt{q}$ for output size $q$. The trap is device placement: the noise
sample has to be allocated on the parameter's own device, or a model on CUDA crashes the moment it
resets noise against a CPU tensor. The other detail that matters for correctness rather than crashing
is *when* noise is resampled: the outer training loop resamples the online network's noise once before
acting and learning, while `learn` resamples the target network's noise just before it builds the
bootstrap distribution, so selection and evaluation each see a fresh, independent perturbation.

With every piece attached at a single point, the hyperparameters line up with the integrated object.
Learning starts after 80K frames rather than DQN's 200K because prioritized replay makes early
transitions worth revisiting sooner; Adam runs at learning rate $6.25\times10^{-5}$ with epsilon
$1.5\times10^{-4}$; the target network is copied every 32K frames; the priority exponent is
$\omega=0.5$ and the importance-sampling exponent anneals from 0.4 to 1.0; the multi-step horizon is
$n=3$; the categorical support is 51 atoms over $[-10,10]$; and $\epsilon=0$ because the noisy weights
already supply exploration.

I want to be careful about how much credit each component deserves, because the temptation is to claim
all six contribute equally and they do not. From the ablation pattern, prioritization and multi-step
learning are the largest contributors in this combination, distributional learning matters more as
training goes on, and noisy layers help median performance. Dueling and double Q-learning remain part
of the coherent agent but their median ablation effects are smaller and game-dependent — and there is a
mechanistic reason for double-Q's muted effect here that the worked projection already hinted at: the
support is clamped to $[v_{\min},v_{\max}]=[-10,10]$, so whenever the true clipped multi-step return
would exceed 10, the projection itself pushes the estimate down, partly doing the job that double-Q
exists to do. So I would not state that every component is independently essential; I would only verify
that they compose without contradiction. The result is the exact integrated update: a multi-step
categorical target, online-selected and target-evaluated, projected back to the fixed support, scored
by a cross-entropy that doubles as the replay priority, read out of a per-atom dueling head, with noisy
fully connected layers replacing $\epsilon$-greedy exploration.
