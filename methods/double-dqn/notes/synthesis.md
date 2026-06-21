# Synthesis - Double DQN

This file is a compact mirror of `notes/discovery_synthesis.md`; the full audit
trail and source list are in `notes/source_matrix.md`.

## Pain Point

DQN's predicted greedy-policy values can run above the actual discounted returns
obtained by the same policy. In the Atari experiments this happens even in
deterministic games and can coincide with falling score, so the issue is not
only value calibration.

## Mechanism

The DQN target is
$$
Y_t^{\mathrm{DQN}}=
R_{t+1}+\gamma Q(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t^-);\theta_t^-).
$$
Target-network estimates both select and evaluate the next greedy action. Since
the max operator prefers positive estimation errors, the target is upward biased
when estimates are inaccurate.

## Verified Math

For a tied state with $Q_*(s,a)=V_*(s)$, balanced errors, and mean squared error
$C>0$ over $m\ge2$ actions,
$$
\max_a Q_t(s,a)\ge V_*(s)+\sqrt{C/(m-1)}.
$$
The tight witness is
$$
\epsilon_a=\sqrt{C/(m-1)}\ (a=1,\dots,m-1),\qquad
\epsilon_m=-\sqrt{(m-1)C}.
$$
For i.i.d. uniform errors in $[-1,1]$,
$$
\mathbb E[\max_a\epsilon_a]=(m-1)/(m+1).
$$

The proof must contradict $\sum_a\epsilon_a^2=mC$, not an assumption
`< mC`. A valid zero-lower-bound witness for the double estimator, under the
paper's stated mean squared error convention, uses
$$
\epsilon_1=\sqrt{C(m-1)},\qquad
\epsilon_i=-\sqrt{C/(m-1)}\ (i>1),
$$
then sets the second estimator's selected-action value to $V_*(s)$.

## Method

Change only the target:
$$
Y_t^{\mathrm{DoubleDQN}}=
R_{t+1}+\gamma Q(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t);\theta_t^-).
$$
The online network selects the next action; the target network evaluates it.
Replay, epsilon-greedy acting, RMSProp/clipped TD error, and target-network
copying remain the DQN machinery. The tuned version increases the copy period
from 10,000 to 30,000 and changes exploration/final-layer bias, but those are
not part of the minimal algorithmic change.

## Code Grounding

DeepMind DQN Zoo's `double_q/agent.py` computes online next-state values
`q_t`, target next-state values `q_target_t`, and passes both to
`rlax.double_q_learning`. RLax defines the TD error as
`r_t + discount_t * q_t_value[q_t_selector.argmax()] - q_tm1[a_tm1]`.
The result deliverable mirrors this selector/evaluator split and sign.
