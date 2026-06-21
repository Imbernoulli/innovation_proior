## Research Question

Value-based reinforcement learning has a clean scalar object: the expected discounted return from a state-action pair. Once the return is averaged, the value function satisfies Bellman's equation, and the corresponding evaluation and optimality operators contract in the sup norm. That contraction is why iterative backups are meaningful rather than just a heuristic.

The object being averaged is richer. The cumulative return is random because rewards can be random, transitions can be random, and the policy itself can randomize. In a game, a state may lead either to survival and many later rewards or to a quick loss; in a navigation problem, a route may be usually safe but occasionally disastrous. A single mean can land between futures that are never actually experienced.

The open problem is whether one can keep the random return as the object of dynamic programming. A solution would need to say what the recursive backup acts on, what its fixed point means, and whether repeated backups converge in a mathematically defensible sense.

## Existing Machinery

For a fixed policy, the scalar value satisfies a recursive equation of the form

```text
Q(x,a) = E[R(x,a)] + gamma E[Q(X',A')]
```

and the associated operator is a gamma-contraction under the maximum norm. The optimality equation replaces the next-policy average by a maximum over next actions and retains a scalar fixed point. Standard temporal-difference learning, Sarsa, Q-learning, and deep Q-learning are all organized around this bootstrapping structure.

The technical burden is not merely to write a recursion. Many recursions can be written formally. The burden is to recover the fixed-point story: a complete space, a useful distance, a unique limiting object, and a convergence argument that survives the mixing introduced by rewards and transitions.

## Prior Attempts

Work before this point already tracked more than the mean of a return. Some methods propagated second moments or variances; others estimated a parametric return density for risk-sensitive or robust decisions; still others used cumulative distribution functions or density models to choose safer policies. These lines show that the return's shape matters, especially when tail risk or variance changes the decision a practitioner wants.

Their limitation is that the extra distributional information is usually subordinate to another goal. A Gaussian return model, a variance recursion, or a value-at-risk policy can be useful, but none by itself establishes a general dynamic-programming object with the same role as the scalar value function. The missing piece is a fixed-point theory for the whole return law, not only a few summaries or a risk criterion.

## Obstacles

The first obstacle is metric choice. A discounted backup scales future returns toward zero and then shifts them by reward. A distance between return laws must see movement along the return axis; otherwise two separated point masses can remain just as far apart after discounting as before. Likelihood and overlap-style comparisons are natural for fitting distributions, but they are not automatically compatible with discounted dynamic programming.

The second obstacle is control. Scalar greedy choice hides the identity of the maximizing action because all optimal policies share the same optimal value. A richer object can remember which action produced it. If two actions have almost equal means but very different future laws, a tiny change in the mean can swap the selected action and replace the backed-up object wholesale. Any theory must separate fixed-policy evaluation from greedy control rather than assuming the scalar proof transfers.

## Implementation Scaffold

A DQN-style agent already has the pieces that would need replacement: a network head, a target backup, an action-selection rule, and a loss. The scalar version emits one number per action and regresses toward a sampled temporal-difference target.

```python
class ReturnHead:
    def forward(self, state):
        # Current baseline: one scalar per action.
        # Open slot: what object should be emitted per action?
        raise NotImplementedError

def choose_next_action(next_prediction):
    # Current baseline: argmax over expected action values.
    raise NotImplementedError

def backup_target(reward, discount, next_prediction, action):
    # Current baseline: reward + discount * scalar next value.
    # Open slot: how should a random future return be transformed?
    raise NotImplementedError

def learning_loss(prediction, target):
    # Current baseline: squared error on scalars.
    # Open slot: what discrepancy is trainable from sampled transitions?
    raise NotImplementedError
```

The scientific question and the engineering question meet here: the representation, backup, action rule, and loss must be consistent with the fixed-point object, not just with a desire to display uncertainty.
