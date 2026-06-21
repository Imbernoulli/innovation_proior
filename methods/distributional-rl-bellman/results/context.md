## Research Question

Value-based reinforcement learning works with a scalar object: the expected discounted return from a state-action pair. Once the return is averaged, the value function satisfies Bellman's equation, and the corresponding evaluation and optimality operators contract in the sup norm. That contraction is why iterative backups converge to a well-defined limit.

The object being averaged is itself random. The cumulative return varies because rewards can be random, transitions can be random, and the policy itself can randomize. In a game, a state may lead either to survival and many later rewards or to a quick loss; in a navigation problem, a route may be usually safe but occasionally disastrous.

The question here is how to make the random return itself the object of dynamic programming: what a recursive backup acts on when it carries the whole return law, what its fixed point is, and in what sense repeated backups converge.

## Existing Machinery

For a fixed policy, the scalar value satisfies a recursive equation of the form

```text
Q(x,a) = E[R(x,a)] + gamma E[Q(X',A')]
```

and the associated operator is a gamma-contraction under the maximum norm. The optimality equation replaces the next-policy average by a maximum over next actions and retains a scalar fixed point. Standard temporal-difference learning, Sarsa, Q-learning, and deep Q-learning are all organized around this bootstrapping structure.

A fixed-point story for the scalar case rests on a complete space, a distance, a unique limiting object, and a convergence argument that holds despite the mixing introduced by random rewards and transitions.

## Prior Attempts

Work before this point already tracked more than the mean of a return. Some methods propagated second moments or variances; others estimated a parametric return density for risk-sensitive or robust decisions; still others used cumulative distribution functions or density models to choose safer policies. These lines show that the return's shape matters, especially when tail risk or variance changes the decision a practitioner wants. In these settings the distributional information serves a downstream goal — a Gaussian return model, a variance recursion, or a value-at-risk policy.

## Considerations

A discounted backup scales future returns toward zero and then shifts them by reward. A distance between return laws must register movement along the return axis: discounting transports probability mass along that axis. Likelihood and overlap-style comparisons are natural for fitting distributions and for measuring closeness of densities.

Control differs from evaluation. Scalar greedy choice hides the identity of the maximizing action because all optimal policies share the same optimal value. A richer object can remember which action produced it. If two actions have almost equal means but very different future laws, a small change in the mean can swap the selected action and the backed-up object along with it. Fixed-policy evaluation and greedy control are thus distinct cases.

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

The representation, backup, action rule, and loss must be consistent with the fixed-point object.
