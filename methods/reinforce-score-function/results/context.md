## Scalar Reinforcement Is Not a Teaching Signal

A connectionist controller can be asked to learn from a single number returned by the world after it acts. That number may say that the outcome was good or bad, but it does not say which unit should have emitted which value, which action would have been better, or how the environment would have responded to a nearby alternative. The usual supervised-learning move of comparing an output with a target is unavailable.

## Random Exploration Is The Only Probe

The system still has to try actions. Earlier adaptive control work handles this by making action-producing elements stochastic: their weights bias a random choice, and reinforcement later strengthens or weakens traces of recent activity. This makes learning possible in tasks such as cart-pole control, where the dynamics are unknown and the feedback may be delayed until failure.

## Prediction Helps But Does Not Assign The Actor's Derivative

Temporal-difference prediction gives a way to turn delayed outcomes into ongoing evaluative signals. A critic can learn from changes between successive predictions and can supply denser internal reinforcement. That helps with temporal credit assignment. The action selector sampled an output, and no teacher supplied the derivative of reward with respect to that sampling probability.

## Local Reinforcement Rules Have The Right Shape But Need A General Test

Reward-modulated eligibility traces are attractive because each weight can keep a local trace and later multiply it by a scalar evaluation. Special cases show useful behavior, and a general theoretical account would say what objective is being followed, what expectation is being differentiated, and when the expected weight change really points uphill.

## Research Question

How can a stochastic unit or network compute a valid gradient estimate of expected reinforcement using only the scalar reward signal, the sampled output, and locally available quantities?

## Code framework

```python
import torch

def policy_gradient_loss(log_probs, rewards, baseline=0.0):
    """
    Policy-gradient loss for a stochastic controller.

    Args:
        log_probs: tensor of log probabilities of sampled actions.
        rewards:   tensor of observed returns or rewards.
        baseline:  scalar or tensor baseline (must be independent of the sample).

    Returns:
        Scalar loss; call .backward() to get the policy gradient update.
    """
    adjusted = rewards - baseline
    loss = -(log_probs * adjusted).mean()
    return loss
```
