The whole ladder is about a single trade-off — earn reward while holding episodic cost under the budget $d = 25$ — and a trade-off cannot be read without an anchor at each end. The constraint-handling mechanism bolts onto a base learner, so before I design any mechanism I have to know what that learner does when safety is ignored entirely. That floor is not a throwaway; it is the *denominator* for everything that follows. If a later method lands cost at 20, I can only tell whether that is a triumph or a no-op by knowing where the unconstrained agent sat — was it 60, or was it already near 20? So the first thing I establish is the constraint-unaware control: the highest return obtainable when nothing is paid for cost, and the uncontrolled cost the environment produces when reward is optimized freely.

The method here is the *naive* fill — pure reward-maximizing PPO running inside the safe-RL harness with the safety apparatus deliberately disconnected. To see why this is genuinely the floor and not merely *a* baseline, look at the two slots the contract lets me edit. The first is the multiplier update inside `_update`; the second is the advantage combination inside `_compute_adv_surrogate`. The substrate is a constrained-PPO loop carrying *two* value functions — one trained on the reward signal, one on the cost signal — each with its own GAE pass, so every transition arrives with both a reward advantage $\text{adv}_r$ and a cost advantage $\text{adv}_c$. The loop is built to blend the two, but nothing forces it to at the start. The naive fill chooses the weakest possible blend: it leaves the scalar multiplier pinned at its initial value, $\lambda = 0.0$, and has the advantage hook return $\text{adv}_r$ untouched, discarding $\text{adv}_c$ completely.

The consequence is structural, not incidental. In `_update`, the measured mean episodic cost $J_c$ is read from the logger — but only to assert it is not NaN, a sanity check on the logging contract — and then `super()._update()` runs the ordinary PPO actor/critic step without ever moving $\lambda$ off zero. In `_compute_adv_surrogate`, the policy gradient PPO ascends is built from $\text{adv}_r$ alone. The cost critic still trains — the loop computes $\text{adv}_c$ on every transition regardless — but its output never reaches the policy. There is, *by construction*, no channel through which a safety violation can change the agent's behavior. That is what makes this the floor: it is not a weak attempt at safety, it is the complete absence of the mechanism, and so it reports the unconstrained extremes the mechanism will later have to bend.

What I expect this floor to do sets up the diagnosis the next step works against. PPO with $\text{adv} = \text{adv}_r$ is a competent reward optimizer, and on these Safety-Gymnasium navigation tasks the reward is reasonably dense — shaped signal for moving toward goals — so the agent should learn the task well and post the *highest* returns of anything on the ladder; nothing holds it back from the reward. The cost is the other half, and it should be badly out of bounds. The hazards in these environments sit exactly where the shortest, highest-reward paths run, so the reward gradient points straight through them. With no penalty on $\text{adv}_c$ the agent has no reason to detour, and the episodic cost should land far above the budget — many times $25$ where hazard density is high. The signature is sharp and asymmetric: strong on return, badly over budget on cost. I expect that asymmetry to be most extreme on the hazard-dense SafetyPointButton1-v0, less so on SafetyPointGoal1-v0 and SafetyCarGoal1-v0. Wherever the cost lands, it lands with nothing pulling it down, so it is the *worst-case* cost — the exact safety debt the first real mechanism has to pay down, and the amount of return it must surrender to do so is the price I will then be measuring.

```python
# EDITABLE region of custom_lag.py -- step 1: naive (no constraint handling)

# (no extra imports)

@registry.register
class CustomLag(PPO):
    """Naive baseline: pure reward-maximizing PPO, safety ignored."""

    def _init(self) -> None:
        super()._init()
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Naive: no multiplier update, stays at 0
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Naive: ignore cost advantage entirely, optimize reward only."""
        return adv_r
```
