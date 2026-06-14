**Problem (from step 3).** Pure GRPO won MATH-500 (0.419) by killing the imitation tax, but it is
structurally blind on the all-wrong AIME prompts (0.087): the group advantage `(R − mean)/std` is zero
when every rollout fails, so pure RL contributes no gradient and cannot bootstrap a capability the model
never samples. The switch fixed those prompts with SFT but lost MATH-500 to the imitation tax. Neither
rung gets both.

**Key idea (LUFFY, off-policy guidance).** Bootstrap the dead-gradient prompts through a *reward* channel,
not imitation. Inject the teacher trace as an off-policy **RL** member of the prompt's group and
standardize the advantage over the union `G_on ∪ G_off`: `Â_i = (R(τ_i) − mean(G_on ∪ G_off)) /
std(G_on ∪ G_off)`. On a stuck prompt (rollouts all 0, teacher 1) the union has spread, the teacher gets a
large positive advantage, and a gradient appears — bootstrapping *through the advantage*, not a copy loss.
When the model already solves the prompt, its own rollouts dominate the statistics and the signal stays
self-driven. The mix self-adjusts with per-question competence.

**Why it beats both prior rungs.** SFT (the switch) pulls up every teacher token uniformly → narrows and
taxes MATH-500. Pure GRPO abandons the stuck prompts. The union-group RL advantage weights the teacher by
its competence-dependent advantage, so it bootstraps the stuck prompts (fixing GRPO's AIME blindness)
without the imitation narrowing (avoiding the switch's MATH-500 tax).

**What the harness exposes vs. omits.** The editable surface is the trainer-side *router* only; the actor
(`offline_loss_type=sft`), the `grpo` advantage code, and the rollout modules are fixed. So the controller
can route stuck prompts to the off-policy-RL arm (`off_add_num = −1`, `whether_off=True`) so the teacher
enters the union advantage group — but LUFFY's full actor-side refinements are **omitted**: the `π_φ = 1`
unclipped off-policy ratio and the policy-shaping `f(x) = x/(x+γ)`, `γ = 0.1` (which amplifies
low-probability surprising teacher tokens by ≈`1/γ` and preserves entropy) live inside the actor's
off-policy surrogate, which this task fixes. The trajectory lands the *routing* realization, not the full
shaped objective.

**Scaffold edit.** Make the off-policy-RL arm the default for stuck prompts: when
`on_solve_num ≤ switch_gate`, return `(rollout.n_verify, 0, −1)` (drop the dead on-policy group, add the
teacher as off-policy RL); otherwise keep GRPO with `(0, 0, 0)`.

**Bar to clear (vs. GRPO's real numbers; finale, no feedback).** AIME24 should lift above GRPO's 0.087
(off-policy bootstrap of the dead-gradient prompts); MATH-500 should hold near 0.419 and not regress toward
the switch's 0.250 (no imitation tax, since stuck prompts go to RL not SFT); AMC23 should recover toward
the switch's 0.325. Caveat: without the actor-side shaping the AIME lift may be smaller than the full
method's; the claim is the direction, not a number.

```python
# EDITABLE region of mix_trainer.py — step 4 (finale): off-policy-RL guidance on stuck prompts
    def select_on_off_ada_balance(self, on_solve_num: int):
        if on_solve_num <= self.config.trainer.switch_gate:
            # all-wrong: GRPO advantage is dead here. Drop the on-policy group and add the teacher
            # trace as an off-policy RL sample (whether_off=True) so it enters the union advantage
            # group and bootstraps through the reward channel rather than as SFT imitation.
            on_remove_num = self.config.actor_rollout_ref.rollout.n_verify
            on_add_num = 0
            off_add_num = -1
        else:
            # reward contrast: keep the on-policy GRPO group untouched (the rung that won MATH-500).
            on_remove_num = 0
            on_add_num = 0
            off_add_num = 0

        return on_remove_num, on_add_num, off_add_num
```
