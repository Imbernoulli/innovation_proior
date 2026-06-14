**Problem (from step 1).** Pure SFT applied to every question pays the imitation tax on prompts the
model already solves — it narrows the policy to one solution style and skips the model's own reasoning,
which is why MATH-500 sat at 0.244. SFT was the only thing that could bootstrap a stuck prompt; the
failure was applying it *unconditionally*. The route must condition on per-question competence.

**Key idea (HPT switch).** SFT and RL are two halves of the gradient of one objective
(`∇J_μ = E_{τ∼π_θ}[r ∇log π_θ] + μ E_{τ∼π_β}[∇log π_θ]`), so the router is a per-prompt choice of
estimator, not a blend of rival forces. The free per-question competence signal is the live solve count
`on_solve_num ∈ {0,…,8}`. Hard-switch on a gate: fire the *same* SFT triple `(n_verify, 0, 1)` only when
`on_solve_num ≤ switch_gate` (the all-wrong case where the GRPO group advantage degenerates to zero and
RL has no gradient), keep the on-policy GRPO group `(0, 0, 0)` when there is reward contrast, and route a
middle band `switch_gate < on_solve_num ≤ switch_gate_off` to the off-policy-RL arm (`off_add_num = −1`).

**Why these choices.** Routing on `on_solve_num` matches the bias-variance of each estimator to the
model's current competence per prompt (a fixed mixing weight cannot). The hard switch repairs a
discontinuity: it injects the teacher exactly where RL's signal dies and nowhere else, paying the
imitation tax only on prompts that earn it. Plain SFT (not off-policy RL) is the clean route for fully
stuck prompts, because off-policy RL needs a reference policy it does not have (forcing the biased
`π_ref ≡ 1` rejection-sampling assumption). The advantage code group-normalizes over on-policy samples
only, so injected samples never contaminate the RL measurement. The mixture self-adjusts as competence
rises.

**Scaffold edit.** Replace the controller with the `switch` logic: read `on_solve_num`, compare against
`switch_gate` and `switch_gate_off`, return the SFT / off-policy-RL / keep-GRPO triple accordingly.

**What to watch.** MATH-500 should lift off the floor (contrast prompts stay on RL); AMC23 should hold;
AIME24 is the risk — if nearly every prompt is all-wrong, the gate fires SFT on almost all of them and
the switch collapses back toward pure SFT.

```python
# EDITABLE region of mix_trainer.py — step 2: HPT per-question switch
    def select_on_off_ada_balance(self, on_solve_num: int):
        if self.config.trainer.unify_strategy == 'switch':
            on_add_num = 0
            if on_solve_num <= self.config.trainer.switch_gate:
                on_remove_num = self.config.actor_rollout_ref.rollout.n_verify
                off_add_num = 1
            elif on_solve_num <= self.config.trainer.switch_gate_off:
                on_remove_num = self.config.actor_rollout_ref.rollout.n_verify
                off_add_num = -1
            else:
                on_remove_num = 0
                off_add_num = 0

            return on_remove_num, on_add_num, off_add_num

        if self.config.trainer.unify_strategy == 'soft':
            on_remove_num = 0
            on_add_num = 0
            off_add_num = 1

            return on_remove_num, on_add_num, off_add_num
```
