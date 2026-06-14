**Problem.** A fixed hybrid post-training stack pairs each prompt with an on-policy GRPO rollout group
and a teacher demonstration `τ★`. The router decides how to use them. The floor is the degenerate rule
that throws the on-policy half away entirely — pure SFT applied to every question — because no clever
routing is worth building if it cannot beat that.

**Key idea (the floor).** For *every* question, drop all on-policy rollouts and replace them with one
off-policy SFT demonstration, ignoring `on_solve_num`. This collapses the hybrid stack to supervised
fine-tuning on `τ★`: token-averaged, prompt-masked NLL `L_SFT = −(1/|τ★|) Σ_t log π_θ(τ★_t | q,
τ★_{<t})`, scaled by `sft_loss_coef = 1.0`. Pure behavior cloning, run inside the harness.

**Why it is the floor.** SFT is the demonstration half of the unified gradient
(`∇J_μ = E_{τ∼π_θ}[r ∇log π_θ] + μ E_{τ∼π_β}[∇log π_θ]`); pure SFT keeps only that half for every prompt
and zeroes the reward half. It can bootstrap a capability the model never samples (its one strength), but
it gets no signal about the model's own good rollouts, cannot rank responses, trains no recovery from
the model's own mistakes, and applies the identical copy-the-teacher update even where the model already
solves the prompt — so it narrows the policy toward the demonstration distribution and overfits.

**Scaffold edit.** Replace the controller with the unconditional triple
`(rollout.n_verify, 0, 1)`: remove all eight on-policy rollouts, add no new ones, add one off-policy SFT
sample. No per-question state, no off-policy RL arm.

**What to watch.** AMC23 (small, demonstration-shaped) should look least bad; MATH-500 and AIME24 should
show pure imitation's narrowing and its inability to use the on-policy signal. That failure forces
*conditioning* the route on per-question competence — but first the opposite extreme (never switch) must
be measured.

```python
# EDITABLE region of mix_trainer.py — step 1: pure SFT (drop on-policy for every question)
    def select_on_off_ada_balance(self, on_solve_num: int):
        del on_solve_num
        return self.config.actor_rollout_ref.rollout.n_verify, 0, 1
```
