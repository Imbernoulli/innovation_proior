# Changelog

- 2026-08-18 `methods/rpo/results/reasoning.md` "How big is `alpha`?" paragraph: replaced the speculative
  closing claim ("I *expect* a single default to carry across environments... an empirical bet I would
  only believe after running the suite... I would expect even this default to be more jitter than
  needed, and a much smaller `alpha` to recover near-PPO behavior there") with the actual documented
  outcome of running the wider suite: `alpha=0.5` catastrophically fails (not merely underperforms) on
  four specific environments — Ant, InvertedDoublePendulum, Reacher, Pusher — with real numbers (e.g.
  InvertedDoublePendulum 5490→303, Ant flips sign 1832→-10), and recovers to at-or-above PPO on all four
  once `alpha` is dropped to `0.01` there. Source: the first author's (`masud99r`) own CleanRL fork,
  commit history + `docs/rl-algorithms/rpo.md`; see `notes/sources.md`.
- 2026-08-18 `methods/rpo/results/answer.md` hyperparameters bullet: tightened "a few environments do
  better with a much smaller `alpha` (near-PPO behavior)" to state the real severity (collapse well
  below PPO, not just underperformance) and name the failure mode, for consistency with the
  reasoning.md fix above. No change to the landing (`rpo_alpha=0.5` default, algorithm, code unchanged).
- 2026-08-18 (epistemic fix) The two entries above put the method's *own* results — the four-env
  failure with real numbers, and the `alpha=0.01` recovery — into the narrator's mouth inside
  reasoning.md and answer.md. This is a single-turn proposal: at this point in the frame the suite
  has not been run yet, so the narrator cannot report having run it, real numbers or not (that
  observation belongs only in a trajectory turn). Reworked both passages back to hypothesis +
  discriminating-experiment design + per-family prediction + decision rule ("if a task collapses
  under `alpha=0.5`, re-run at `alpha=0.01` and check whether it recovers to at-or-above PPO"),
  dropping the claimed sweep run, the specific outcome numbers, and "which tells me" hindsight
  framing. Kept: the algebra (gradient-noise-scale limits), the real reward-structure distinction
  between narrow-band tasks (Reacher/Pusher/InvertedDoublePendulum/Ant) and locomotion tasks, and
  the falsifiable prediction + decision rule. `rpo_alpha=0.5` landing, algorithm, and code unchanged.
  This unit now needs a trajectory-conversion pass to carry the removed observation as a proper
  observation turn.
