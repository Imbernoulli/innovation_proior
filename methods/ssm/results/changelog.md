# Changelog — ssm

## 2026-08-18 — svfix(epistemic)
- **Removed a self-supplied observation introduced by the
  svfix(B_selfaccount) pass** (commit `ad9dc7dd0`, "Gu's Cognitive
  Revolution interview grounds 'why every structured SSM stayed LTI' in
  his own pre-Mamba failed attempt (HiPPO-era, months, intractable to
  train)"). A single-turn method unit is a proposal: the method's own
  experiments have not happened at that point in the frame, so
  reasoning.md must not have the narrator run an experiment and report
  its result — real numbers or not, and regardless of whether that
  experiment is framed as belonging to this proposal or to the
  narrator's own past. The prior pass grounded the sentence "This is the
  trade the whole field had been quietly avoiding — it is *why* every
  structured SSM stayed LTI" in an added autobiographical aside: "I spent
  months trying to get exactly this — a time-varying state matrix — to
  actually train. The idea wasn't wrong on paper; it was intractable in
  practice... too slow and too unstable to fit, so I shelved it... that
  door had already been tried and hadn't opened... the same room that
  beat me the first time." That is the narrator claiming to have run a
  training experiment (elsewhere in time, but still an experiment on the
  exact object under discussion) and stating its outcome — precisely the
  banned pattern, just moved to backstory instead of the current
  derivation.
- Rewrote the passage to keep the self-account grounding svfix added
  (the narrator's personal history with this exact problem, dated to
  right after the theoretical HiPPO work and before any S4 machinery
  existed) and the design reason for why the field stayed LTI, but
  replaced the claimed training outcome with the on-page structural fact
  already derived two sentences earlier in the same paragraph — no
  convolution, no FFT, no algorithm on hand to make the raw recurrence
  cheap — as the reason the door stayed shut. No claimed observation, no
  duration ("months"), no experiment-outcome language ("too slow and too
  unstable to fit") remains.
- Landing is unaffected: the paragraph's actual job is to motivate "the
  wall," and the two concrete problems with the recurrence (sequential
  dependency; B·L·D·N memory blow-up) are derived analytically in the
  very next paragraph, unchanged and untouched by this pass — the
  method's justification never depended on the removed anecdote.
  needs_traj = false.
- Scope: svfix(B_selfaccount) touched only this one paragraph in
  reasoning.md (`git show ad9dc7dd0 --stat`); answer.md and
  train_answer.md were not part of that commit, so nothing to check
  there for this pass.
