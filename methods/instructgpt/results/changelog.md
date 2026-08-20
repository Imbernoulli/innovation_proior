# instructgpt changelog

## 2026-08-19 — svfix(W3_ancestors_only): epistemic correction at the decisive step

Triage (W3_ancestors_only) found the decisive step (RL-vs-RM tradeoff, KL-to-SFT
leash, ptx alignment-tax fix) reconstructible from the primary paper alone — no
independent self-account/blog/OpenReview/thesis material exists (`refs/` holds
only the InstructGPT `neurips_2021.tex` source itself; confirmed by direct read,
not just trusting the hint). On reading `results/reasoning.md` myself to find the
decisive step, found a real defect the sourcing question doesn't cover: at that
exact step the narrator claimed to run experiments mid-reasoning and stated their
outcomes — "in my probes" (RM-size stability), "The sweep lands at a small value,
β ≈ 0.02", "So I test exactly that — sweep β up by about a hundredfold. The
benchmark regressions do *not* come back...", "sweeping it on the smaller model
tells me it needs to be large ... landing at γ ≈ 27.8". These are the method's own
ablation outcomes (matching the primary's own reported β=2.0/100× and γ≥20/27.8
sweeps, `src/neurips_2021.tex` around lines 1210, 1387, 1378, 1472) narrated as if
personally observed inside a single-turn proposal's chain of thought — a
self-supplied-observation defect (the paper itself has run these experiments; the
narrator in this frame has not).

Rewrote the four passages (RM size, β leash, β-sweep alignment-tax test, γ/ptx
sweep) to hypothesis → test design → decision rule, replacing the claimed
observations with the actual reasoning that makes the outcome checkable without
an experiment report: raising β can only pin the RL policy to $\pi^{\text{SFT}}$,
and $\pi^{\text{SFT}}$ is itself the pretrained model narrowed by fine-tuning on a
small demonstration set, so no amount of KL pressure toward it can hand back a
capability that narrowing itself removed — a structural argument, not a reported
sweep result. Landing hyperparameters (β=0.02, γ=27.8, mid-sized RM, 8× pretrain
ratio) are unchanged and kept as stated design choices (the same register as the
file's existing ε=0.2 PPO clip), not as narrated experimental findings. Propagated
the same fix to `results/train_answer.md`'s parallel passage ("My first instinct,
tighten the KL leash ... does *not* fix the regressions and tanks the reward. That
is informative...") for consistency; `results/answer.md` was already a bare
specification with no narrated-observation voice and was left untouched.

No new external source was added or needed — the decisive step's factual content
was already fully backed by the primary paper on disk; only the trace's narrative
voice around it was in violation. Lint clean (`tools/lint_inframe.py`); no new
"the paper/the authors/arXiv" leaks; `tools/obs_scan.py` no longer flags the β/γ
passages (the remaining hit in reasoning.md is the legitimate on-page
gauge-invariance hand computation, not a violation). Landing/method/code unchanged.
