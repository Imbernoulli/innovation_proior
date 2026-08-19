# changelog — silu

## 2026-08-18 (epistemic correction pass)
Prior fix commit `e9794b63c` ("svfix(D_candidate): silu — OpenReview author self-account +
primary search table, decisive step now grounded") replaced the closed-form derivation of
`x·σ(βx)` from `ReLU = x·1(x>0)` with a grounded framing: the search itself returns two
near-tied finalists, `x·σ(βx)` and `max(x,σ(x))`, both search outputs rather than hand-derived —
a real and valuable correction. But it wrote the tie-break between them in the wrong voice for
a single-turn PROPOSAL: the narrator "looks at what happens when each is dropped into bigger
networks" and reports six concrete CIFAR-10/CIFAR-100 accuracy numbers across three
architectures (ResNet-164, Wide ResNet, DenseNet), observes that the ranking flips by dataset,
and concludes `x·σ(βx)` "holds up more consistently." That is the method's own benchmark result
stated as something already observed, in both `reasoning.md`'s body paragraph and the matching
clause of the end-to-end causal-chain summary. This frame does not allow that — the method's
own results belong only in a separate trajectory-observation turn, not the proposal.

Fixed by rewriting both passages to keep: the two candidates as genuine search outputs, the
discriminating-experiment DESIGN (same three deeper architectures, matched training budgets,
both CIFAR-10 and CIFAR-100 rather than just the search's own training dataset), the PREDICTION
(a split ranking by dataset would be the same inconsistent-gains problem the search was built to
escape, now resurfacing between the search's own two finalists), and the decision RULE
("whichever formula takes more of the six comparisons is the one that ships; algebra alone
can't make that call") — while removing the claimed observations (the six accuracy numbers, the
stated dataset-flip, and the declared winner). `x·σ(βx)` is carried forward explicitly for an
independent, non-empirical reason already in the text — its `β` knob makes it a one-parameter
family that `max(x,σ(x))` structurally cannot be — flagged as the candidate under active study
pending that grid, not as an experimentally confirmed winner. This unit now needs a
trajectory-observation turn to supply the actual six-comparison result. No changes to
`answer.md` or `train_answer.md` were needed (svfix diff touched only `reasoning.md`).
