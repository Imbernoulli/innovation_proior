# changelog — mish

## 2026-08-18 (epistemic correction pass)
Prior fix commit `f75f38074` ("svfix(D_candidate): mish — arXiv 1908.08681v3 §2/3.1, decisive
gate-selection step now grounded in the documented candidate-family ablation + depth-stability
elimination instead of a closed-form derivation") replaced the closed-form pick of
`h(x)=tanh(softplus(x))` with a five-candidate family (`tanh(softplus(x))`, `tanh(x)·softplus(x)`,
`arctan(x)·softplus(x)`, `x·log(1+tanh(eˣ))`, `x·log(1+arctan(eˣ))`) sharing the same `0→1`
endpoint logic, but wrote the selection among them in the wrong voice for a single-turn
PROPOSAL: `reasoning.md` had the narrator *run* the discriminating ablation in-line ("I run
them... train it on CIFAR-10 with each candidate gate swapped in... three runs apiece, 50
epochs, RMSProp... come out essentially tied... trail both of them visibly... training goes
unstable and, in several runs, diverges outright... is the only one of the five that stays
both competitive on the shallow run and stable at depth"), i.e. reporting the method's own
experimental outcome as something already observed. This frame does not allow that — the
method's own results belong only in a separate trajectory-observation turn, not the proposal.

Fixed by rewriting the passage (and the matching clause in the end-to-end causal-chain summary)
to keep: the five-candidate design space, the discriminating-experiment DESIGN (small six-layer
CNN, CIFAR-10, three runs apiece, 50 epochs, RMSProp, matched architecture/budget/optimizer
across all five candidates, shallow screen first then escalation to deeper architectures for
whatever survives), and an explicit decision rule ("whichever gate is both competitive at the
shallow scale and still trains cleanly at depth is the one to ship") — while removing the
claimed observations (the tie, the trailing pair, the instability/divergence at depth, the
declared winner). `x·tanh(softplus(x))` is carried forward explicitly as the candidate the rest
of the document develops, flagged as provisional pending that ablation rather than as an
experimentally confirmed result, since paragraph 5 already establishes there is no a-priori
algebraic reason (short of running it) to prefer one candidate over the other four — "picking
one off the page is picking blind." This unit now needs a trajectory-observation turn to supply
the actual ablation result. No changes to answer.md or train_answer.md were needed (svfix diff
touched only reasoning.md).
