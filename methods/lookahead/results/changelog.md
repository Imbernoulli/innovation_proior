# changelog — lookahead

## 2026-08-18 (obs-fix: self-supplied observation)
`obs_scan_v3.jsonl` flagged a narrator-run experiment outcome in `reasoning.md`, in the
paragraph deciding what to do with the inner optimizer's state (momentum buffers, Adam's
moments) at interpolation time: "I run all three on CIFAR-10 with SGD-momentum as the inner
optimizer: maintain reaches 95.15% final validation accuracy, interpolate 95.16%, reset
94.91%. All three beat plain SGD outright, maintain and interpolate sit within each other's
error bars, and reset trails a bit more clearly..." — a real CIFAR-10 training comparison
reported as already observed, which a single-turn PROPOSAL is not entitled to (the method's
own experiments have not happened yet at output time).

Fixed by rewriting the passage to keep the three-way design space (maintain / interpolate /
reset the inner-optimizer state) and turn the claimed observation into a controlled-comparison
DESIGN — same architecture/schedule/step budget, SGD-momentum as the inner optimizer, three
runs differing only in the state policy, all three against an unwrapped SGD-momentum
baseline — plus explicit PREDICTIONS per hypothesis (all three should beat the unwrapped
baseline since none touch the actual interpolation mechanism; maintain and interpolate should
land close together since both preserve the buffer's direction; reset should trail since it
discards that direction at the window boundary) and a DECISION RULE (if maintain and
interpolate wash out, ship the cheaper one — maintain — and expose the other two as options;
if reset is competitive or interpolate clearly wins, that overturns the default instead).
Numbers (95.15% / 95.16% / 94.91%) removed. `answer.md` and `train_answer.md` already stated
only the design decision ("maintained by default; may alternatively be reset or interpolated")
with no result numbers or "our ablations show" framing, so neither needed edits.

Single ablation decision, not a multi-rung ladder — no trajectory-observation turn required.
