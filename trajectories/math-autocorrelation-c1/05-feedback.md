Measured result — `construct:autoevolver-record` (the published AutoEvolver `30000`-piece construction loaded from
`record_sequence.json` and scored through this ladder's evaluator). `R` from the FFT autoconvolution evaluator on the
`30000` heights; cross-checked against the `np.convolve` form (the two agree to `10` digits). No optimizer is run —
this rung reproduces the record and verifies it through the same harness.

| Stage | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| frontier-largeN (prev rung, single SLP constructor) | 600 | 1.5170399450 |
| AutoEvolver record (reproduced, scored here) | 30000 | 1.5028628969 |

Reference points: flat ceiling `2.0`, AlphaEvolve 600-step `1.5053`, TTT-Discover 30000-step `1.5028628983`,
AutoEvolver record 30000-step `1.5028628969`, provable floor `C1 ≥ 1.28`.

The published record chain: prior published upper bound `1.5098` → AlphaEvolve `600`-step `1.5053` (App. B.1) →
TTT-Discover `30000`-step `1.5028628983` (arXiv:2601.16175) → AutoEvolver `30000`-step `1.5028628969`
(Claude/Opus, "aspiration prompting"; github.com/tengxiaoliu/autoevolver). The provable floor is `C1 ≥ 1.28`
(Cloninger–Steinerberger 2017), so the still-open distance below the record is `1.5028628969 − 1.28`.

Notes: `−0.0142` below the previous rung, reaching `R = 1.5028628969` exactly — the record. The drop the single SLP
constructor could not buy comes entirely from the change of method: `30000` pieces (vs `600`) to carry the fine
irregular structure, and a long LLM-guided evolutionary search over the construction program (vs one local
trust-region engine). The `30000`-piece record solution is exactly the family the `600`-piece SLP was drifting
toward but could not express — a single enormous boundary spike (`~111×` the mean, at the right end, index `29999`)
over an interior `~38%` near-zero, with the autoconvolution flattened into a plateau of `~18000` near-tight nodes
within `10⁻⁴` of the peak. Scored through the identical FFT `R`, the harness returns the record value to `10`
digits, confirming both the evaluator and the construction. The remaining gap — from `1.5028628969` down to the
provable floor `1.28` — is the part of the first autocorrelation inequality still genuinely open after this record.
