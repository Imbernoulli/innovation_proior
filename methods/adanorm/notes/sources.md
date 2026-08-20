# Sources — decisive-step sourcing check (svfix, W3_primary_plus_ancestors)

## Decisive step (per TRIAGE, independently re-identified)
reasoning.md, the "Before I trust the claim that this makes each block the
identity..." paragraph (the block immediately after the six-vector
`(shift,scale,gate)` code is introduced): the claim that the zero-initialized
**gate** — not `scale = 0` alone — is what makes a residual block the identity
function at init, because `scale = 0, shift = 0` still leaves `LN(x)` a live
nonzero vector that a randomly-initialized sublayer maps to a nonzero branch
output, whereas `gate = 0` kills the branch outright before the residual add.

## Quality-gate verdict: SOUND_AS_IS — no rewrite made

### (a) Genuinely derived on the page, not asserted
The passage is a worked, checkable hand computation, not a hindsight
restatement:
- Takes a concrete `x = [1, 2, 3, 4]`, computes `LN(x) = [-1.342, -0.447,
  0.447, 1.342]` (mean 2.5, var 1.25, std 1.118 — I recomputed this
  independently and it is exact to the stated precision) and notes its norm
  is `2.0`, i.e. nonzero.
- Applies `modulate(LN(x), 0, 0) = LN(x)` (identity of the `1+scale`
  parameterization at `scale=0`) — still nonzero.
- Pushes it through an illustrative random linear map to get a nonzero
  branch output (`[-0.201, -0.134, 0.0, 0.470]`, flagged "e.g." — an
  illustration, not a claimed-exact number) and shows `x + branch =
  [0.799, 1.866, 3.0, 4.470] != x` — arithmetic checks out exactly against
  the stated branch values.
- Contrasts with `gate = 0`: `x + gate·branch = x + 0 = [1,2,3,4]`, exactly
  the input.
- Concludes the gate must sit at the residual add (outside the sublayer),
  not inside it, because only a zero multiplying the *whole branch output*
  cleanly zeroes the contribution.
- Explicitly frames this as a self-correction ("the obvious-looking
  shortcut here is wrong and I nearly took it: surely `scale = 0` already
  does the job?") built on real algebra, not a hedge word — the failure
  (scale=0 leaves a live nonzero branch) is concrete and checkable, exactly
  what the quality gate calls a legitimate derivation.

### (b) Backed twice over — the trace's own computation, and the primary
The gate holds two ways at once:
1. It's the trace's own honest computation (above) — self-contained,
   correct arithmetic, no invented anecdote.
2. It is also *exactly* what the primary paper's own related-work framing
   states, independent of any hand-worked example. `src/main.tex` (the DiT
   paper's own LaTeX source, on disk) lines 159-161, the "adaLN-Zero block"
   paragraph:

   > "In addition to regressing $\gamma$ and $\beta$, we also regress
   > dimension-wise scaling parameters $\alpha$ that are applied
   > immediately prior to any residual connections within the DiT block.
   > We initialize the MLP to output the zero-vector for all $\alpha$;
   > this initializes the full DiT block as the identity function."

   This is the primary's own statement that regressing `gamma, beta`
   (reasoning.md's `scale, shift`) alone — the vanilla adaLN block described
   one paragraph earlier in the same source — is NOT what gives identity
   init; the separate zero-initialized `alpha` (reasoning.md's `gate`) is
   what does it. reasoning.md's hand-worked counterexample is a faithful,
   independently-derived re-proof of the exact distinction the primary
   paper's own text draws between the two block variants.

Also spot-checked and confirmed against sources already on disk:
- Goyal et al. zero-gamma claim: `refs/goyal_1hour.txt` lines 409-415, "For
  BN layers, the learnable scaling coefficient gamma is initialized to be 1,
  except for each residual block's last BN where gamma is initialized to be
  0. Setting gamma = 0 in the last BN of each residual block causes the
  forward/backward signal ini[tially]..." — matches reasoning.md exactly.
- ADM AdaGN formula: `refs/adm.txt` line 292, "AdaGN(h, y) = ys GroupNorm(h)
  + yb" — matches reasoning.md exactly.

No source needed grafting: the decisive step is already correctly derived
on the page and is corroborated (not contradicted) by the primary source
that is already on disk in `src/main.tex`. Bolting on a citation here would
be decorative, not load-bearing — the derivation does not depend on any
external material to hold.

## notes/explainer_dit_adaln_zero.md (already on disk, read in full)
Confirms `modulate`, the 6-vector split, and — most load-bearingly for a
*different* step — that `cond = t_emb + c_emb` is a SUM (one TDS explainer
wrongly says "concatenated"; the primary source text and canonical
`models.py` both say sum). This corroborates the cond-formation step, not
the gate-vs-scale decisive step audited here. Nothing in it bears on the
identity-init question beyond restating the already-correct trace.

## Search for a self-account (per TRIAGE: none expected)
- `grep -i "adaln\|peebles\|adanorm\|scalable diffusion" SELF_ACCOUNT_SOURCES.md`
  → no hits.
- `notes/synthesis.md` (pre-existing, this method) already logs: "No author
  Nobel/Turing-style self-account exists (recent CVPR paper). The paper's
  own related-work narrative... IS the derivation backbone."
- Consistent with TRIAGE's own framing: this is a self-derived hand-algebra
  step citing primary papers, not a self-account/obstacle trace — there is
  no missing struggle-account to retrieve here.

No rewrite made. No factual errors found in the audited passage or its
immediate context (Goyal / AdaGN citations both verified word-for-word
above).
