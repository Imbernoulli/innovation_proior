# Cocrystal Coformer Selection: Solubility Under a Stability Floor

## Problem

Your API (active pharmaceutical ingredient) molecule has, for each of `K`
hydrogen-bond functional-group **types** `t = 0..K-1`, `Ad[t]` donor sites and
`Aa[t]` acceptor sites, with per-type donor/acceptor bond strengths
`donor_strength[t]`, `acceptor_strength[t]` (a bond formed between a type-`t`
donor and a type-`t` acceptor is worth `W[t] = donor_strength[t] *
acceptor_strength[t]`).

You are given a **regulatory-approved list** of `M` coformers (GRAS/approved
partner molecules). Coformer `i` has its own donor/acceptor site counts
`fd[i][t]`, `fa[i][t]` per type, a lattice-packing constant `lc[i]`, a
polarity index `p[i]`, and its own short list of **regulatory-approved
stoichiometry ratios** (former molecules per 1 API molecule) -- you may only
submit a `(coformer, ratio)` pair whose ratio is on that coformer's list.

**Hydrogen-bond synthon-match score.** At ratio `r`, coformer `i` contributes
```
h(i,r) = sum_t  W[t] * ( min(Ad[t], r*fa[i][t]) + min(Aa[t], r*fd[i][t]) )
```
(each API site can only be satisfied once, so matches saturate against `Ad`/`Aa`).

**Lattice energy** (stability) of the cocrystal: `L(i,r) = h(i,r) + lc[i]*r`.

**Feasibility (stability requirement).** The cocrystal only *forms* if
`L(i,r) >= L_min` (given in the input). Below the threshold, output is
rejected.

**Solubility improvement** (what you actually want to maximize):
```
dSol(i,r) = P_BONUS * p[i]  -  DECAY * (L(i,r) - L_min)
```
`P_BONUS`, `DECAY`, `L_min` are given per instance. A larger polarity `p[i]`
helps dissolution, flat regardless of stoichiometry; but *any* lattice energy
above the stability floor locks the crystal more tightly and hurts
dissolution, and that excess grows with `r` (through `lc[i]*r`). So the best
ratio for a fixed coformer is always the smallest regulatory-approved `r`
that still clears the stability floor -- more coformer than needed only adds
lattice-energy excess for zero extra solubility credit. A feasible cocrystal
with `dSol(i,r) <= 0` earns no credit (it forms, but does not help).

Maximizing `h(i,r)` alone (the "strongest hydrogen bond" heuristic common in
coformer-screening practice) tends to push `L` far past `L_min`, which the
`DECAY` penalty punishes -- it does **not** maximize `dSol`.

## Input (stdin)
```
K
donor_strength[0..K-1]
acceptor_strength[0..K-1]
Ad[0..K-1]
Aa[0..K-1]
P_BONUS DECAY L_min
M
fd[0][0..K-1] fa[0][0..K-1] lc[0] p[0] R ratio_1 ... ratio_R
...  (M such lines, one per coformer, index 0..M-1)
```

## Output (stdout)
Two whitespace-separated integers: `idx r` -- the chosen coformer index and
stoichiometry ratio.

## Feasibility (checker rejects with Ratio 0 on ANY violation)
- output is not exactly two parseable finite integers;
- `idx` not in `[0, M-1]`;
- `r` not in coformer `idx`'s approved ratio list;
- `L(idx,r) < L_min` (unstable, does not form);
- `dSol(idx,r) <= 0` (stable but does not improve solubility).

## Objective
Maximize `F = dSol(idx,r)` over all regulatory-approved, feasible pairs.

## Scoring
`B` = `dSol(0,1)` (the checker always evaluates coformer 0 at ratio 1, the
weakest-bonding "reference" entry that is always present and always feasible).
```
sc    = min(1000, 100*F/B)
Ratio = sc / 1000
```
Matching the reference gives `0.1`; ~10x its solubility gain caps at `1.0`.

## Example (K=2, illustrative shape only -- not to scale with real cases)
`donor_strength=[5,3]`, `acceptor_strength=[4,6]` -> `W=[20,18]`.
`Ad=[2,2]`, `Aa=[2,2]`. `P_BONUS=10, DECAY=1, L_min=76` (`=2*sum(W)=2*38`).
- Coformer 0 (reference): `fd=fa=[1,1]`, `lc=6`, `p=8`, ratios `[1]`.
  `h(0,1)=Sum W[t]*(1+1)=76`; `L=76+6=82`; excess `6`;
  `dSol=10*8 - 1*6 = 74 = B`.
- Coformer 1: `fd=fa=[3,3]` (saturates all API sites), `lc=3`, `p=2`, ratio `[1]`.
  `h(1,1)=Sum W[t]*(2+2)=152`; `L=155`; excess `79`; `dSol=10*2-79=-59<=0`
  -> rejected (stable, but no gain). The "maximize bond strength" heuristic
  lands here.
- Coformer 2: `fd=fa=[2,2]`, `lc=2`, `p=25`, ratio `[1]`.
  `h(2,1)=Sum W[t]*(2+2)=152` (already saturated); `L=154`; excess `78`;
  `dSol=10*25-78=172`. `sc=min(1000,100*172/74)=232` -> `Ratio=0.232`.

## Constraints
`3 <= K <= 5` (fixed at `4` per case), `8 <= M <= 30`, all counts/strengths
small positive integers. Time limit `5s`, memory `512m`.
