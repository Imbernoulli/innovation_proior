# Bell-Sounding: Worst-Case Crack Probes

## Problem
A cast bell is modeled as a resistor mesh built from `L` concentric rings of
`b` nodes each, plus one extra **core** node at the crown. Ring `l` (`l =
0..L-1`) has nodes `(l,0..b-1)` arranged in a circle. Edges: **radial**
`(l,i)-(l+1,i)` for `l=0..L-2` with conductance `g_r`; **circumferential**
`(l,i)-(l,i+1 mod b)` for every ring with conductance `g_c`; and **core**
edges `(L-1,i)-CORE` for every `i` with conductance `g_core`. Ring `0` is the
outer rim — the only place you may tap the bell.

The bell may hide a crack at any ONE of `m` given candidate sites
`(pos_k, layer_k)`, `1<=layer_k<=L-2` (strictly interior). A crack at
`(pos,layer)` multiplies the conductance of the 4 edges touching node
`(layer,pos)` — its two radial edges and its two circumferential edges — by
a given factor `alpha` (`0<alpha<1`); everything else is unchanged.

Tapping the bell means injecting an integer current pattern `p_0..p_{b-1}`
on the rim (`sum p_i = 0`, current conservation) and reading the resulting
rim potentials. Given `p`, Kirchhoff's law fixes the potential vector `v`
over all nodes up to one additive constant; fix it by `v[CORE]=0`. The
**response** `resp(p)` is `v` restricted to the rim. For a candidate crack
`k`, its **detection strength** against probe `p` is
`D(p,k) = sum_i (resp_k(p)_i - resp_pristine(p)_i)^2` — literally how much
the crack perturbs the boundary response to that current pattern.

You have a budget of `q` probes. Since you don't know which of the `m`
candidates (if any) is the real crack, your probes must jointly cover the
WORST case: your score is driven by the crack that is hardest to see using
your best probe for it.

## Input (stdin)
```
b L m q I_max
g_r g_c g_core
alpha
pos_1 layer_1
...
pos_m layer_m
```

## Output (stdout)
`q` lines, each `b` space-separated integers `p_0..p_{b-1}`.

## Feasibility
Each of the `q` lines must have exactly `b` integers, each with
`|p_i| <= I_max`, and `sum_i p_i = 0`. Any violation scores `0`.

## Objective (maximize)
`F = min over k=1..m of ( max over your q probes p of D(p,k) )`.

## Scoring
The checker builds its own reference `B`: for each of the first `q`
candidates (cycling if `q>m`), a tight dipole straddling that candidate's
rim position with electrodes at `pos-s` and `pos+s`,
`s = max(1, floor(b/5))`, magnitude `I_max` — plausible-looking but barely
pushes current past the rim. `score = min(1.0, F / (10*B))`.

## Constraints
`6<=b<=12`, `3<=L<=5`, `4<=m<=10`, `2<=q<=5`, `4<=I_max<=8`,
`0.15<=alpha<=0.45`. Time limit 5s, each `.in` well under 1 MB.

## Example (illustrative, smaller than the real test cases)
`b=6, L=3` (one interior ring, layer 1), `g_r=g_c=g_core=1`, `alpha=0.3`,
two candidates `(pos=0,layer=1)` and `(pos=3,layer=1)` — diametrically
opposite on the rim — `I_max=3`, `q=2`.

The checker's baseline straddles each candidate tightly:
`B`-probe 1 = `(0,-3,0,0,0,3)` (flanking position 0), `B`-probe 2 =
`(0,0,3,0,-3,0)` (flanking position 3). This gives `B ≈ 0.00873`.

Submit instead two SPREAD dipoles, each firing straight through one
candidate and out the diametrically opposite rim node:
probe 1 = `(3,0,0,-3,0,0)`, probe 2 = `(-3,0,0,3,0,0)`. Both push current
through the interior ring where the candidates sit, so both detect BOTH
candidates almost equally well: `D ≈ 0.376` for every (probe, candidate)
pair here, so `F ≈ 0.376`. Score `= min(1, 0.376/(10*0.00873)) = 1.0`
(capped) — spreading current through the bulk, rather than hugging the rim
near a suspect, is what actually reveals a crack one ring in. On the real
(larger, deeper, more numerous) test cases a *single* fixed axis like this
stops covering every candidate, and only a probe set that both focuses
current at the right depth for each candidate AND spreads across the whole
family keeps the worst case high.
