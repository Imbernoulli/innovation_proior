# Coherent Pixel Typeface: Distinct Yet On-Style

You design a pixel typeface: **N** glyphs, each an `H x W` grid of pixels that are
**ink (1)** or **blank (0)**. A good typeface is two things at once — its letters must
be **easy to tell apart**, yet share a **coherent visual style**. These pull against
each other, and your score measures the tension directly.

## Input (stdin)
```
N H W          # N glyphs on an H x W grid
inkLo inkHi    # each glyph's ink count must lie in [inkLo, inkHi]
mh mw          # motif window size (rows x cols)
gnum gden      # style-penalty weight gamma = gnum / gden
```

## Output (stdout)
Print `N * H * W` integers, each `0` or `1`. The first `H*W` values are glyph 0 in
**row-major** order (row 0 left-to-right, then row 1, …, then row `H-1`); the next
`H*W` values are glyph 1, and so on. Whitespace and line breaks are ignored.

## Feasibility (a glyph is valid only if)
1. its ink count (number of 1s) is within `[inkLo, inkHi]`; and
2. its ink pixels form a **single 4-connected** component (up/down/left/right).

If **any** glyph is invalid, or the count of values is not exactly `N*H*W`, or any
token is not `0`/`1`, the score is `0`.

## Objective (maximize)
Two quantities are read off the whole font:

- **Distinctness** `D` = the **minimum pairwise Hamming distance** over all
  `C(N,2)` glyph pairs (the number of differing pixels for the closest-looking pair).
- **Vocabulary** `V` = the number of **distinct `mh x mw` ink windows** that appear
  anywhere in any glyph (slide the window over every in-bounds position of every
  glyph and collect the distinct binary patterns). A small `V` means the whole font
  is drawn from a few shared stroke motifs — that is style coherence.

Your raw quality is

```
Q = D^2 / (1 + gamma * V) ,   gamma = gnum / gden .
```

Bigger `D` helps (quadratically); bigger `V` hurts (a larger `gamma` makes style
coherence matter more than raw separation). The reported score is `Q` normalized
against a fixed weak reference font `B` that the judge builds itself:
`Ratio = min(1, Q / (10 * B))`, so the weak reference scores about `0.1` and there
is room to climb well above the sample solutions.

## Why it is not a one-liner
Chasing `D` alone (spray-painting maximally different glyphs) explodes `V`: every
glyph invents new local patterns, so `1 + gamma*V` swamps the gain. Chasing a single
motif alone keeps `V` tiny but collapses `D`. The trade-off is genuine, and for the
larger `gamma` cases a distance-only construction lands far below a design that fixes
a small motif basis first and then spreads the glyphs apart **within that basis**.

## Constraints
`2 <= N <= 26`, `H = 7`, `W = 5`, `2x3` motif windows, `12 <= inkLo <= inkHi <= 24`,
`0 < gamma < 1`. Time limit 5 s, memory 512 MB.

## Example (illustrative — small, not a scored case)
Say `N=3, H=7, W=5, inkLo=12, inkHi=24, mh=2, mw=3, gamma=0.2`. Three glyphs built
from full-row arms on a shared left spine, with arm-row sets `{0,1}`, `{0,2}`,
`{1,2}`. Every pair differs in exactly 2 rows, so `D = 2*(W-1) = 8`. Only a few
distinct `2x3` windows occur (full rows, spine-only rows, blanks), say `V = 7`.
Then `Q = 8^2 / (1 + 0.2*7) = 64 / 2.4 = 26.67`. A font of the same three glyphs
scrambled to maximize distance might reach `D = 12` but `V = 30`, giving
`Q = 144 / 7 = 20.57` — more distinct, yet lower quality once style is priced in.
