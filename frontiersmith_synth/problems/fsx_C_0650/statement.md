# Setlist Against the Crowd: Manufacturing the Pre-Encore Valley

## Problem

You are booking a headline set. You have a pool of candidate songs, each with an
**energy** rating, a **duration**, and a binary **style vector** (which sonic tags it
carries). You must pick an ordered subset of songs — a play order — whose total
duration fits inside the time budget, to maximize how the crowd feels by the end of
the night.

The crowd has a single **excitement state** `E`, starting at 0. Playing song `i`
(energy `e_i` in (0,1], style vector `s_i`) updates it as:

```
sim_i    = dot(s_i, M) / K              # similarity of this song to recent memory
fatigue_i = gamma * sim_i * E
E        = clip( E + e_i*(1 - E) - fatigue_i , 0, 1 )
M        = decay*M + (1-decay)*s_i      # memory drifts toward whatever style just played
```

`M` (length-`K`, starts at all zeros) is a decayed memory of recently played styles.
`gamma`, `decay`, `alpha` are instance constants given in the input.

Two mechanisms are built into this update:
- **Saturating excitation**: the term `e_i*(1-E)` means a song's contribution shrinks
  as `E` approaches 1 — the same song is worth more when the crowd is *not* already
  maxed out.
- **Similarity fatigue**: repeating (or closely repeating) recent styles builds `sim`
  toward the memory, and `fatigue_i` grows with both that similarity *and* the current
  state — a stacked run of similar songs can pull `E` down hard.

After the whole chosen sequence is played, let `peak` = the maximum `E` reached at any
point, and `final` = `E` after the last song. The **peak-end score** is:

```
score = alpha * peak + (1 - alpha) * final
```

Note this rewards a high `final` just as much as a high `peak` — a song that only
matters early is worth less than one whose payoff lands at the very end.

## Input (stdin)

```
N T K
alpha_milli decay_milli gamma_milli
e_1 d_1 s_1_1 ... s_1_K
...
e_N d_N s_N_1 ... s_N_K
```
`N` songs (0-indexed 0..N-1), time budget `T`, style-vector length `K`. `alpha_milli`,
`decay_milli`, `gamma_milli` are integers in [0,1000] giving `alpha = alpha_milli/1000`
(and likewise for `decay`, `gamma`). Each song: `e_i` is an integer in [1,1000] giving
energy `e_i/1000`; `d_i` is a positive integer duration; `s_i_1..s_i_K` are bits (0/1).

## Output (stdout)

```
m
i_1 i_2 ... i_m
```
`m` = number of songs you play, followed by their indices **in play order** (0-indexed,
distinct, each in [0,N-1]).

## Feasibility

- All `m` indices distinct and within `[0, N-1]`.
- `sum(d_i for i in your order) <= T`.
Any violation, or malformed output, scores 0.

## Objective

Maximize `score = alpha*peak + (1-alpha)*final` as defined above, replayed exactly
(exact rational arithmetic) from your chosen order.

## Scoring

The checker replays your sequence to get your score `F`, and separately replays its
own unambitious reference construction (calmest songs first, using only a fraction of
the budget) to get `B`. It prints `Ratio = min(1, F / (10*B))`. Doing nothing
clever (matching the reference) scores ~0.1; genuinely better sequencing/selection
scores higher.

## Constraints

`1 <= N <= 70`, `1 <= K <= 8`, `1 <= T <= 400`, time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

Instance: `N=2 T=5 K=2`, `alpha=0.5 decay=0.5 gamma=0.5`, song 0 = `(e=0.800, d=3,
style=[1,0])`, song 1 = `(e=0.900, d=2, style=[0,1])`. Playing `[0, 1]`: step 0 gives
`E=0.800` (peak=0.800, M=[0.5,0]); step 1: `sim=0`, `fatigue=0`,
`E=0.800+0.900*0.200=0.980` (peak=0.980, final=0.980). `score =
0.5*0.980+0.5*0.980=0.980`. (This tiny example has no style overlap between the two
songs, so no fatigue bites — the real instances plant same-style clusters that do.)
