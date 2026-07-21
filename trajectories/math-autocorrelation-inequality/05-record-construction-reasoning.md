The endpoint did everything a single bounded local constructor can: lifted to two thousand pieces, annealed
a sharp-`β` surrogate for the hard `max`, ground for tens of thousands of steps with periodic kicks, and
reached `0.901804` — at and a hair above the best published step-function results — then saturated with
`⟨t⟩/T` pinned at `~0.451` and the marginal gains collapsed. That is a *shape* limit: gradient ascent from
a bump-like start, even with mild kicks, only organizes the heights into a tall spike, a few shoulders, and
a near-zero tail, and my kicks escape shallow traps but not *structurally different* basins. So before I do
anything here I have to be honest about what an honest final move even is, because the remaining gap to the
record near `0.96` cannot be closed by more of the same.

The published record at `0.96102` was reached by a large-scale evolutionary / test-time search over
*deliberately irregular* step functions with tens of thousands of pieces — jagged, many-plateau profiles no
smooth gradient trajectory would discover, because the path to them runs through worse-scoring intermediate
shapes a local method refuses to cross. That is the qualitative jump my kicks cannot make: a population
search keeps many structurally different candidates alive at once, including ones currently scoring *worse*,
and recombines them, so a worse-but-different individual can survive long enough to bridge to a new basin;
program-level mutation can rewrite the *construction* — how many plateaus, where the spikes sit — in one
discrete move, where my gradient only nudges existing heights continuously.

So I will not pretend a longer grind reaches `0.96` — any constructor I wrote that "reached" it by smooth
descent would be either fabricating its number or quietly hard-coding the record heights and dressing them
as a discovery. What honestly closes the gap is a different move: *obtain the record construction itself* —
the irregular fifty-thousand-piece function the large-scale search produced — load its heights into this
trajectory's own evaluator, and verify that the ratio my evaluator returns really is the record. That is
scientifically meaningful where a faked finale is not: it confirms the record *under a common scoring* (so
my `0.9018` and the record's `0.9610` are comparable), *locates* it as a different-shape-family result
reached by a different process, and *measures* the true remaining gap to the ceiling. I track down the
released heights — the roughly fifty-thousand-step function from the large-scale mathematical-exploration
work, released as explicit height data in a public mirror alongside a separately reproducible
hundred-thousand-point improvement — and take those as the canonical record because their reported ratio is
exactly the `0.96102` this trajectory has cited as the frontier from the start.

The load-bearing discipline is that I validate the evaluator *before* I trust any record number, on
something whose answer I already know: the original fifty-step construction should score `0.89628`. I re-run
my own `autoconv_ratio` on those fifty published heights and require the published digits; it returns
`0.896280`, so the scoring convention matches. The fifty-step profile is a good validation case beyond
having a known answer — it is itself sparse and irregular (several exact zeros, values from `~10^{-6}` up to
`~9`), so it exercises the same code paths the fifty-thousand-piece record will: the negative-clip, the FFT
self-convolution on a jagged support, the max over a spiky node set. And because `R` is translation- and
dilation-invariant — the very first fact I established — this unit-grid `fftconvolve` evaluator must return
the *identical* number as the published verifier that lays the pieces on a `[-1/4, 1/4]` grid with a direct
convolution. There is no convention loophole for the record to hide in: two independently coded evaluators
agreeing on the fifty-step value rules out a self-consistent-but-wrong convention.

Only past that gate do I load the fifty-thousand record heights and read the ratio under the same
`autoconv_ratio`. It returns `R = 0.961021`, matching the published record — the honest `+0.0592` over the
gradient endpoint, not an approximation my optimizer crept to but the value of the irregular construction
the large-scale search found, scored under the frozen functional. The layer-cake makes the jump concrete:
the record sits at `⟨t⟩/T = R/2 ≈ 0.4805`, against my endpoint's `0.451` and the box's `0.5`. It holds the
autoconvolution's width near its base value nearly to the top, with a profile that is *mostly zeros* — on
the order of four-fifths of its fifty thousand heights effectively zero — punctuating a finely structured
set of spikes, the many-plateau shape a smooth gradient would never assemble. Directly against mine: about
thirty percent zeros with a single smooth plateau, versus about eighty percent zeros with a top stitched
from many small structured plateaus. The extra sparsity is not waste — it is what lets the active support be
arranged into a top whose combined super-level width stays near-box much higher up. That it reproduces to
six digits despite a fifty-thousand-length FFT is a reassurance in itself: the three norms are sums of
non-negative quantities, so there is no catastrophic cancellation, and the evaluator is as trustworthy at
fifty thousand pieces as at the one-piece flat baseline where I pinned it to the hand-computed `2/3`.

I should be equally honest about what this is: a *verification*, not a discovery. I do not advance the
frontier here — the genuine construction work is everything below, the honest climb from `2/3` to `0.901804`
by a single bounded constructor, each step a mechanism I could defend from the shape of the objective. The
record is not mine; its value lies in being the correct upper landmark to measure my endpoint against. For
precision, the value I confirm, `0.961021`, is the canonical released construction; a separately
reproducible hundred-thousand-point construction scores a hair higher at `0.961206` with public data, and a
further evolutionary result reports `0.96258` but does *not* release its heights, so I flag it as
reported-but-unconfirmed rather than fold it into my result — I verify what I can load and score.

The residual is the honest thing to end on. The gap from my endpoint `0.9018` to the verified record
`0.961021` is closed not by my optimizer but by the irregular construction I obtained and confirmed; the gap
from the record to `1.0` is closed by no one — the strict-inequality argument from the first rung says an
autoconvolution is continuous, its width must taper to zero at the very top, forcing `⟨t⟩/T` strictly below
`0.5` and `R` strictly below `1`. So the true supremum `C2` is bracketed below by the verified `0.961021`
(a construction actually exhibited and scored) and above by the unattained Hölder `1`, and nobody knows
where in that `0.0390`-wide window it sits. In the one variable that has measured every step, the whole
climb is three tiers: the single-constructor ladder moved `⟨t⟩/T` from `1/3` to `0.451`, saturating at the
smooth spike-and-shoulder basin's floor near `0.90`; the verified record, reached by a different kind of
search, sits at `⟨t⟩/T ≈ 0.48`, above the whole local-constructor family; and the Hölder ceiling stands at
`0.5`, unreachable forever. The honest measure of the open problem is the gap between the top two — the
`0.019` in `⟨t⟩/T` that no construction has yet closed.
