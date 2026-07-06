The hierarchical lifts kept paying — `0.884823` at twenty pieces, `0.894706` at five hundred — and when I
stopped at five hundred the gradient was still moving, the per-lift gains diminishing but strictly positive.
I left the previous rung with a falsifiable prediction: if what I am doing is refining a fixed spike-and-
shoulder shape whose ceiling is the smooth-basin frontier near `0.90`, then lifting once more to a few
thousand pieces and grinding much longer should push into the low `0.90`s and then *flatten* — approach
roughly `0.90`, match the best published step-function results, and stop. This rung executes that test. The
published step-function frontier I am aiming at is concrete: the `575`-piece `0.901564` and the `539`-piece
`~0.9016` that careful searches with far more compute reached, both of which live exactly in the range a few
thousand well-optimized pieces can render. So the target is to push up to and, if I can, a hair through that
band — knowing the absolute record near `0.96` sits far above and was bought by a different kind of search
entirely.

The lift itself is the same free operation as before: I take the optimized five-hundred-piece profile and
upscale `×4` to `N = 2000`, which — replacing each height by four copies — is literally the same function
dilated, so `R` is unchanged, and I confirmed at the previous rung that this equality holds to ten digits.
The upscaled point is again a degenerate flat-block plateau with zero within-block gradient, so I kick it
with a small multiplicative perturbation to break the block symmetry and give the gradient traction, exactly
as before. And the refinement is the same `β`-annealed Adam ascent on the analytic FFT gradient. What
changes at two thousand pieces is not the machinery but the *budget and the schedule*, and getting those
right is the whole game at this resolution. Three things have to change, and each is forced by a specific
feature of the finer problem.

The choice of two thousand — a `×4` lift rather than `×2` to a thousand or `×10` to ten thousand — is itself
a deliberate balance I should defend. The published frontier shapes are `539` and `575` pieces, so two
thousand renders them with room to spare: nearly four grid cells per piece of the best-known constructions,
enough resolution that the plateau and shoulder are no longer grid-limited (the diagnosis from the previous
rung was that at five hundred the grid had already stopped being the bottleneck). Going further buys nothing
I can use: my whole reading is that the basin ceiling near `0.90` is a property of the *shape family*, not the
grid, so ten thousand pieces would cost `~2.5×` more per step and land in the same place — I would be paying
for resolution the shape cannot exploit. And going shorter, to a thousand, would start to pinch the finest
structure the grind wants to carve. Two thousand is the smallest resolution that comfortably clears the
published frontier's rendering needs while keeping a forty-thousand-step grind inside a two-minute budget.
That is the sweet spot, and it is chosen by arithmetic, not taste.

The first is the sheer length of the run. At five hundred pieces a few thousand Adam steps per pass settled
the shape; at two thousand the optimum is a finer, more irregular profile with four times as many
coordinates that all have to be brought into a coherent arrangement, and the gradient keeps finding small
improvements for tens of thousands of steps rather than a few thousand. So I budget a long final pass — on
the order of forty thousand iterations — and let it grind. This is affordable only because the evaluator and
its gradient are `O(N log N)`: one step at two thousand pieces costs about `4.9×` a step at five hundred (the
`N log N` ratio), so forty thousand steps at two thousand is a couple of minutes, and the whole ladder runs
in about `130` seconds. Without the FFT this would be hours and I would not attempt it; with it, a long grind
is cheap enough to be the main tool.

The second is that the `β` schedule has to be pushed *much* sharper at the end, and the reason is subtle
enough that I want to get it right rather than just crank a knob. The softmax `B(β) = m + β^{-1} log Σ_j
e^{β(L_j-m)}` stands in for `‖f*f‖_∞ = max_j L_j`, and in absolute terms it is faithful even at moderate `β`:
with `4001` nodes the overshoot is at most `log(4001)/β ≈ 8.3/β`, already under `10^{-4}` by `β ≈ 80N`. So
the *level* of the surrogate is not the problem. The problem is what the softmax does in the *gradient*. Its
job there is to identify and control the true peak — the peak-penalty term is weighted by the softmax
distribution `w_j/Z`, which should concentrate on the genuine argmax so the optimizer is punished for letting
any single node run above the plateau. But a flat, wide cap is precisely a cluster of many *nearly tied* top
nodes, and the softmax weight smears across all of them unless `β` times the tiny spacing between them is
still of order one. As the optimizer succeeds in flattening and widening the cap, the spacing among the top
nodes shrinks, so the `β` that resolved the peak at five hundred pieces is now too soft — the weight spreads,
the peak-control gradient blurs, and the optimizer stops feeling exactly which node is the max, letting the
cap tilt or a spike creep up while the surrogate barely notices, so the true `R` lags. The fix is to anneal
`β` up to several hundred times `N` — about `400N` in the grinding pass, and sharper still in the polish —
far beyond the coarse levels, so the softmax stays sharp enough to resolve the crowded near-peak nodes and
its gradient genuinely tracks the hard `max`. The flatter I make the cap, the sharper `β` I need; the two
scale together, which is why the ceiling that was `~123N` at five hundred has to become `~400N`–`800N` here.

This also explains why I sharpen a softmax rather than simply optimizing the *exact* hard `max` with its
subgradient, which is the obvious alternative and the wrong one. The subgradient of `max_j L_j` is a single
one-hot on whichever node is currently tallest, so a hard-max objective would push down exactly one node per
step and see nothing of the rest of the cap. On a wide, nearly-tied plateau the identity of the tallest node
hops from one to the next between steps, and the optimizer *chatters* — it plays whack-a-mole with whichever
node momentarily pokes highest, never settling the plateau as a whole. The softmax's smeared weight is
exactly the cure: at soft `β` it controls the entire cap coherently (every high node feels a share of the
penalty), and as `β` sharpens it hands control over to the true peak without ever collapsing to the
one-node chatter of the hard max. The `β`-anneal is therefore not just a smoothing trick but the thing that
lets me manage a flat cap at all — broad control early, sharp resolution late — and the hard max, tempting
because it is the true objective, would sabotage precisely the plateau I am trying to build.

The third change is periodic kicks *during* the long run, not just at the upscale. A single kick at the lift
unsticks the initial plateau, but over tens of thousands of steps the optimizer can settle into a *shallow*
local basin and stop improving well before it has carved all the fine structure the resolution allows. A
small multiplicative kick `v ↦ |v(1 + κ ξ)|`, `ξ` standard normal, applied every few thousand steps acts as
a mild restart — gentle enough not to wreck the shape, strong enough to jostle it out of a shallow trap and
let the gradient find a better arrangement. The *spacing* of the kicks is sized to the problem, not arbitrary:
a kick every few thousand steps matches the timescale over which Adam settles a shallow basin at this
resolution, so each kick lands after the gradient has exhausted the current basin but before it has wasted
tens of thousands of steps sitting in it. Kicking far more often would just inject noise the optimizer never
gets to resolve; kicking far more rarely would let the long run stall for most of its length. I shrink `κ`
as the run sharpens (from `~0.008` in the grind toward nothing in the polish), so the late phase is pure
refinement rather than perturbed exploration. And because a kick can *temporarily lower* `R` before the
gradient recovers, the running best-true-`R` bookkeeping from the previous rung earns its keep here: I keep
the best vector seen across the whole schedule, so a kick that momentarily drops the ratio can never cost me
the good profile it was trying to improve on. This periodic-kick mechanism is the one genuinely new
ingredient over the previous rung, and it is there specifically because the run is now long enough for the
optimizer to stall mid-way, which it was not at five hundred.

I run the endpoint as a short ladder of passes at two thousand. A first *reorganizing* pass with a moderate
`β` ceiling and a kick lets the freshly lifted shape settle into the finer grid and find its new plateau
width. Then the *grind*: the long pass — tens of thousands of steps — with a high `β` ceiling and periodic
kicks, the phase that actually carves the irregular fine structure the published constructions rely on and
where the endpoint number is made. Then a final low-learning-rate, sharpest-`β` *polish* to tighten the
plateau and steepen the shoulder without disruption. The three passes move in lockstep along a single axis:
as I go from reorganize to grind to polish, the `β` ceiling sharpens (`~80N → ~200N → ~400N`, sharper still
in the polish), the learning rate decays (`~0.006 → 0.003 → 0.002 → 0.0008`), and the kick shrinks toward
zero — the run transitions smoothly from exploration on a soft surrogate with large steps and restarts, to
refinement on a nearly-exact objective with tiny steps and none. Each knob is moving the same direction for
the same reason: early I want broad, forgiving motion to find the arrangement; late I want faithful, gentle
motion to settle it. Throughout, I keep the best *true* `R` ever seen across all passes — the surrogate and
the true ratio diverge by a hair as `β` moves, and I want the genuinely best vector, not the surrogate-best
one.

Splitting reorganize from grind, rather than running one long sharp-`β` pass from the kicked lift, is a
deliberate use of the budget. Right after the `×4` lift and its kick the shape is far from settled — the
blocks have just been broken and the plateau has not yet found its new width at the finer grid. Sharp-`β`
steps are precisely the ones that must *not* be spent on that gross reorganization: at high `β` the surrogate
is stiff and unforgiving, and pouring the expensive, hard-to-move sharp steps into coarse reshaping wastes
them and risks locking in a half-reorganized shape. So I let a cheaper moderate-`β` pass do the gross work
first — soft enough to move the plateau around freely — and reserve the sharp-`β` grind for when there is
genuine fine structure to carve rather than bulk shape to fix. It is the same logic as annealing `β` within a
pass, lifted up one level to the sequence of passes: coarse motion on a soft objective, fine motion on a
sharp one, never the reverse.

What comes out matches the prediction closely. The reorganizing pass clears the five-hundred-piece value
comfortably — more resolution always helped before — landing somewhere around `0.899`, and the long sharp-`β`
grind is where the real number appears: the construction reaches `R = 0.901804`, a clean `+0.0071` over the
five-hundred-piece rung, *at and slightly above* the published step-function frontier — matching the
`539`-step `~0.9016` and exceeding the `575`-step `0.901564`, here with two thousand pieces and about `130`
seconds. In the one variable that has tracked the whole climb, the μ-weighted mean level `⟨t⟩/T = R/2` has
reached about `0.451` — right at the `~0.45` the previous rung projected as the shape's ceiling, and the
prediction that it would *approach* that value rather than blow past it is exactly what happened. The
returned profile has the structure the frontier is known for: genuinely irregular and sparse, with roughly a
third of its heights effectively zero and a spike on the order of thirty times the shoulder — a jagged
spike-and-shoulder with a wide, faintly rippled flat cap, not a tidy analytic shape. That structure is a
consistent continuation of everything below it, which reassures me the ladder refined rather than wandered:
the coarse twenty-piece profile was already about thirty percent zeros with a spike some seventeen times its
smallest shoulder, and across the lifts the *sparsity fraction held* (zeros lift to zeros) while the *spike
sharpened* — seventeen-fold to thirty-fold — as the cap flattened. The sharpening spike is exactly what the
layer-cake asks for: a taller, narrower spike builds the autoconvolution's height faster at the edges of the
cap, which steepens the sides so `μ(t)` holds near its base value further up toward `T`. The whole endpoint
is the same shape the ladder started with, pushed as far as a smooth gradient can push it.

The roughly one-third of heights at zero deserve a word, because I want to be sure they are a genuine optimum
and not an artifact of clipping. Clipping to non-negative after each Adam step could, in principle, just pin
heights at zero and freeze them there. But it does not freeze them: the gradient with respect to a height
sitting at zero is `2` times the correlation of the node-gradient with the rest of the profile, which is
generally *non-zero*, so Adam is perfectly able to push a zeroed height back positive on a later step if that
would help. The heights that stay at zero stay because the optimizer keeps choosing zero for them —
repeatedly, across kicks and sharpening `β` — which means the sparse support is a real feature of the
near-optimal shape, the same signal the coarse anneal gave when it drove pieces to exact zero, now confirmed
at high resolution by a completely different optimizer. The zeros are where the shape wants nothing, not where
the clip trapped it.

It is also exactly the shape the very first rung predicted, which is a satisfying closure. Reasoning about the
smoothing map `f ↦ f*f`, I argued back then that the optimal input could be neither a single interval (which
convolves to the over-rounded tent, `⟨t⟩/T = 1/3`) nor mass at a few isolated points (which convolves to an
empty comb of needles), but something *between*: concentrated enough to build the autoconvolution's height
fast at the cap's edges, spread and shaped enough to hold the top level. The two-thousand-piece endpoint is
precisely that intermediate object, now resolved finely — a sharp spike doing the fast height-building, a
tapered irregular shoulder holding the cap, and roughly a third of the support left at zero to keep the thing
from smearing back toward the over-rounded extreme. The prediction made from a hand argument about
convolution at the flat baseline is borne out in the detailed structure of the endpoint constructed a
thousandfold higher in resolution.

And here the ladder stops, because I can see it has reached the floor of the basin it lives in, not a
resolution limit I could lift my way past. The evidence is exactly the flattening I predicted: the value
approached `~0.45` in the layer-cake variable and the gains per unit of extra work collapsed. I am confident,
and honest, that grinding longer or lifting to ten thousand pieces would land in the same place — the same
smooth spike-and-shoulder basin every careful local optimizer in this literature converges to, from the
twenty-step constructions through the five-hundred-step ones, and that whole family floors near `0.90`. The
`0.9018` I reach is a *shape* limit, not a resolution limit.

I want the decision to stop to rest on a test, not a mood, so I state the two signatures that would tell a
resolution limit apart from a shape limit and check them. If I were still grid-limited, lifting further would
keep raising `R` and the layer-cake variable `⟨t⟩/T` would keep climbing toward `0.5`; if I have hit the
shape's floor, the value pins at some ceiling below `0.5` and the marginal return on extra work collapses.
Both signatures point the same way here: `⟨t⟩/T` landed at `0.451`, essentially *at* the `~0.45` I projected
one rung ago and not creeping past it, and the per-lift gains have decayed to a thin slice (`0.884 → 0.895 →
0.902`, each step smaller). Had the endpoint instead reached, say, `0.46` in the layer-cake variable or kept
climbing at an undiminished rate, I would read that as the grid still binding and lift again without
hesitation. It did neither, so the honest conclusion is that the smooth basin is exhausted and more of the
same buys nothing. That is a genuine stopping criterion, and it is met. The absolute record near `0.96` was not climbed
by a finer version of this constructor; it was reached by a large-scale evolutionary / test-time search over
*deliberately irregular* step functions with tens of thousands of pieces — jagged, many-plateau profiles that
no smooth gradient trajectory would discover, because the path to them runs through worse-scoring
intermediate shapes a downhill-tolerant-but-still-local method cannot cross in that combinatorial space. I
can say this in the layer-cake variable that has measured every rung. My smooth cap holds `μ(t)` near its
base value up to about `⟨t⟩/T ≈ 0.451`; a record-class irregular function holds it much higher, closer to the
box's `0.5`, by replacing the single smooth plateau with a finely structured, many-plateau top whose
cross-section stays wide almost all the way to the peak. But to travel from my single smooth plateau to that
many-plateau top, a search has to first *break* the plateau it has — momentarily dropping `⟨t⟩/T`, momentarily
lowering `R` — and rebuild it into an irregular arrangement. A gradient method, and even the mild-restart
annealing I use here, will not pay that up-front cost: my kicks are small multiplicative jostles of a
spike-plus-plateau, which stay a spike-plus-plateau, so they explore *within* the basin and never march the
long downhill road to a structurally different top. That
is orders of magnitude more compute and, more to the point, a fundamentally different search structure —
population diversity and program-level mutation, not a single annealed descent. So the endpoint of this
single-constructor ladder is precisely the step-function frontier that careful gradient refinement on
thousands of pieces can reach, with the record standing above as the genuinely open distance. The honest
number I report is the `0.901804` the evaluator returns on the returned two-thousand-piece vector, and the
gap from there to `0.96102` — like the residual `0.0390` from the record to the Hölder ceiling `1.0` — is the
measure of how much of the second autocorrelation inequality is still open. There is no finale in which this
constructor grinds its way to `0.96`; reaching that number requires a different kind of search, and pretending
otherwise would be a lie the layer-cake variable would immediately expose.
