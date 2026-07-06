FastTD3 is the strongest of the three, and the reason it wins tells me exactly where its ceiling
is. It beat FastSAC by dropping the entropy tax and getting exploration for free from 128 parallel
environments, and it beat PPO by reusing experience while exploiting the value function greedily
with a deterministic actor. Both wins are about *the algorithm*: the losses, the targets, the
exploration. What FastTD3 deliberately did *not* invest in is the network — it uses a plain
descending ReLU MLP with no normalization, on the explicit bet that data diversity (the fast-filling
buffer plus the 32,768 batch) tames the deadly triad so architecture is unnecessary. That bet is
the right call for being *fast and simple*, but it leaves a specific opening that I named at the
close of the last step: a plain MLP critic does not reliably convert extra capacity into a clean
value fit on the hardest-to-fit value surfaces. On stand the value function is gentle and the plain
critic is plenty; on walk and especially run the return surface is sharper, more contact-driven, and
longer-horizon, and a bare MLP critic chasing a moving bootstrapped target there is exactly the
regime where unregularized capacity overfits the transient target and the triad bites. FastTD3's
answer to instability was "more diverse data, smaller-than-tempting network." The complementary
answer — the one FastTD3 left on the table — is "let the network scale, by giving it the bias that
makes scaling safe." That is the move of this finale: keep the entire FastTD3 backbone untouched and
change only the function class of the actor and critic, importing the simplicity-bias architecture
that makes a larger network help instead of hurt.

Let me motivate why the architecture is the right lever and not just another knob. In supervised
vision and language, scaling up parameters reliably improves results, and the reason is not that the
networks are small — it is that modern architectures carry an implicit *simplicity bias* from
standard components (normalization, residual paths, careful init) that steer optimization toward
simple, generalizable functions even when heavily over-parameterized. Deep RL has not inherited
this: the standard actor-critic is a bare MLP, and widening it usually regresses because the value
target moves (it is bootstrapped through the critic's own next-state value) and the data
distribution drifts (the policy is changing), so raw capacity fits the transient target too hard and
feeds the error back. FastTD3's plain MLP is precisely this bare network. The hypothesis I act on is
that the missing thing is not RL-specific — it is the simplicity-inducing components — so importing
exactly those, and nothing algorithm-specific, should let FastTD3's critic finally scale.

Before I settle on the residual encoder, I should walk the architecture design space, because there
are three ways to spend capacity on the critic and only one of them is the bet I want to make. The
first is the obvious one FastTD3 already implicitly rejected: just widen the plain MLP — take the
`1024→512→256` critic to `2048→1024→512` and hope the extra units help. This is the option whose
failure motivates the whole rung: with no identity path and no normalization, the added width is
added capacity to overfit the moving bootstrapped target, and the standard RL result is that it
regresses rather than helps. Eliminated on exactly the mechanism I am trying to fix. The second is to
add LayerNorm to the plain MLP — essentially FastSAC's normalized body, which I already have evidence
stabilizes an off-policy critic. That is a real improvement over the bare MLP and would control
feature scale, but it does not create an *identity pathway*: a stack of `Linear→LayerNorm→activation`
layers still has to learn the near-linear map from scratch through every layer, so the simplest
function is not free, it is something the optimizer must assemble, and deep normalized-but-non-residual
stacks still degrade with depth. Normalization treats the symptom (scale drift) without granting the
structural bias (simple-by-default) that lets depth *help*. The third option is the residual
inverted-bottleneck block, which adds the identity pathway on top of the normalization, and that is the
one that makes the simplest function free and any nonlinearity an opt-in correction. So the elimination
is principled: widening bare MLP fails on overfitting, LayerNorm-only fixes scale but not the
depth-degradation, and only residual-plus-norm-plus-careful-init grants the simplicity bias that makes
scaling the critic safe. This is a clean, falsifiable claim about the architecture, and it fits the
edit surface perfectly: the task lets me redesign the `Actor` and `Critic` networks while leaving
`update_critic`, `update_actor`, the clipped double-Q logic, the distributional projection, the
per-env exploration, and the whole fixed loop exactly as FastTD3 has them.

The encoder I install, applied identically to actor and critic, has three load-bearing ingredients,
and I will reason each against FastTD3's specifics. First, observation normalization with running
statistics. The substrate *already* provides this — `EmpiricalNormalization` runs in the fixed loop
and standardizes actor obs, critic obs, and next obs online before every update — so this ingredient
is present in the FastTD3 baseline too, and I do not need to duplicate it inside the network. That is
worth being precise about: the canonical form of this architecture pairs the residual body with its
own running-stat input normalization (RSNorm), but on this scaffold the running-stat normalization is
a fixed-loop service, so my edit is the *body*, and the input normalization is inherited. Second, the
residual feedforward block — this is the ingredient FastTD3's plain MLP lacks and the one that does
the real work. Instead of a stack of fresh affine-plus-ReLU layers with no direct input-to-output
path, each block computes a correction and *adds the input back*: `out = x + F(x)`. The addition
creates a linear identity pathway, so the simplest function a block can represent — the identity — is
free, and any nonlinearity is an additive opt-in correction. That is simplicity bias made
structural: the easy near-linear map is the default, complex behavior is something the optimizer must
actively choose. For a bootstrapped value function chasing a moving target, defaulting to simple is
exactly the regularization that stops the overfitting FastTD3's bare MLP was exposed to on the hard
tasks. The block body is the standard inverted bottleneck: pre-LayerNorm the input, a linear map up
to four times the width, a ReLU, a linear map back down, then the residual add. The *pre*-LN keeps
the residual stream clean — the identity branch carries the un-normalized signal forward while each
correction is computed from a normalized version — and the 4× expansion is where the parameters that
*scale* live, so widening the critic grows the bottleneck and the capacity rides along. Third, a
final LayerNorm after the last block, before the value/policy head: the residual stream accumulates
(it is the input plus every block's correction), so its magnitude grows with depth, and the head
wants a well-scaled input — one LayerNorm right before the head standardizes the stream regardless of
block count.

Initialization is the quiet piece that makes the deep stack train from step one, and I set it
deliberately rather than by default. The embedding linear that lifts the input into the residual
stream gets orthogonal init at unit gain — a clean, well-conditioned start that neither inflates nor
shrinks the representation. The two linears inside each block get He (Kaiming) normal init, the
variance-preserving choice for ReLU, so the block's correction starts at a sane magnitude and neither
vanishes nor explodes as blocks stack. The actor's final tanh head keeps FastTD3's small-init
convention (orthogonal, tiny gain `0.01`) so the deterministic policy starts near zero action — that
head behavior is part of the backbone and I do not disturb it.

I want to verify that this initialization actually delivers the "starts near the identity" property I
am counting on, because if it does not, the simplicity-bias argument is decorative. Consider a block
at initialization. The input `x` passes through pre-LN (which standardizes it), then `fc_up` and
`fc_down` with He-normal weights and zero bias. He init sets the weight variance to preserve
activation scale through the ReLU, so `F(x) = fc_down(ReLU(fc_up(LN(x))))` produces a correction whose
magnitude is on the order of the standardized signal — order one — added onto `x`. Crucially the block
output is `x + F(x)`, so whatever `x` was, the block passes it through *plus* a bounded, sane-scale
perturbation, rather than *replacing* it with a freshly-computed map. Stack `N` such blocks and the
encoder at init is the embedding followed by the input plus a sum of `N` bounded corrections — a
near-linear, well-conditioned function, not a random deep tangle. That is the concrete meaning of "the
identity is free": at step zero the network is already close to a benign linear map from embedded obs
to features, and training moves it away from that only where the value target rewards the move. Drop the
residual add and the same `N`-layer stack at init is a composition of `N` random maps whose output has
lost the input — the very thing that makes deep bare stacks hard to train. The verification confirms
the combination residual + pre-LN + He init is what buys the property; any one alone does not.

Now the sizing, which is the whole reason to do this: I want capacity to *convert*. The value
function is the harder object — it has to fit a sharp, bootstrapped, contact-driven return surface —
while the policy is smoother, so I make the critic the larger of the two. The canonical configuration
is one residual block at hidden width 128 for the actor and two residual blocks at hidden width 512
for the critic, and that asymmetry is exactly the bet: the critic gets the extra capacity, the
residual/normalized structure makes that capacity safe, and the actor stays lean. Let me count the
critic to see that this is genuine capacity and comparable in scale to what I am replacing. Each block
at width 512 with a 4× bottleneck is `fc_up: 512→2048` and `fc_down: 2048→512`, i.e. `2 × (512·2048) ≈
2.1M` weights per block, so two blocks are about `4.2M`, plus the embedding `n_in·512`, the final
LayerNorm, and the atom head `512·101 ≈ 52k`. Against FastTD3's critic — a `1024→512→256` plain MLP,
about `n_in·1024 + 655k` weights, roughly two affine layers of real depth with no skip — the SimBa
critic is a genuinely deeper, normalized, residual network of comparable parameter scale but with the
simplicity bias that lets the depth help rather than hurt. The point is not that it has *more*
parameters (it is in the same ballpark); it is that its parameters are arranged so that adding them, or
adding a block, moves the function toward a better fit instead of toward overfitting the moving target.
The optimizer pairs with it: AdamW with a modest weight decay is itself a simplicity-bias regularizer
(it pulls weights toward small norm, toward simpler functions) and composes with the residual
structure. The substrate already uses AdamW with weight decay 0.1 and cosine annealing, so I keep that
optimizer family; the canonical SimBa decay is on the order of one part in a hundred, in the same
regime, and I leave the substrate's value rather than introduce a new knob, because the architectural
change is the variable under test.

The asymmetry deserves its own justification rather than being taken as a recipe, because "make the
critic bigger" is only right if the two networks are fitting objects of different difficulty, and here
they are. The critic must fit a bootstrapped, distributional return surface: its target is a moving
projected distribution built from its own next-state prediction, it must resolve the bimodal
survive-versus-fall structure into atom masses, and it is trained on off-policy data across the whole
buffer — a hard, sharp, non-stationary object. The actor must fit a much gentler thing: a smooth map
from state to a single action that ascends the critic, with the deterministic policy gradient giving
it a clean, low-variance direction to follow. A smoother target needs less capacity, so one residual
block at width 128 is enough for the actor, and spending the parameter budget on the actor instead of
the critic would put capacity where the fit is already easy. There is also a stability reason to keep
the actor lean: a larger, sharper actor would chase the critic's argmax more aggressively into
whatever spurious peaks the critic still has, which is the DDPG failure mode target smoothing exists to
suppress — so a lean actor is not only sufficient, it is safer. The critic gets the two blocks and the
width because that is where the hard, non-stationary fit lives and where the residual simplicity bias
has something to protect against; the actor gets one block because its job is smooth and I do not want
to amplify its pull on the critic. That reasoning, not a canonical table, is why the sizes are `1/128`
and `2/512`.

Everything else stays exactly FastTD3. The critic is still twin categorical distributional networks
over 101 atoms on `[−250, 250]`; the projection, clamp, floor/ceil split, and cross-entropy target
are unchanged; clipped double-Q still keeps the whole distribution whose mean is smaller; the actor
is still deterministic with the per-env mixed exploration noise and target-policy smoothing; the
actor still updates on the delayed `policy_frequency` schedule; `num_updates=2`, `tau=0.1`, the
bootstrap mask `(truncations | ~dones)`, and the fixed loop's observation normalization all carry
over. The *only* thing that changes between FastTD3 and this finale is that the actor's and critic's
network bodies become SimBa encoders — embedding, residual blocks, final LayerNorm — instead of plain
MLPs. That confinement is the point: it makes the comparison a clean test of the architecture and
nothing else, and it is what makes the claim "this is stronger than FastTD3" attributable to the
simplicity-bias body rather than to any algorithmic change. It also means I inherit every stability
property FastTD3 already had — the clipped-min pessimism, the target smoothing, the fleet exploration —
so I am not trading one stabilizer for another; I am adding the architectural bias on top of an
algorithm that is already stable, which is the only clean way to isolate whether the architecture is
the missing ingredient.

I should be honest about the one substrate friction, because it bears on whether the gain
materializes. FastTD3's design philosophy is anti-architecture for *speed* — plain MLPs are cheaper
per update, and the residual encoder adds LayerNorms and a 4× bottleneck that cost wall-clock. Under
a fixed *gradient-step* budget (100,000 steps) that cost is acceptable because the budget is counted
in steps, not seconds, so a heavier-but-better-converging critic is a fair trade; the risk is only
that the extra compute does not pay for itself within 100k steps if the plain MLP was already near
its ceiling on a given task. That risk is highest on stand, where FastTD3's plain critic is likely
already sufficient, and lowest on run, where the harder value surface is exactly what the scalable
critic should fit better. It is worth noting the risk is bounded in the right direction: because the
budget is in gradient steps rather than seconds, a slower-per-step critic is not penalized for its
wall-clock, so the only way this loses is if the residual body converges to a *worse* fit within the
same number of steps — and the at-init near-identity property says the worst case is that it starts
close to a benign linear map and simply fails to improve on the plain MLP, not that it destabilizes.

So the bar this finale must clear, against the strongest baseline's place on the ladder: it must
*not be worse* than FastTD3 on `h1hand-stand-v0` (where there is little headroom and the architecture
mostly has to avoid hurting), and it must be *better* than FastTD3 on `h1hand-walk-v0` and especially
`h1hand-run-v0` — the tasks whose sharper, longer-horizon value surfaces are where a plain MLP critic
overfits its moving target and a simplicity-biased residual critic should convert its capacity into a
cleaner fit and a higher final return. The mechanism is falsifiable in the same way the architecture's
central claim is: if scaling the critic with this encoder did *not* help on the hard tasks — if the
residual/normalized critic matched or trailed FastTD3's plain MLP on run — then the premise that the
missing ingredient is simplicity bias rather than something RL-intrinsic would be wrong, and the finale
would not deserve its place above FastTD3. What I would validate first is precisely that: the per-task
margin on run and walk against FastTD3's numbers, holding every algorithmic setting fixed, isolating the
encoder as the only difference. The distilled SimBa-in-the-scaffold fill — FastTD3's backbone with
residual encoders in the actor and critic — is in the answer.
