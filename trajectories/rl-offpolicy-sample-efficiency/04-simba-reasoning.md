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
exactly those, and nothing algorithm-specific, should let FastTD3's critic finally scale. This is a
clean, falsifiable claim about the architecture, and it fits the edit surface perfectly: the task
lets me redesign the `Actor` and `Critic` networks while leaving `update_critic`, `update_actor`, the
clipped double-Q logic, the distributional projection, the per-env exploration, and the whole fixed
loop exactly as FastTD3 has them.

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
convention (orthogonal, tiny gain) so the deterministic policy starts near zero action — that head
behavior is part of the backbone and I do not disturb it. The point of the discipline is that
residual + LayerNorm + He init is the *combination* that lets a deep, wide stack train; drop any one
and the same architecture can stall.

Now the sizing, which is the whole reason to do this: I want capacity to *convert*. The value
function is the harder object — it has to fit a sharp, bootstrapped, contact-driven return surface —
while the policy is smoother, so I make the critic the larger of the two. The canonical configuration
is one residual block at hidden width 128 for the actor and two residual blocks at hidden width 512
for the critic, and that asymmetry is exactly the bet: the critic gets the extra capacity, the
residual/normalized structure makes that capacity safe, and the actor stays lean. Against FastTD3's
critic (a `1024→512→256` plain MLP, roughly two affine layers of real depth with no skip) the SimBa
critic is a genuinely deeper, normalized, residual network of comparable parameter scale but with the
simplicity bias that lets the depth help. The optimizer pairs with it: AdamW with a modest weight
decay is itself a simplicity-bias regularizer (it pulls weights toward small norm, toward simpler
functions) and composes with the residual structure. The substrate already uses AdamW with weight
decay 0.1 and cosine annealing, so I keep that optimizer family; the canonical SimBa decay is on the
order of one part in a hundred, in the same regime, and I leave the substrate's value rather than
introduce a new knob, because the architectural change is the variable under test.

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
simplicity-bias body rather than to any algorithmic change.

I should be honest about the one substrate friction, because it bears on whether the gain
materializes. FastTD3's design philosophy is anti-architecture for *speed* — plain MLPs are cheaper
per update, and the residual encoder adds LayerNorms and a 4× bottleneck that cost wall-clock. Under
a fixed *gradient-step* budget (100,000 steps) that cost is acceptable because the budget is counted
in steps, not seconds, so a heavier-but-better-converging critic is a fair trade; the risk is only
that the extra compute does not pay for itself within 100k steps if the plain MLP was already near
its ceiling on a given task. That risk is highest on stand, where FastTD3's plain critic is likely
already sufficient, and lowest on run, where the harder value surface is exactly what the scalable
critic should fit better.

So the bar this finale must clear, against the strongest baseline's place on the ladder: it must
*not be worse* than FastTD3 on stand (where there is little headroom and the architecture mostly has
to avoid hurting), and it must be *better* than FastTD3 on walk and especially run — the tasks whose
sharper, longer-horizon value surfaces are where a plain MLP critic overfits its moving target and a
simplicity-biased residual critic should convert its capacity into a cleaner fit and a higher final
return. The mechanism is falsifiable in the same way the architecture's central claim is: if scaling
the critic with this encoder did *not* help on the hard tasks — if the residual/normalized critic
matched or trailed FastTD3's plain MLP on run — then the premise that the missing ingredient is
simplicity bias rather than something RL-intrinsic would be wrong, and the finale would not deserve
its place above FastTD3. What I would validate first is precisely that: the per-task margin on run
and walk against FastTD3's numbers, holding every algorithmic setting fixed, isolating the encoder as
the only difference. The distilled SimBa-in-the-scaffold fill — FastTD3's backbone with residual
encoders in the actor and critic — is in the answer.
