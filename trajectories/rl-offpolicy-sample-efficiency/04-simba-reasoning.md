FastTD3 is the strongest of the three, and the reason it wins tells me where its ceiling is. It beat
FastSAC by dropping the entropy tax and getting exploration from 128 parallel environments, and it beat
PPO by reusing experience while exploiting the value function greedily with a deterministic actor. Both
wins are about *the algorithm* — the losses, the targets, the exploration. What FastTD3 deliberately did
*not* invest in is the network: a plain descending ReLU MLP with no normalization, on the bet that data
diversity tames the deadly triad so architecture is unnecessary. That bet is right for being fast and
simple, but it leaves the opening I named at the close of the last step — a plain MLP critic does not
reliably convert extra capacity into a clean value fit on the hardest-to-fit value surfaces. On stand
the value function is gentle and the plain critic is plenty; on walk and especially run the return
surface is sharper, more contact-driven, and longer-horizon, and a bare MLP chasing a moving
bootstrapped target there is exactly where unregularized capacity overfits the transient target and the
triad bites. FastTD3's answer to instability was "more diverse data, smaller-than-tempting network." The
complementary answer it left on the table is "let the network scale, by giving it the bias that makes
scaling safe." That is the move of this finale: keep the entire FastTD3 backbone untouched and change
only the function class of the actor and critic.

Why is the architecture the right lever and not just another knob? In supervised vision and language,
scaling parameters reliably improves results — not because the networks are small, but because modern
architectures carry an implicit *simplicity bias* from standard components (normalization, residual
paths, careful init) that steers optimization toward simple, generalizable functions even when heavily
over-parameterized. Deep RL has not inherited this: the standard actor-critic is a bare MLP, and
widening it usually regresses, because the value target moves (bootstrapped through the critic's own
next-state value) and the data distribution drifts (the policy changes), so raw capacity fits the
transient target too hard and feeds the error back. FastTD3's plain MLP is precisely this bare network.
The hypothesis I act on is that the missing thing is not RL-specific — it is the simplicity-inducing
components — so importing exactly those, and nothing algorithm-specific, should let FastTD3's critic
finally scale.

There are three ways to spend capacity on the critic and only one is the bet I want. Widen the plain MLP
— take the `1024→512→256` critic to `2048→1024→512` — and with no identity path and no normalization the
added width is added capacity to overfit the moving target; eliminated on exactly the mechanism I am
trying to fix. Add LayerNorm to the plain MLP — essentially FastSAC's normalized body, which I have
evidence stabilizes an off-policy critic — controls feature scale but creates no *identity pathway*: a
stack of `Linear→LayerNorm→activation` layers must still learn the near-linear map from scratch through
every layer, and deep normalized-but-non-residual stacks still degrade with depth. Normalization treats
the symptom (scale drift) without granting the structural bias (simple-by-default) that lets depth
*help*. The third option is the residual inverted-bottleneck block, which adds the identity pathway on
top of the normalization — making the simplest function free and any nonlinearity an opt-in correction.
So the elimination is principled: widening fails on overfitting, LayerNorm-only fixes scale but not
depth-degradation, and only residual-plus-norm-plus-careful-init grants the simplicity bias that makes
scaling safe. It fits the edit surface exactly: I redesign the `Actor` and `Critic` networks while
leaving `update_critic`, `update_actor`, the clipped double-Q logic, the distributional projection, the
per-env exploration, and the whole fixed loop as FastTD3 has them.

The encoder, applied identically to actor and critic, has three ingredients. First, running-statistic
observation normalization — but the substrate *already* supplies this: `EmpiricalNormalization` in the
fixed loop standardizes actor, critic, and next obs online before every update, so this ingredient is
inherited, not duplicated inside the network. My edit is the *body*. Second, the residual feedforward
block, the ingredient FastTD3's plain MLP lacks and the one that does the real work. Instead of a stack
of fresh affine-plus-ReLU layers, each block computes a correction and adds the input back: `out = x +
F(x)`. The addition creates a linear identity pathway, so the simplest function a block can represent —
the identity — is free, and any nonlinearity is an additive opt-in correction. For a bootstrapped value
function chasing a moving target, defaulting to simple is exactly the regularization that stops the
overfitting FastTD3's bare MLP was exposed to on the hard tasks. The block body is the standard inverted
bottleneck: pre-LayerNorm the input, a linear map up to 4× the width, a ReLU, a linear map back down,
then the residual add. The *pre*-LN keeps the residual stream clean — the identity branch carries the
un-normalized signal forward while each correction is computed from a normalized version — and the 4×
expansion is where the scaling parameters live, so widening the critic grows the bottleneck. Third, a
final LayerNorm before the head: the residual stream accumulates (input plus every block's correction),
so its magnitude grows with depth, and one LayerNorm standardizes it regardless of block count.

Initialization makes the deep stack train from step one. The embedding linear gets orthogonal init at
unit gain — a well-conditioned lift into the residual stream. The two linears inside each block get He
(Kaiming) normal init, variance-preserving for ReLU, so a block's correction starts at a sane magnitude
that neither vanishes nor explodes as blocks stack. The actor's final tanh head keeps FastTD3's small-init
convention (orthogonal, tiny gain `0.01`) so the deterministic policy starts near zero action.

This delivers the "starts near the identity" property the simplicity-bias argument counts on. At
initialization a block's `F(x) = fc_down(ReLU(fc_up(LN(x))))` produces, from He-init weights, a
correction of order the standardized signal — order one — added onto `x`, so the block passes `x`
through *plus* a bounded perturbation rather than replacing it with a freshly-computed map. Stack `N`
blocks and the encoder at init is the embedding plus a sum of `N` bounded corrections — a near-linear,
well-conditioned function, not a random deep tangle. Drop the residual add and the same `N`-layer stack
is a composition of `N` random maps whose output has lost the input, the very thing that makes deep bare
stacks hard to train. So residual + pre-LN + He init together buy the property; any one alone does not.

Now the sizing, the whole reason to do this: I want capacity to *convert*, and I make the critic the
larger of the two — one residual block at width 128 for the actor, two blocks at width 512 for the
critic. The asymmetry is the bet, and it is only right because the two networks fit objects of different
difficulty. The critic must fit a bootstrapped, distributional return surface: a moving projected
target built from its own next-state prediction, resolving the bimodal survive/fall structure into atom
masses, trained on off-policy data across the whole buffer — a hard, sharp, non-stationary object, and
exactly where the residual simplicity bias has something to protect against. The actor fits a much
gentler thing: a smooth map from state to a single action ascending the critic, with the deterministic
policy gradient giving a clean, low-variance direction. A smoother target needs less capacity, and a
leaner actor is also *safer* — a larger, sharper actor would chase the critic's argmax more aggressively
into whatever spurious peaks remain, the DDPG failure mode target smoothing exists to suppress. So the
critic gets the depth and width, the actor stays lean. Counting the critic to check this is genuine
capacity comparable to what I replace: each width-512 block with a 4× bottleneck is `2 × (512·2048) ≈
2.1M` weights, so two blocks are `≈4.2M`, plus the embedding and the atom head — against FastTD3's
`1024→512→256` plain MLP at `≈ n_in·1024 + 655k`, a genuinely deeper, normalized, residual network of
comparable parameter scale but arranged so that adding a block moves the function toward a better fit
instead of overfitting. The optimizer composes with this: AdamW's weight decay is itself a
simplicity-bias regularizer, and the substrate already uses AdamW with weight decay `0.1` and cosine
annealing, so I keep it and leave its value rather than introduce a new knob, because the architectural
change is the variable under test.

Everything else stays exactly FastTD3: twin categorical distributional critics over 101 atoms on
`[−250,250]`; the projection, clamp, floor/ceil split, and cross-entropy target unchanged; clipped
double-Q keeping the whole distribution whose mean is smaller; the deterministic actor with per-env
mixed exploration noise and target-policy smoothing; the delayed `policy_frequency` actor update;
`num_updates=2`, `tau=0.1`, the bootstrap mask `(truncations | ~dones)`, the loop's observation
normalization. The *only* change between FastTD3 and this finale is that the actor's and critic's bodies
become residual encoders instead of plain MLPs. That confinement is the point: it makes the comparison a
clean test of the architecture and nothing else, and it means I inherit every stability property FastTD3
already had — the clipped-min pessimism, the target smoothing, the fleet exploration — so I am adding the
architectural bias on top of an already-stable algorithm.

One substrate friction bears on whether the gain materializes. FastTD3's design is anti-architecture for
*speed* — plain MLPs are cheaper per update, and the residual encoder's LayerNorms and 4× bottleneck cost
wall-clock. But the budget is counted in gradient steps (100,000), not seconds, so a heavier,
better-converging critic is a fair trade; the risk is only that the extra compute does not pay for itself
if the plain MLP was already near its ceiling on a task. That risk is highest on stand, where FastTD3's
plain critic is likely already sufficient, and lowest on run, where the harder value surface is exactly
what the scalable critic should fit better. The risk is bounded in the right direction: because the
budget is in steps, a slower-per-step critic is not penalized for wall-clock, and the at-init
near-identity property says the worst case is that it starts close to a benign linear map and simply
fails to improve, not that it destabilizes.

So the bar. This finale must *not be worse* than FastTD3 on `h1hand-stand-v0` (little headroom, the
architecture mostly has to avoid hurting), and *better* on `h1hand-walk-v0` and especially
`h1hand-run-v0`, the tasks whose sharper, longer-horizon value surfaces are where a plain MLP critic
overfits its moving target and a simplicity-biased residual critic should convert its capacity into a
cleaner fit. The mechanism is falsifiable: if scaling the critic with this encoder did not help on run —
if the residual critic matched or trailed the plain MLP there — then the premise that the missing
ingredient is simplicity bias rather than something RL-intrinsic would be wrong, and the finale would not
deserve its place above FastTD3. What I would validate first is exactly that: the per-task margin on run
and walk against FastTD3, holding every algorithmic setting fixed, isolating the encoder as the only
difference. The distilled fill — FastTD3's backbone with residual encoders in the actor and critic — is
in the answer.
