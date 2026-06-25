Let me start from the thing that actually bothers me, which is not a gap in any formula but a gap in *meaning*. Clausius hands me a quantity, entropy, defined by S(B) − S(A) = ∫ dQ/T along a reversible path, and tells me it never decreases for an isolated system, that the entropy of the world tends to a maximum. Every word of that is experimentally solid. And I do not understand a single thing about why it is true. A gas is molecules — points or little spheres flying around, colliding, obeying mechanics. Mechanics has no preferred direction of time: reverse every velocity and the gas runs backwards just as lawfully as forwards. So where, in a pile of time-symmetric collisions, does a one-way arrow come from? Clausius's S increases; Newton's equations don't care which way the clock runs. Those two statements sit on my desk refusing to be reconciled, and I am not going to be satisfied with "it just does."

There's a second thing, smaller but nagging. Clausius's S is only defined *between equilibrium states*, because dQ/T needs a temperature and a quasi-static path. If I let a gas loose in a vacuum and watch it halfway through filling the box, it has no temperature, no well-defined dQ. So strictly Clausius cannot even *tell me the entropy* of the gas at that instant. Yet surely the gas is "more disordered" at that instant than when it was crammed in one corner. There ought to be a quantity that is defined at every instant, equilibrium or not, that rises through the whole messy spreading-out. Clausius's entropy is the value that quantity should take *at the ends*, when equilibrium has returned. I want the quantity itself.

So let me not ask "how do I prove dQ/T increases." Let me ask: what is the gas actually *doing*, microscopically, when it goes from corner-of-the-box to filling-the-box, and why does it never go back?

Picture the box split in two, gas crammed in the left half, right half empty, partition pulled. The gas spreads. Why does it never spontaneously crowd back into the left half? Not because the back-trajectory is forbidden — it's a perfectly good solution of the equations of motion, just run the velocities in reverse. It's allowed. It simply doesn't happen. That word — doesn't *happen*, not *can't* — is the whole thing. It's the language of likelihood, not of law.

The lottery keeps coming back to me. Draw five numbers; the draw 1-2-3-4-5 feels miraculous when it comes up. But it is *exactly* as probable as any other specific quintet — 7-19-23-31-44, say. Every single combination is equally rare. What makes 1-2-3-4-5 feel special is not that it's improbable as a combination but that it belongs to a tiny, recognizable *class* — "the ordered ones" — while the draws that feel ordinary belong to an enormous class — "the jumbled ones." When I say "a jumbled draw is likely," I mean the *class* is likely, because the class is huge, even though each member of it is as rare as 1-2-3-4-5.

Now overlay that on the gas. A particular complete specification of the gas — molecule 1 here with this velocity, molecule 2 there with that one, every molecule pinned down — is one "combination." Every such complete specification is, I'll insist, equally probable; there's nothing to prefer one detailed micro-arrangement over another. The gas-crammed-in-the-left specification is exactly as probable, *as a single detailed arrangement*, as any particular spread-out one. So why does spread-out win? Because "spread out roughly uniformly" is a *class*, like "the jumbled lottery draws," and it contains an unfathomably larger number of detailed arrangements than "all in the left half" does. The gas drifts toward spreading-out for the same reason a shuffled deck comes up disordered: not because order is forbidden, but because disorder is where almost all the detailed arrangements live. The second law is not a law of mechanics at all. It's a statement that the gas, wandering among equally-likely detailed arrangements, spends essentially all its time in whichever coarse class owns the most of them.

If that's right, then entropy — the thing that's big at equilibrium and small when the gas is crammed in a corner — *measures how many detailed micro-arrangements belong to the gas's current coarse description*. That's the object. Let me make it precise enough to compute, because right now it's a picture, and pictures lie.

I need three levels, kept strictly apart, or I'll confuse myself. Top level: the macrostate — temperature, pressure, volume, the things a thermometer sees. Bottom level: a complete microscopic specification — every molecule's energy (or velocity) named individually. I'll call one of these a *complexion*. And in between, the level that actually matters: how many molecules have each energy (or each velocity), *without* caring which molecules they are. Call that a *state distribution*: a list (w_0, w_1, w_2, …) saying w_0 molecules have energy 0, w_1 have energy one unit, and so on. Many complexions collapse to the same state distribution, because relabeling which specific molecule sits where doesn't change the list of occupation numbers. The probability of a state distribution is the number of complexions that produce it, divided by the total number of complexions. So: *count the complexions per state distribution*. That number is the whole game.

Let me make it countable. A continuum of velocities can't be counted — there's no such thing as "how many points in an interval." So I'll do something that looks crude and turns out to be exactly the right move: chop the kinetic energy into a discrete ladder, 0, ε, 2ε, 3ε, …, pε. Each molecule sits on a rung. This is not physics — molecules don't have quantized energy in this picture — it's a scaffolding to make "count the arrangements" a literal counting problem, and at the end I'll let ε → 0 and p → ∞ and recover the continuum. In nature all infinities are limiting cases anyway; I'll take the limit when I need it and not before.

So: n molecules, each on one of the rungs 0…p, with total energy fixed at L = λε (λ units), and the molecule count fixed at n. A complexion is a full assignment — molecule 1 on rung 2, molecule 2 on rung 6, … A state distribution is the occupation list (w_0, …, w_p) with

    w_0 + w_1 + … + w_p = n        (the molecules are all accounted for)
    w_1 + 2w_2 + … + p·w_p = λ      (the energy is fixed).

How many complexions give a particular list (w_0,…,w_p)? This is just: in how many distinguishable ways can I hand out n labeled molecules so that w_0 of them land on rung 0, w_1 on rung 1, and so on? That's a multinomial. The n molecules can be ordered n! ways, but permuting the w_0 on rung 0 among themselves changes nothing, likewise the w_1, etc. So the number of complexions for this state distribution is

    P = n! / (w_0! w_1! … w_p!).

Let me sanity-check on something small enough to list by hand, because if the picture is right the most-arrangeable distribution should be visibly the most balanced one. Take seven molecules, total energy seven units, rungs up to seven. The constraints Σw_i = 7 and Σi·w_i = 7 are tight enough that I can write out every state distribution; there turn out to be fifteen of them. Rather than trust my eye, let me compute P = 7!/(Πw_i!) for each and actually find the maximum. A few representative ones: the distribution "three molecules at 0, two at 1, one at 2, one at 3" — I'll write it as the digit-string 0001123 — has P = 7!/(3!·2!·1!·1!) = 5040/12 = 420. The lopsided distribution "six molecules at 0, one at 7," written 0000007, has P = 7!/(6!·1!) = 7. Running the full table, the largest P is 420 (the 0001123 distribution), and the next-largest is 210 — so the balanced distribution wins, and wins by a clear factor of two over its nearest competitor and by 420/7 = 60 over the lopsided pile-it-on-one extreme. Good: the most-balanced distribution is the unique most-arrangeable one, not a tie, even at n = 7. With Avogadro's number of molecules the winning margin isn't sixty, it's an exponential of something enormous; the most-balanced distribution would then own essentially all the complexions compatible with the constraints, to any precision a thermometer could resolve. So the overwhelming likelihood of "spread out" over "piled up" is already visible in a seven-molecule toy, and it sharpens as N grows. That dominance is what I'll want to identify with the second law — but let me hold that identification until I've checked it against Clausius, not assert it now.

So the equilibrium state distribution should be the one that maximizes P. Maximize n!/(w_0!…w_p!): the numerator is fixed, so I want to *minimize the denominator* w_0! w_1! … w_p!, subject to the two constraints. Products of factorials are nasty to differentiate, so I'll minimize the logarithm instead — minimizing a product is minimizing its log — and the log turns the product into a sum:

    minimize  M = ln w_0! + ln w_1! + … + ln w_p!.

Factorials of large numbers I can tame with Stirling: ln w! ≈ w ln w − w (keeping the leading pieces; more precisely ln w! ≈ w ln w − w + ½ ln 2πw, but the logarithmic correction is subextensive next to the terms that scale with the whole gas — I'll keep an eye on it but it won't control the maximum). So

    M ≈ Σ_i (w_i ln w_i − w_i) = Σ_i w_i ln w_i − n,

and since n is fixed, I'm minimizing Σ_i w_i ln w_i subject to Σ w_i = n and Σ i·w_i = λ.

Two constraints, so two Lagrange multipliers; call them h (for the molecule-number constraint) and k (for the energy constraint). Minimize Σ w_i ln w_i + h Σ w_i + k Σ i w_i. Take the derivative with respect to a generic w_i and set it to zero:

    ∂/∂w_i [ w_i ln w_i + h w_i + k·i·w_i ] = ln w_i + 1 + h + k·i = 0,

so

    ln w_i = −(1 + h) − k·i ,   i.e.   w_i = e^{−(1+h)} · e^{−k i}.

Stare at that. The occupation of rung i falls off as e^{−k i} — *geometrically* in the rung index, *exponentially* in the energy. Write x = e^{−k}; then w_i = w_0 x^i. The ratios of successive occupations are all equal:

    w_1/w_0 = w_2/w_1 = w_3/w_2 = … = x.

This isn't an assumption I put in; it dropped out of "count the arrangements and find the fattest distribution." The most-arrangeable way to spread a fixed total energy over the molecules is a *geometric* decline of population with energy. Whether that geometric form is actually the right *physical* equilibrium I can't tell yet — I'll have to confront it with Maxwell's known law and with Clausius — but it's striking that an exponential in energy fell straight out of pure counting, with no dynamics assumed.

Now nail down x from the constraints. With w_i = w_0 x^i,

    w_0 (1 + x + x² + … + x^p) = n,
    w_0 (x + 2x² + 3x³ + … + p x^p) = λ.

The first sum is geometric, (x^{p+1}−1)/(x−1); the second is x·d/dx of the first. Divide the second equation by the first to eliminate w_0; the ratio is λ/n, the mean energy per molecule. In the limit of a tall ladder, p → ∞, the high powers x^p, x^{p+1} are negligible (x is a proper fraction below 1), so the first sum is 1/(1−x) and the second is x/(1−x)². Their ratio is then

    [ x/(1−x)² ] / [ 1/(1−x) ] = x/(1−x) = λ/n.

Solve x/(1−x) = λ/n: cross-multiply, nx = λ(1−x) = λ − λx, so x(n+λ) = λ, hence

    x = λ / (n + λ).

Let me check this actually satisfies the constraint I imposed, rather than trust the algebra blind. Put λ/n = 1 (mean energy one rung per molecule, as in the seven-molecule case): then x = 1/(1+1) = ½, and x/(1−x) = (½)/(½) = 1 = λ/n. The mean energy per molecule comes back out as 1. So x = λ/(n+λ) does reproduce the energy constraint — good.

Now what does this *mean*? The total energy is L = λε = n·μ, where μ = L/n is the mean kinetic energy per molecule. So λ/n = μ/ε, and therefore n/λ = ε/μ. Then

    x = λ/(n+λ) = 1/(1 + n/λ) = 1/(1 + ε/μ) ≈ e^{−ε/μ}   for small ε/μ.

So the probability that a molecule sits on rung s — has energy E = sε — is

    w_s / n ∝ x^s = e^{−sε/μ} = e^{−E/μ}.

The exponential of minus the energy over the temperature scale. In this one-coordinate count the scale μ plays the role of kT, the energy unit attached to temperature, so this is the Boltzmann factor e^{−E/kT} falling out of pure counting. And notice the thing that makes it physics rather than bookkeeping: x depends *only* on the ratio λ/n, the mean energy per molecule, not on n or λ separately. Two gases with the same mean energy per molecule have the same energy distribution regardless of size. The distribution is set by temperature alone. That's exactly the scale-invariance a temperature-controlled law must have, and it's reassuring that the counting respects it.

Before I go to the continuum I should make sure I've actually got a *minimum* of M, not a maximum or a saddle. The Hessian quadratic form comes from differentiating w_i ln w_i twice in w_i, giving 1/w_i, so

    δ²M = Σ_i (δw_i)² / w_i ,

which is positive for any nonzero variation, since every w_i > 0. So M is genuinely minimized, P genuinely maximized. Good — it's a peak, the gas really does pile up here.

Now the continuum. Let ε → 0, write w_i = ε f(iε) so f is a smooth density. The sum Σ w_i ln w_i, after pulling out the constant pieces involving ε and n (they're the same for every state distribution, so they can't affect *which* distribution wins), becomes an integral, and the quantity to maximize is

    Ω = − ∫ f(x) ln f(x) dx,

with the constraints ∫ f dx = n and ∫ x f dx = L. I write it with a minus sign because I'm now phrasing it as a *maximum* (the log of the number of arrangements, which grows as the denominator Σ w_i ln w_i shrinks). Vary f: add the constraints with multipliers and demand

    δ ∫ [ f ln f + k f + h x f ] dx = ∫ [ ln f + 1 + k + h x ] δf dx = 0.

For this to vanish for arbitrary δf, the bracket must vanish:

    ln f(x) = −1 − k − h x   ⇒   f(x) = C e^{−h x},   C = e^{−k−1}.

The continuous version of the same exponential. The Hessian quadratic form for Ω is δ²Ω = − ∫ (δf)²/f dx, strictly negative since f > 0, so this is a clean maximum. In energy the equilibrium density is f(x) ∝ e^{−hx}.

Before I celebrate, let me test this against the one piece of equilibrium physics I already trust — Maxwell's distribution — because if my count is right it had better reproduce his speed law. Maxwell's law for a three-dimensional gas of spheres is f(speed) ∝ v² e^{−mv²/2kT}: a Gaussian *times v²*. Take my result f(E) ∝ e^{−hE} in energy and convert it to a speed distribution. Energy is E = (m/2)v², so dE = m v dv, and the density transforms as f(E) dE = e^{−h(m/2)v²}·(m v) dv. That carries a factor of v¹, not v². So I get f(speed) ∝ v·e^{−(hm/2)v²}, and Maxwell has v²·e^{−(…)v²}. They disagree by exactly one power of v. My count is missing a factor of v.

So something about how I seeded the count is wrong, and the discrepancy is specific enough to point at the culprit. When I built the urn of energy-labeled slips I tacitly assumed equal numbers of slips for each equal *energy* interval — equal weight per dx of energy. But the molecules don't live in energy-space; they live in velocity-space, and the natural, mechanically-correct uniform measure is equal weight per *volume of velocity space*, du dv dw, not per interval of energy. Energy is (m/2)(u²+v²+w²), so a thin shell of energy corresponds to a thick shell of velocity — the Jacobian dE = mv dv means equal-energy slices are *not* equal-velocity-volume slices — and that conversion is exactly the factor of v I dropped. So I must redo the counting with the three velocity components as the variables, seeding the urn uniformly in du dv dw. Minimize Σ w ln w over cells in velocity space with the constraint that the total kinetic energy (m/2)(u²+v²+w²) summed over molecules is fixed. The same variational calculation gives

    f(u,v,w) = C e^{−h·(m/2)(u²+v²+w²)},

a Gaussian in each velocity component. Now convert *this* to a speed law: integrate over a spherical shell of radius v in velocity space, whose volume element is 4πv² dv. That supplies the v² directly, so f(speed) ∝ v²·e^{−(hm/2)v²} — Maxwell's law exactly, with the missing power of v now accounted for by the shell volume rather than dropped. The two now agree. And the lesson is sharp: the answer depends on *what you take to be equally likely a priori*, and counting in energy gives a wrong physical law that is nonetheless internally consistent — variational, normalizable, exponential — so I cannot pick the right measure by elegance. Only the comparison with Maxwell's known v² told me which measure is correct, and it is the one uniform in velocity-phase-space, the measure mechanics actually preserves.

There are *other* tempting "maximize a probability" recipes, and most of them give the wrong physics — and seeing exactly why sharpens what the correct object is. Suppose I tried to maximize the *product of the single-rung probabilities* — make B = w_0 · w_1 · w_2 · … large, i.e. maximize Σ ln w_i, with the same two constraints. Differentiate: 1/w_i + h + k·i = 0, so 1/w_i = a + b·i for constants a, b. That's a completely different distribution — the occupations go like 1/(a + b i), and for some ranges of parameters w even *increases* with energy. It does not give the Maxwellian spread of velocities at all; the dispersion comes out wrong. Or suppose I drew molecules from an urn with replacement and called the resulting multinomial-with-extra-weights the "probability." Working it out, the weight on the high-energy outcomes is so heavy that the expression has no clean limit as the discretization is refined — it doesn't even converge to a distribution. So neither "maximize the product of the rung-probabilities" nor "the with-replacement urn" reproduces thermal equilibrium. The *only* recipe that does is the one I started with: count *complexions* — permutations of labeled molecules, n!/Πw_i! — and find the state distribution with the most of them. That's not an aesthetic preference; it's forced by demanding agreement with the equilibrium gas. The multiplicity, the number of arrangements, is the right probability measure; nothing else is.

Now I have Ω = −∫ f ln f, the logarithm of the number of complexions of the most-probable distribution, up to an additive constant. The picture I started from said this counting quantity should *be* entropy — large when the gas is spread out, small when it's crammed in a corner. But that's a hope, not a result. To earn it I have to confront Ω with Clausius on his own terms. Two things must hold, and if either fails the identification is dead: Ω must be additive the way his S is, and it must numerically reduce to his ∫dQ/T for a gas in equilibrium. Let me test both.

The additivity is where the logarithm justifies itself, and I want to be careful because it's the keystone. Take two independent bodies, A and B. The number of complexions of the *joint* system is the product: any complexion of A can be paired with any complexion of B, so W(A+B) = W(A)·W(B). Multiplicities *multiply*. But Clausius's entropy *adds*: the entropy of the compound body is S(A) + S(B). If my mechanical entropy is to match his, it must be a function of W that turns multiplication into addition — it must satisfy g(W_A · W_B) = g(W_A) + g(W_B). With the ordinary continuity or monotonicity demanded of a physical state function, this leaves only a constant multiple of the logarithm. So entropy must be (a constant times) the *logarithm* of the number of complexions,

    S = k · log W.

That's why the log is there. Not for convenience, not to tame large numbers — though it does — but because entropy is additive while the thing it counts is multiplicative, and the log is the unique bridge between the two. And it's exactly why I defined Ω with the additive constants stripped out: with them gone, the permutability measure of two bodies is the sum of the permutability measures of each, precisely the extensivity Clausius demands. If instead I'd carried the raw count W around, the *number of permutations* of a compound system would be the *product* of the numbers for its parts — and that's the wrong arithmetic for an entropy. The log fixes the arithmetic.

Now the numerical match. For a monatomic ideal gas in equilibrium, plug the Maxwell distribution back into Ω = −∫ f ln f, integrating over the container volume V and all velocities. With N molecules, mean kinetic energy T (I'm using T to denote the mean kinetic energy per molecule itself — that's my temperature unit), and mass m, the equilibrium density is f = (N/V)(3m/4πT)^{3/2} e^{−3m(u²+v²+w²)/4T}, and the integral gives

    Ω = (3N/2) + N ln[ V (4πT/3m)^{3/2} ] − N ln N.

Now compare with thermodynamics. The ideal gas obeys pV = (2/3)N·T (the pressure comes from the mean kinetic energy, and with T defined as the mean kinetic energy per molecule the factor is 2/3), and the first law gives the heat dQ = N dT + p dV. Rather than integrate, let me compare the two *differentials* — it's cleaner and leaves no ambiguity about the constant. Differentiate the Ω above: the V-dependence is N ln V, so ∂Ω/∂V = N/V; the T-dependence is N ln(T^{3/2}) = (3N/2) ln T, so ∂Ω/∂T = (3N/2)/T; and the −N ln N piece has no V or T in it, so it drops. Hence

    dΩ = N dV/V + (3N/2) dT/T.

For the thermodynamic side, dQ/T = (N dT + p dV)/T, and substituting p = (2/3)N T/V so that p/T = (2/3)N/V,

    dQ/T = N dT/T + (2/3)N dV/V.

Put them side by side. Multiply dΩ by two-thirds:

    (2/3) dΩ = (2/3)[ N dV/V + (3N/2) dT/T ] = (2/3)N dV/V + N dT/T,

which is term-for-term identical to dQ/T. So the differentials agree, and integrating,

    ∫ dQ/T = (2/3) Ω.

The thermodynamic entropy of the gas equals two-thirds of my permutability measure, to within an additive constant. The two-thirds is nothing deep — it's the price of having measured temperature as the mean kinetic energy T rather than in degrees; absorb it into the constant and write μ for the mean energy and the relation is just S = k·(−∫ f ln f) = k log W. Clausius's entropy *is* the logarithm of the number of microscopic arrangements, times a constant. The opaque thermodynamic quantity has a mechanical meaning, and it is the meaning the lottery picture promised.

And now the two deficiencies I started with are gone in one stroke. Ω is defined for *any* state distribution, not just equilibrium ones — I can write down −∫ f ln f for any f whatsoever, in or out of equilibrium. So the gas spreading through the box, with no temperature and no Clausius entropy, *does* have an Ω at every instant, and the most numerous states are precisely the ones with the largest Ω. The second law's increase should be the gas climbing from a low-multiplicity macrostate to the overwhelmingly-dominant high-multiplicity one. It "increases" not because mechanics forbids the reverse but because the reverse is the lottery's 1-2-3-4-5: allowed, and never seen.

There's a trap here I have to mark, or someone will fall in it. One might guess that this Ω is the logarithm of the *phase-space volume the system occupies* as it moves. It is not, and it can't be — because by Liouville's theorem that volume is conserved along the motion: a blob of initial conditions, dξ…dw, evolves into dΞ…dW with exactly the same volume. If entropy were the log of that volume it would be constant in time, and nothing would ever increase. The resolution is that Ω is the log of the number of complexions of a *coarse-grained* description — the state distribution, the occupation numbers — not of a fine-grained phase volume. The blob keeps its volume but stretches into ever-finer filaments that thread through ever-more coarse cells, so the coarse multiplicity grows while the fine volume doesn't. The coarse-graining — the step where I stopped caring *which* molecule is where and counted only *how many* are in each cell — is not a sloppiness to apologize for; it is the very thing that lets entropy increase.

So far this is a story about *probability*: equilibrium is the most-arrangeable state, the gas is overwhelmingly likely to be there. But "overwhelmingly likely" still leaves me wanting an actual *dynamical* statement — a quantity that I can show, from the collisions themselves, can only fall (or rise) as time runs forward. I want to watch the gas relax and prove the relaxation is one-way. Let me build the collisions in.

Let f(x,t) be the density of molecules with energy x at time t. In a small time, collisions change it. A collision takes two molecules with energies x and x′ and leaves them with energies ξ and x+x′−ξ (energy conserved). So f at energy x is depleted by collisions that scatter an x-molecule away — proportional to f(x)f(x′), the number of x-and-x′ pairs available — and replenished by collisions whose products land at x — proportional to f(ξ)f(x+x′−ξ). Summing gain minus loss over all collision partners and geometries (this is the collision-number ansatz: the rate of a collision type is proportional to the product of the densities of the two incoming molecules — I'm assuming the colliding pair's energies are uncorrelated before they meet, which is the one statistical input):

    ∂f(x,t)/∂t = ∫∫ [ f(ξ)f(x+x′−ξ) − f(x)f(x′) ] · (collision kernel) dx′ dξ.

That's the evolution equation. Now I want a quantity built from f that the collisions can only push one way. Guided by the counting story, where −∫ f ln f was the thing that's maximal at equilibrium, let me try its negative as the candidate that should *decrease*. Define

    H[f] = ∫ f(x,t) ln f(x,t) dx

over the same one-particle cell measure used in the collision equation. Differentiate in time, pulling ∂/∂t under the integral:

    dH/dt = ∫ (ln f + 1) ∂f/∂t dx = ∫ (ln f + 1) ·[collision integral] dx.

The "+1" integrates against ∂f/∂t, and since the total number of molecules ∫ f dx is conserved, ∫ ∂f/∂t dx = 0, so the +1 contributes nothing. Substitute the collision integral:

    dH/dt = ∫∫∫ ln f(x) · [ f(ξ)f(x+x′−ξ) − f(x)f(x′) ] · (kernel) dx dx′ dξ.

This is *not* manifestly signed — there's a lone ln f(x) sitting against a difference of products, and I can't tell what sign it has. A collision (x, x′) → (ξ, x+x′−ξ) can be described four equivalent ways, by relabeling which molecule I call which and by running it forward versus backward: swap x ↔ x′, swap the two outgoing energies, and exchange incoming ↔ outgoing (the reverse collision, which the mechanics allows with the same kernel — that's where time-reversibility of the *microdynamics* enters, honestly, as a symmetry of the kernel, not as a fudge). Each relabeling gives an expression for dH/dt that is equal to the original, because it's the same physical sum reorganized. Write the four out, with s = f(x)f(x′) the incoming-pair density and σ = f(ξ)f(x+x′−ξ) the outgoing-pair density. The two incoming labelings give ln f(x) and ln f(x′) multiplying (σ − s); the reverse labelings give ln f(ξ) and ln f(x+x′−ξ) multiplying (s − σ), which is the same as a minus sign against (σ − s). Average the four equal forms:

    dH/dt = (1/4) ∫∫∫ [ ln f(x) + ln f(x′) − ln f(ξ) − ln f(x+x′−ξ) ] · (σ − s) · (kernel) dx dx′ dξ.

The two outgoing logs come with minus signs from the reverse-collision relabeling. But ln f(x) + ln f(x′) = ln[ f(x)f(x′) ] = ln s, and ln f(ξ) + ln f(x+x′−ξ) = ln σ. So the bracket is ln s − ln σ = ln(s/σ), and

    dH/dt = (1/4) ∫∫∫ ln(s/σ) · (σ − s) · (kernel) dx dx′ dξ.

Now look at the integrand's sign, point by point. The kernel is nonnegative. Consider ln(s/σ)·(σ − s). If s > σ: then s/σ > 1 so ln(s/σ) > 0, while (σ − s) < 0 — product negative. If s < σ: then ln(s/σ) < 0 while (σ − s) > 0 — product negative again. If s = σ: the product is zero. So for *every* pair of positive numbers s, σ, the quantity ln(s/σ)·(σ − s) ≤ 0, with equality only when s = σ. The integrand is non-positive everywhere. Therefore

    dH/dt ≤ 0,

always. H can only decrease, never increase, and it stops changing only when s = σ for every collision with nonzero kernel — that is, when

    f(x)f(x′) = f(ξ)f(x+x′−ξ)   whenever   x + x′ = ξ + (x+x′−ξ).

Take the log: ln f(x) + ln f(x′) = ln f(ξ) + ln f(x+x′−ξ) whenever the energies before equal the energies after. So ln f must be a quantity that is *conserved in collisions and additive over the pair*. In the rest frame, after molecule number and total energy are fixed, that leaves a constant plus the energy. Hence ln f(x) = −h x + const, i.e. f(x) ∝ e^{−hx} — the exponential again, the Maxwell distribution, now arrived at from a completely different direction: as the stationary point of the relaxation. The distribution that makes H stop falling is exactly the distribution that maximizes the number of complexions. The dynamics and the counting agree: H falls unless f is Maxwellian, and Maxwellian is where Ω = −H is maximal.

I should check H doesn't simply fall forever to minus infinity — it can't, because at fixed molecule number and fixed total energy the value of H is bounded below, with the Maxwell distribution as the minimizing stationary state. Under this collision description any stationary endpoint must be Maxwellian, and once that state is reached it remains there. And since −H is, up to the constant k, exactly the entropy Ω I matched to Clausius, this *is* the second law made dynamical: entropy ∝ −H rises monotonically until the maximum-entropy, maximum-multiplicity, Maxwell-distributed state is reached.

Let me also be honest with myself about what this proof rests on, because the time-reversibility worry from the start hasn't actually gone away — it's been *located*. The microdynamics is reversible; my dH/dt ≤ 0 is monotone. How? The asymmetry entered at exactly one point: the collision-number ansatz, where I assumed the two molecules about to collide have *uncorrelated* energies — that the pair density factorizes as f(x)f(x′) *before* the collision. After a collision the two molecules are correlated (their energies are linked by what just happened), so the assumption is not symmetric in time; it's a statement about the state *before* collisions, not after. That is where the irreversibility enters. It's not a theorem of mechanics that the pre-collision energies are uncorrelated; it's an overwhelmingly-good probabilistic statement about a gas of enormously many molecules — the same overwhelming likelihood as the lottery picture. So the monotone H-theorem is not in contradiction with reversible mechanics; it's a probabilistic statement wearing a dynamical coat, and the coat fits because the number of molecules is astronomically large. The reversibility objection — that I could reverse all velocities and make H climb — is correct *as a statement about a single measure-zero set of contrived microstates*, and irrelevant as a statement about what a real gas does, for the very same reason the gas never crawls back into the corner of the box. The two pictures, counting and dynamics, are the same picture.

The chain is now tight. Clausius's entropy is real but meaningless at the molecular level, undefined out of equilibrium, and unexplained as to its increase. I reconceive the gas's state in three tiers — macrostate, state distribution, complexion — and grant every complexion equal likelihood. The probability of a macrostate is then the *number of complexions* it contains, which for discretized energy is the multinomial n!/Πw_i!. Maximizing that count, via Stirling and two Lagrange multipliers, forces the occupations into a geometric/exponential ladder w_i ∝ x^i with x set by the mean energy alone — Maxwell's distribution, gotten by counting. The logarithm of the maximal count, Ω = −∫ f ln f, is forced to be the entropy because entropy is additive while complexion-counts multiply, and the log is the unique converter of products into sums — S = k log W. Plugged into the ideal gas, Ω matches Clausius's ∫dQ/T exactly (up to the unit factor), so the abstract count *is* the thermodynamic entropy; and being defined for any f, it supplies the missing non-equilibrium entropy and explains the increase as the climb to the dominant macrostate. When I build the collisions explicitly, H = ∫ f ln f carries the same functional with the opposite sign, and the four-fold symmetry of a collision forces dH/dt ≤ 0 with equality only at the exponential — so the collision dynamics drives the gas toward Maxwell, the maximum-multiplicity state, while entropy rises to its maximum. The second law is the overwhelming statistical likelihood of high-multiplicity macrostates, and entropy is the logarithm of how many ways a macrostate can be.

```python
from math import factorial, log

# --- The complexion count: how many labeled-molecule arrangements give a macrostate ---
# macrostate = occupation list (w_0, w_1, ...): w_i molecules on energy rung i.
def arrangements(occ):
    # P = n! / (w_0! w_1! ...)  -- the multinomial; this is "W", the multiplicity.
    n = sum(occ)
    p = factorial(n)
    for w in occ:
        p //= factorial(w)
    return p

# --- Entropy = k * log(multiplicity).  log forced by additivity (W multiplies, S adds). ---
def entropy(occ, k=1.0):
    return k * log(arrangements(occ))            # S = k log W

# --- Enumerate macrostates of n molecules, total energy = units (each rung i contributes i) ---
def macrostates(n, units, max_level):
    def rec(level, left_mol, left_energy):
        if level == 0:                            # rung 0 mops up the remaining molecules
            if left_energy == 0:
                yield (left_mol,)
            return
        for w in range(left_mol + 1):
            if w * level <= left_energy:
                for tail in rec(level - 1, left_mol - w, left_energy - w * level):
                    yield tail + (w,)
    yield from rec(max_level, n, units)

# --- Most probable macrostate = the one with the most complexions (= equilibrium) ---
def most_probable(n, units, max_level):
    return max(macrostates(n, units, max_level), key=arrangements)

# --- The equilibrium occupations in the large-n limit: geometric ladder w_i ∝ x^i, ---
#     x = (mean energy)/(mean energy + 1 rung) -> Boltzmann factor e^{-i / mean}. ---
def equilibrium_occupations(n, units, max_level):
    mean = units / n                              # mean energy per molecule (sets "temperature")
    x = mean / (mean + 1.0)                       # x = lambda/(n+lambda) per rung; ~ e^{-1/mean}
    raw = [x**i for i in range(max_level + 1)]
    Z = sum(raw)
    return [n * r / Z for r in raw]               # w_i ∝ x^i  (Maxwell–Boltzmann)

if __name__ == "__main__":
    # Boltzmann's hand-checkable case: 7 molecules, total energy 7 units, rungs 0..7.
    eq = most_probable(7, 7, 7)
    print("most-probable macrostate (w_0..w_7):", eq)        # (3,2,1,1,0,0,0,0)
    print("its multiplicity W =", arrangements(eq))          # 420  -- the maximum
    print("its entropy  S = log W =", round(entropy(eq), 4))
    # the lopsided 'all energy on one molecule' macrostate, for contrast:
    lop = (6,0,0,0,0,0,0,1)
    print("lopsided W =", arrangements(lop))                 # 7  -- 60x fewer arrangements
    print("large-n occupations w_i (geometric/Boltzmann ladder):",
          [round(w, 3) for w in equilibrium_occupations(7, 7, 7)])
```
