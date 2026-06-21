Let me sit with the thing that bothers me. I have the electrodynamics of a massless charged scalar — a complex `φ`, two real components `φ₁, φ₂`, minimally coupled to the photon, with a quartic self-coupling and no mass term:

```
L = −¼Fμν² + ½(∂μφ₁ − eAμφ₂)² + ½(∂μφ₂ + eAμφ₁)² − (λ/4!)(φ₁²+φ₂²)² .
```

Two dimensionless couplings, `e` and `λ`, and not one dimensionful constant. The classical potential is `V_cl = (λ/4!)(φ₁²+φ₂²)²`. For `λ>0` the only stationary point is the origin, and it's a minimum. So the semiclassical verdict is: symmetric vacuum, massless scalar, massless photon, nothing happens. People have stared at this theory for years and gotten paradoxes out of it. I want to know whether the verdict is actually right.

Here is what nags me. The classical analysis answers "where is the vacuum?" by minimizing the classical potential — but the classical potential is just the non-derivative part of `L` with all the quantum fluctuations switched off. The fields fluctuate. Those fluctuations carry zero-point energy, and that energy depends on where in field space I sit, because the masses of the fluctuations depend on the background. A background `φc` gives the scalar fluctuations an effective mass-squared like `λφc²/2` and the photon an effective mass-squared `e²φc²` — minimal coupling does that automatically. So the zero-point energy of the vacuum is a function of `φc`. The classical potential ignores all of it. Whether the true vacuum is at the origin is therefore not a classical question at all; it's a question about the *total* energy, fluctuations included, as a function of the background. The classical answer could be completely wrong.

So I need the right object: the energy density of the best state with `⟨φ⟩ = φc` held fixed, as a function of `φc`, with quantum corrections in. That object is the effective potential, and I should build it carefully rather than hand-wave.

Start from Schwinger's generating functional. Couple `φ` linearly to a classical source, `L → L + J(x)φ(x)`. Define `e^{iW[J]} = ⟨0⁺|0⁻⟩_J`, the connected generating functional. The classical field is `φc(x) = δW/δJ(x) = ⟨0⁺|φ(x)|0⁻⟩_J / ⟨0⁺|0⁻⟩_J` — the source-driven expectation value of the field. Now Legendre-transform, the way Jona-Lasinio did when he first turned this machinery onto symmetry breaking: `Γ[φc] = W[J] − ∫d⁴x J(x)φc(x)`. Differentiate and the source falls out cleanly, `δΓ/δφc(x) = −J(x)`. So at vanishing source, the physical condition for a vacuum is `δΓ/δφc = 0`.

What is `Γ`? Expand it two ways. In powers of `φc` it's a sum over the 1PI Green's functions `Γ⁽ⁿ⁾` — the proper vertices, the sum of all one-particle-irreducible graphs with `n` external legs and no propagators on those legs. Alternatively, expand in derivatives of `φc`:
```
Γ[φc] = ∫d⁴x [ −V(φc) + ½(∂μφc)² Z(φc) + … ] .
```
The leading, no-derivative term defines `V(φc)`, an ordinary function — the **effective potential**. Comparing the two expansions, the `n`-th derivative of `V` is the sum of all 1PI graphs with `n` external legs at *zero* external momentum. And for a translation-invariant vacuum — which is all I want, I'm not interested in breaking momentum conservation — `δΓ/δφc = 0` collapses to `dV/dφc = 0`. Symmetry is spontaneously broken precisely when the minimum of `V` sits at `φc = ⟨φ⟩ ≠ 0`, even with the source off. Stability says it must be a minimum, not just a stationary point.

This is exactly the object I wanted. Its minima are the true vacua, fluctuations and all, and — this is the part I like — it surveys *all* candidate vacua at once. I don't have to commit in advance to expanding around a particular point and hope it's the right one. The symmetric configuration and any asymmetric one are all just values of `φc`; I compute one function and read off which value wins. If I had instead done ordinary perturbation theory around `φ=0`, I'd have built the whole series on a point that might be unstable, and I'd never see a minimum that the loops put somewhere else.

In the tree approximation `V` is just `V_cl`, the negative of the non-derivative terms in `L`. Everything beyond that — every shift of the vacuum from where the classical potential puts it — lives in the closed-loop graphs. So the entire question reduces to: compute the loop corrections to `V`, see what they do to the shape.

Now the honest difficulty. To get `V` exactly I'd have to sum an infinite set of Feynman diagrams. I can't. I need a truncation that is *sensible* — and "sensible" has a specific meaning here that I should pin down, because the obvious choice is wrong.

The obvious choice is: expand in the coupling, order by order. But think about what an expansion in the coupling does when I'm trying to compare vacua. If I shift the field, `φ = φ' − ⟨φ⟩`, the split of `L` into "free" and "interacting" parts changes — the shift generates new quadratic and cubic terms. An expansion organized by powers of the coupling is sensitive to that split; expanding about `0` and expanding about `⟨φ⟩` give differently-organized series. That's exactly the wrong property for an object whose whole job is to be evaluated at every `φc` impartially.

What I want is an expansion that doesn't care how I split the Lagrangian. Let me find one. Put a bookkeeping parameter `a` in front of the whole Lagrangian: `L(φ,a) = a⁻¹ L(φ)`. (It's playing the role of `ħ`.) Trace the power of `a` through any 1PI graph. The propagator is the inverse of the quadratic operator, which carries `a`, so each internal line gives one power of `a`; each vertex comes from `a⁻¹L`, so each vertex gives `a⁻¹`. A graph with `I` internal lines and `V` vertices carries `a^{I−V}`. And the number of loops is `L = I − V + 1` — every internal line is an integration momentum, every vertex a delta function, and one delta function survives as overall momentum conservation. So `I − V = L − 1`, and the power of `a` is `a^{L−1}`. The expansion in `a` is the **loop expansion**.

That's the property I needed. The number of loops in a graph is `I−V+1`, which only counts the topology; it is completely blind to how I divided `L` into free and interacting pieces, and blind to field shifts like `φ → φ − ⟨φ⟩`. So the loop expansion of `V` preserves exactly the virtue I care about: I can evaluate the truncated `V` at any `φc`, having shifted or not, and the loop order of each term is unambiguous. The fact that `a` is actually `1`, not a small number, doesn't matter — I'm not using `a` as a smallness parameter (for small couplings the loop expansion is anyway no worse than ordinary perturbation theory, since `n` loops includes all graphs of order `n` and lower in the couplings). I'm using it because loop order is shift-invariant. Tree level is the classical potential; one loop is the first quantum correction; that's where I'll work.

Let me do the simplest possible case first to learn the mechanics, before touching the photon: a single massless self-interacting scalar, `L = ½(∂φ)² − (λ/4!)φ⁴`, plus the renormalization counterterms `½A(∂φ)² − ½Bφ² − (1/4!)Cφ⁴`. One remark before I start: even though I'm calling this theory "massless," I have to keep a mass-renormalization counterterm `B`. There's no symmetry that forbids a bare mass for a single real scalar, so a mass counterterm is generated whether I like it or not; "massless" is going to be a *condition I impose* — that the renormalized mass vanish — not something the structure hands me for free.

Tree level: only `V = (λ/4!)φc⁴`. One loop: the `n`-th derivative of `V` is the sum of 1PI graphs with `n` zero-momentum external legs, so at one loop I need the polygon graphs — a single closed scalar loop with `n` vertices, each vertex attaching two external `φc` legs and continuing the loop. The `n`-gon has `2n` external legs. Each vertex brings `−iλ/2` (the `½` is a Bose factor: the two external lines at a vertex are interchangeable, so the `1/4!` of the quartic vertex is only partly cancelled). The internal propagators give `[i/(k²+iε)]ⁿ`. And there's an overall combinatoric `1/2n`: a cyclic factor `1/n` because rotating the `n`-sided polygon doesn't make a new Wick contraction, and a `1/2` from reflection. So the one-loop term is

```
i ∫ d⁴k/(2π)⁴ Σ_{n=1}^∞ (1/2n) [ ½λφc²/(k²+iε) ]ⁿ ,
```

and adding tree and counterterms,

```
V = (λ/4!)φc⁴ + ½Bφc² − (1/4!)Cφc⁴ + i∫ d⁴k/(2π)⁴ Σ_{n=1}^∞ (1/2n)[½λφc²/(k²+iε)]ⁿ .
```

At first sight this looks hideous. Every single term in that sum is infrared-divergent: at `k→0`, `[½λφc²/k²]ⁿ` blows up like `k^{−2n}`, worse and worse as `n` grows. If I tried to compute this function by its power series in `φc² ` at `φc=0` — which is what computing radiative corrections "at zero momentum" amounts to — I'd get a sequence of ever-worse infrared divergences. This is the same disaster I'd hit trying to compute the massless propagator's radiative corrections at `p²=0`: nontrivial behavior like `p²ln p²` shows up, and the naive power series chokes.

But I shouldn't compute term by term. Let me sum the series first. With `x = ½λφc²/(k²+iε)`,
```
Σ_{n=1}^∞ xⁿ/(2n) = ½ Σ_{n=1}^∞ xⁿ/n = −½ ln(1−x) .
```
Rotate the integral to Euclidean space (`k² → −k_E²`, and the `iε` lets me do it), so `1 − x → 1 + λφc²/2k_E²`, and the `i` out front becomes the right real measure. The whole one-loop piece collapses to a single integral:

```
V = (λ/4!)φc⁴ + ½Bφc² − (1/4!)Cφc⁴ + ½∫ d⁴k_E/(2π)⁴ ln(1 + λφc²/2k_E²) .
```

Look at what happened. The infrared catastrophe is gone. Each *term* in the sum diverged at `k=0`, but the *sum* is `ln(1 + λφc²/2k²)`, which at small `k` behaves like `ln(λφc²/2k²)` — only a logarithmic singularity, and as a function of `φc` it's just a log singularity at `φc=0`, which is perfectly integrable and harmless. The resummation cured the infrared. This is the same trick that saves the massless propagator: I can't expand at zero momentum, but I don't have to — stay away from it. Here I stay away from `φc=0`. There's also an apparent logarithmic singularity in the coupling lurking, but I'll see in a moment it gets eaten by the counterterms.

The remaining integral is ultraviolet-divergent — `∫d⁴k ln(1+ .../k²)` grows. Cut it off at `k² = Λ²`. Doing the integral:

```
V = (λ/4!)φc⁴ + ½Bφc² − (1/4!)Cφc⁴ + λΛ²φc²/64π² + (λ²φc⁴/256π²)[ln(λφc²/2Λ²) − ½] .
```

So the cutoff threw up a quadratic divergence `λΛ²φc²/64π²` and a quartic-with-a-log piece. Now fix the counterterms by renormalization conditions. I want the *renormalized* mass to vanish — that's what "massless" means: `d²V/dφc²|₀ = 0`. The quartic-and-log term contributes nothing to the curvature at the origin (it's `O(φc²ln)`, vanishes); the quadratic terms give `B + λΛ²/32π²`. Setting that to zero fixes `B` to cancel the quadratic divergence. Good — the `Λ²` is absorbed.

Now the coupling. The natural condition would be `d⁴V/dφc⁴|₀ = λ`. But that fourth derivative *doesn't exist* at the origin — the `φc⁴ ln φc²` term has a fourth derivative `∝ ln φc² → −∞` as `φc→0`. The log singularity strikes again. I can't define the coupling at zero field. Fine — do exactly the analogue of what's done in momentum space: define the coupling at an arbitrary off-singularity point. Introduce a scale `M` with the dimensions of a mass, and impose

```
d⁴V/dφc⁴|_{φc=M} = λ ,
```

and the field-normalization condition at the same scale, `Z(M) = 1`. This fixes `C`. `M` is completely arbitrary — it's the off-shell subtraction point, dressed up in field space instead of momentum space. Any nonzero `M` is as good as any other.

Put it all together and the cutoff drops out entirely (it had better — the theory is renormalizable):

```
V = (λ/4!)φc⁴ + (λ²φc⁴/256π²)[ ln(φc²/M²) − 25/6 ] .
```

This is the one-loop effective potential of the massless self-interacting scalar. Let me check the arbitrariness of `M`: differentiate the coupling condition at `M'` instead, and `λ' = λ + (3λ²/32π²)ln(M'²/M²) + O(λ³)`. Reparametrize `V` in terms of `λ'` and `M'` and it comes back to the identical functional form. So changing `M` is just a change of definition of the coupling — a reparametrization, not a change of physics. Good.

Now stare at the shape and ask the question I started with: do the loops create a new minimum away from the origin? The two `φc⁴` terms have *opposite-signed* logs: for `φc < M`, `ln(φc²/M²) < 0`, so the one-loop term is negative; for `φc > M` it's positive. So the effective potential tilts. To get a stationary point off the origin I'd balance the tree term `(λ/4!)φc⁴` against the loop term `(λ²/256π²)φc⁴ ln(φc²/M²)`. Setting `dV/dφc = 0` (away from the origin) means balancing a term of order `λ` against a term of order `λ²ln(φc/M)`. For these to cancel I need `ln(φc/M) ∼ 1/λ`, i.e. `φc/M ∼ e^{1/λ}` — enormous, exponentially large for small `λ`.

That's a wall. The one-loop approximation is only trustworthy where the logarithm is `O(1)`; out at `φc/M ∼ e^{1/λ}` the log is `∼ 1/λ`, so `λ²ln` is the same size as `λ`, and two-loop terms `λ³ln²` are the same size again, and everything I dropped is as big as everything I kept. The "new minimum" sits squarely outside the region where I'm allowed to trust the calculation. In this pure `λφ⁴` theory I have to reject it as an artifact of the truncation. The radiative correction *wants* to make a minimum, but it can only do so by balancing `λ` against `λ²ln`, which forces a huge log, which kills the approximation. The single coupling defeats me.

So the pure scalar can't do it. What was the obstruction, precisely? It was that the only thing the loop had to balance the tree term against was *itself one order higher in the same coupling*, `λ²` against `λ`, and bridging one power of `λ` costs one power of `1/λ` in the log. If there were a *second*, independent coupling, the loop correction could be the same numerical size as the tree term *without* a large log — because two different couplings can be comparable in magnitude even when each is small.

That's why I went to scalar electrodynamics in the first place. There are two couplings, `λ` and `e`, and the photon runs in loops too. Let me redo the computation with the photon in.

```
L = −¼Fμν² + ½(∂μφ₁ − eAμφ₂)² + ½(∂μφ₂ + eAμφ₁)² − (λ/4!)(φ₁²+φ₂²)² + counterterms .
```

By the U(1) symmetry `V` can depend on the two real fields only through `φc² = φ₁c² + φ₂c²`, so I can take all external legs to be `φ₁`. Pick the Landau gauge — the photon propagator is transverse, `Dμν = −i(gμν − kμkν/k²)/(k²+iε)`. Why this gauge? Because for the polygon graphs the external scalars carry zero momentum, so the internal photon momentum equals the internal scalar momentum, and when it's contracted with the longitudinal seagull vertex the `kμkν/k²` piece kills it — whole classes of graphs vanish. In Landau gauge only three kinds of polygons survive: `φ₁` running around the loop, `φ₂` running around the loop, and the photon running around the loop. Each has exactly the same structure as the scalar polygons I just summed, with the appropriate vertex factors and an extra factor of three for the photon's polarizations (the trace of the transverse numerator in four dimensions is `3`).

Doing the same geometric sum and renormalizing exactly as before, with the same subtraction scale `M`:

```
V = (λ/4!)φc⁴ + (5λ²/1152π² + 3e⁴/64π²) φc⁴ [ ln(φc²/M²) − 25/6 ] .
```

The `5λ²/1152π²` is the scalar-loop coefficient — and I can see where it comes from: along the background the radial fluctuation has mass-squared `λφc²/2` and the would-be Goldstone (angular) fluctuation has mass-squared `λφc²/6`, and `(1/64π²)[(λ/2)² + (λ/6)²] = 5λ²/1152π²`. The `3e⁴/64π²` is the photon loop, `(3/64π²)(e²)²`. The `3e⁴` matters; the `5λ²` is about to disappear.

Here's the point I was missing in the pure scalar theory. Now I can balance the tree term `(λ/4!)φc⁴`, order `λ`, against the *photon* loop term `(3e⁴/64π²)φc⁴ ln`, order `e⁴`. Because `λ` and `e` are independent couplings, there is nothing in the world that forces `λ` to be much bigger or smaller than `e⁴`. In fact renormalizability *forces* `λ` to be at least of order `e⁴`: the quartic scalar self-interaction is needed to cancel the divergence in scalar Coulomb scattering, a process that is itself order `e⁴`. So `λ ∼ e⁴` is exactly what I should expect. And if `λ ∼ e⁴`, then `λ² ∼ e⁸`, and the `5λ²/1152π²` term is order `e⁸` — negligible against the `3e⁴/64π²` photon term, and in fact I'm *obliged* to drop it for consistency, since the two-loop electromagnetic corrections I haven't computed are also `O(e⁶, e⁸)`. Drop it:

```
V = (λ/4!)φc⁴ + (3e⁴/64π²)φc⁴ [ ln(φc²/M²) − 25/6 ] .
```

Now find the minimum. `dV/dφc = 0` away from the origin:
```
0 = (λ/6 − 11e⁴/16π²)φc³ + (3e⁴/16π²)φc³ ln(φc²/M²) .
```
Here is the freedom I've been saving. `M` is arbitrary; let me choose it to be the location of the minimum itself, `M = ⟨φ⟩`. Then `ln(⟨φ⟩²/M²) = 0`, and the minimum condition collapses to a pure algebraic relation:
```
λ/6 − 11e⁴/16π² = 0  ⟹  λ = 33 e⁴ / 8π² .
```
And it's self-consistent: with `M = ⟨φ⟩` the troublesome logarithm vanishes right at the vacuum, so I'm evaluating the one-loop result exactly where it's most reliable, not out at `e^{1/λ}`. The balance `λ ∼ e⁴` needs the log to be `O(1)`, not `O(1/λ)` — there is no large logarithm forced on me, because I balanced two *different* couplings of comparable size, not one coupling against its own square. The wall from the pure scalar theory is gone precisely because the second coupling let me match magnitudes without paying in logs.

And now something I didn't ask for falls out. I started with two free dimensionless parameters, `e` and `λ`. The minimum condition just *determined* `λ` in terms of `e`. I'm left with one dimensionless parameter, `e`, and one *dimensionful* one, `⟨φ⟩` — the location of the vacuum, which sets the scale of the theory. A dimensionless coupling has been traded for a mass scale. The original Lagrangian had no dimensionful constant anywhere; it was exactly scale-invariant; and yet the quantum theory has chosen a scale. The arbitrary renormalization mass `M` is what carried the scale in — but `M` is arbitrary, so it's not that `M` set the scale; it's that for a *fixed* theory a change in `M` is exactly compensated by a change in the numerical value of the dimensionless coupling. The dimensionless `λ` and the dimensionful `⟨φ⟩` are not independent; the physics has one of each. This trading of a dimensionless coupling for a dimensionful vacuum value is forced on a massless theory, and it's the inevitable face of spontaneous symmetry breaking when there's no scale to begin with. Call it dimensional transmutation.

Let me write the final potential with the determined `λ` and `M = ⟨φ⟩` plugged in:
```
V = (3e⁴/64π²) φc⁴ [ ln(φc²/⟨φ⟩²) − ½ ] .
```
Parametrized entirely by `e` and `⟨φ⟩`; `λ` and `M` have vanished into them. Now read off the spectrum. The scalar mass is the curvature at the vacuum:
```
m²(S) = V''(⟨φ⟩) = 3e⁴⟨φ⟩²/8π² .
```
The would-be Goldstone boson — the angular mode — is eaten by the photon, which gets the conventional Higgs mass:
```
m²(V) = e²⟨φ⟩² .
```
The U(1) is spontaneously broken; the theory is now a massive vector plus a massive scalar. And the ratio:
```
m²(S)/m²(V) = 3e²/8π² = (3/2π)(e²/4π) .
```
This is the thing I find genuinely striking. A small *ratio* of masses — not a small mass *difference*, a small ratio — has emerged as a natural consequence of a small coupling, computable as a power series in `e`. The depth of the new vacuum below the origin is
```
V(⟨φ⟩) = −3e⁴⟨φ⟩⁴/128π² < 0 ,
```
so the asymmetric vacuum really is lower than the symmetric one. The loops created the breaking. Nothing was put in by hand; the Lagrangian stayed massless and symmetric throughout.

But I was burned once already by trusting a loop-induced minimum, so let me be careful about *which* features I believe. Near `⟨φ⟩` (with `M = ⟨φ⟩`) the logarithm `ln(φc/⟨φ⟩)` is small, so the one-loop result is reliable there — I trust the new minimum. What about the origin? Near `φc = 0` the logarithm is large and negative; one-loop is *not* reliable there, and naively it looks like the correction turns the origin into a maximum with a phony minimum further out — exactly the pathology that made me reject the pure-scalar minimum. I need to check the origin honestly, and for that the logs have to be resummed.

This is where the arbitrariness of `M` becomes a tool rather than a nuisance. Physics is independent of `M`; a small change in `M` is compensated by a small change in `λ` and a rescaling of the field. That invariance is the renormalization-group equation. Let me exploit it. Define the running coupling `λ'(t)` along `t = ln(φc/M)`, governed by `dλ'/dt = β̄(λ')`, with `β̄` the beta function. For the self-interacting scalar the one-loop beta function is `β̄ = 3λ²/16π²` (and the anomalous dimension `γ̄ = 0` at this order, so `Z` doesn't run). Solve the flow:
```
dλ'/dt = 3λ'²/16π²  ⟹  λ'(t) = λ / (1 − 3λt/16π²) .
```
The renormalization-group-improved `V⁗` (the dimensionless fourth derivative) is just `λ'(t)`. Now `t = ln(φc/M)` is *negative* near the origin (small `φc`), and there `λ'(t) = λ/(1 + 3λ|t|/16π²)` stays *small* — it never blows up for `t<0`. So the improved result is trustworthy all the way down to the origin, and it says `V⁗ > 0` and small there: the origin is a genuine maximum, no hidden minimum. My earlier mistrust was justified for the right reason — the unimproved one-loop couldn't be trusted near the origin — but the improved calculation *can* be, and it confirms the qualitative picture. (Also, no graph can move `V(0)` itself, which stays pinned at zero, since at `φc=0` there's no scale for a correction to be made of.)

For scalar electrodynamics I do the same with both couplings running. The anomalous dimension is `γ̄ = 3e²/16π²`, the gauge coupling runs as `β̄_e = e³/48π²` so `e'² = e²/(1 − e²t/24π²)`, and `λ` runs with its own one-loop beta function. The upshot is that for small `e` I can move `λ` away from `O(e⁴)` to any other small value by a change in `M` that costs only an `O(e⁴)` change in `e` — meaning the restriction "`λ` must be of order `e⁴`" that I imposed to drop the `5λ²` term was, after the renormalization-group analysis, no restriction at all. The effective potential develops its asymmetric minimum, and spontaneous breaking occurs, for arbitrary small `e` and `λ`.

Now I want to be sure the things I found at one loop aren't accidents of one loop. Two worries: did the infrared divergences really go away to *all* orders, and did the dependence on `ln(coupling)` really disappear to all orders, or did I just get lucky at one loop?

Take the infrared first, in the self-interacting scalar. Classify the graphs by vertex type: a "type-`n`" vertex is one with `n` internal lines attached (the rest of its four legs being external `φc`), so for these 1PI graphs the vertices are type-2, type-3, or type-4. Let `V_n` be the number of type-`n` vertices. Counting line-ends, `2I = 2V₂ + 3V₃ + 4V₄`, so the number of loops `L = I − V + 1 = ½V₃ + V₄ + 1`. For *fixed* `L` there are only finitely many "prototype" graphs — the ones with *no* type-2 vertices. Every other `L`-loop graph is obtained from a prototype by sticking type-2 vertices onto its internal lines. A type-2 vertex inserted on a line is just `(−½λφc²)` times another propagator, so inserting arbitrarily many of them onto one internal line is a geometric series:
```
i/(k²+iε) → Σ_m [½λφc²/(k²+iε)]^m · i/(k²+iε) = i/(k² − ½λφc² + iε) .
```
So the computational rule to all orders is: take the finitely-many prototype graphs at `L` loops, and in each, replace every internal scalar propagator `i/(k²+iε)` by `i/(k² − ½λφc² + iε)`. The mass-shift `½λφc²` in the denominator is exactly what regulates `k=0`: the per-graph infrared divergence becomes, in the resummed propagator, a singularity at `φc=0` and nothing worse — the same structure as one loop, now at every order. (For scalar electrodynamics the photon line gets the analogous shift, `e²φc²` in its denominator.) The infrared cure is general.

Now the logs of the coupling, again in the self-interacting scalar. In any prototype graph rescale every loop momentum `k = λ^{1/2}k'`. Then each shifted propagator `1/(k² − ½λφc²) = λ⁻¹/(k'² − ½φc²)` pulls out `λ⁻¹`, and each loop measure `d⁴k = λ²d⁴k'` pulls out `λ²`. Count the explicit powers of `λ` too — one per vertex. An `L`-loop graph then carries `φc⁴ · f(φc/M) · λ^{V} · λ^{2L} · λ^{−I}`, and using `V − I = 1 − L` this is `φc⁴ f(φc/M) λ^{L+1}`. So the `L`-loop contribution to `V` is a *pure power* `λ^{L+1}` times a function of `φc/M` — no `ln λ` anywhere. The one-loop being `∝ λ²`, two-loop `∝ λ³`, and so on: the would-be logarithms of the coupling are absent to all orders. (This rescaling is legitimate precisely because I renormalized at a field-space point `φc = M`, not at a fixed Euclidean momentum — had I subtracted at fixed external momentum the rescaling would have moved the subtraction point and spoiled it.) For scalar electrodynamics the same argument runs with `λ → e²` after determining `λ` in terms of `e`, by a trick of shifting the renormalization point to the minimum; the `L`-loop contribution is `∝ (e²)^{L+1}`. Both surprises of the one-loop calculation are structural, not accidents.

One more thing I want to understand, because it connects to the only precedent I know of for "compute the one-loop effective Lagrangian in a constant background and see what it does." Let me ask what happens if I *do* put a mass term in — first a positive `μ²`, then a negative one — and take the massless limit to make sure my massless answers are the smooth limit of something sensible. With a mass the scalar propagator is `1/(k² + μ²)` in Euclidean, never vanishing, so there's no infrared problem and I could even renormalize at zero field. Computing the one-loop `V` and taking `μ → 0` reproduces the massless result smoothly — good, the massless theory is the honest limit. But the more interesting case is *negative* `μ²`. Then `k² + μ²` can vanish at Euclidean `k`, and I cannot drop the `iε`: the effective potential develops an **imaginary part**,
```
Im V = −(1/64π²)[(μ² + ½λφc²)² θ(−μ² − ½λφc²) − μ⁴] ,
```
which the real counterterms `B` and `C` cannot cancel — and must not, on pain of making the Lagrangian non-real. This is not a calculational mistake. I've seen this before: the Euler-Heisenberg / Schwinger effective Lagrangian of QED in a constant external field is real for a magnetic field but acquires an imaginary part for an electric field, and that imaginary part means the vacuum decays into electron-positron pairs. The analogy is exact: just as a propagator has an imaginary part at momenta where the particle is kinematically unstable, here the *vacuum itself* is kinematically unstable — the negative mass term makes the symmetric configuration want to fall apart — and `V` records that by going imaginary. For negative `μ²` the right renormalization condition is on the *real* part, `Re V''|₀ = μ²`. The imaginary part doesn't touch the real, asymmetric vacuum where the field actually sits; at the true vacuum the step function's argument is positive and `V` is real. So even the massive theory tells a consistent story, and it tells it through the same imaginary-part mechanism that Euler and Heisenberg found.

There's a small bonus in the positive-`μ²` massive scalar electrodynamics that's worth noting. Computing `V` there gives `½μ²φc² + (3e⁴/64π²)φc⁴[ln(φc²/m²) − ½]`, and for sufficiently small positive `μ²` this has *two* minima — one at the origin, one out near `m`, the latter lower. So spontaneous breaking occurs even for positive `μ²`. That seems to contradict the textbook statement that scalar electrodynamics with `μ² > 0` doesn't break. The resolution is that the textbook statement implicitly fixes `λ` at a particular value, and the value of `λ` depends on the arbitrary `M`; differentiating my `V` four times shows the relevant `λ` here is negative when `M` is taken near `μ`. So it's not the conventional positive-coupling theory at all — no contradiction, just two different parametrizations of the same physics, and the resolution is again the `M`-dependence of the dimensionless coupling.

Finally, the whole thing generalizes. The one-loop polygon sum I did for the scalar and the photon is, structurally, the trace-log of the field-dependent inverse propagator — the functional determinant of the operator governing the fluctuations about the background `φc`. For a general massless renormalizable theory, the one-loop effective potential is
```
V₁ = (1/64π²) [ Tr W²(φc) ln W(φc) + 3 Tr M⁴(φc) ln M²(φc) − Tr (mm†)²(φc) ln (mm†)(φc) ] ,
```
where `W(φc) = ∂²V₀/∂φa∂φb` is the field-dependent scalar mass-squared matrix, `M²(φc)` the field-dependent vector mass-squared matrix (`M²_ab = g_a g_b (T_a φ, T_b φ)`, the gauge generators acting on the background), and `mm†` the field-dependent fermion mass-squared. The `3` on the gauge term is the polarization count in Landau gauge; the minus sign on the fermion term is the closed-fermion-loop sign. In words: each field contributes `± (2J+1)/(64π²)` times `(field-dependent mass²)² ln(field-dependent mass²)`, with the plus for bosons and the minus for fermions. The whole structure of the massless scalar-electrodynamics result is just this formula with `W = λφc²/2` (radial and Goldstone modes) and `M² = e²φc²`. Apply it to an `SU(2)` triplet of gauge fields coupled to a scalar isovector and the same thing happens — radiative corrections break the symmetry, an unbroken `U(1)` (the photon) survives, the charged vectors eat scalars and get masses, and `m²(S)/m²(V) = (3/π)(e²/4π)`, twice the Abelian ratio because the potential is twice as large. Apply it to the electroweak gauge theory of leptons and the would-be scalar mass is predicted from the vector masses, `m²(S) = (3/32π²)[2g²m²(W) + (g²+g'²)m²(Z)]`. The mechanism — quantum loops, with no asymmetric input, choosing an asymmetric vacuum and a scale — is general.

Let me state the chain in one breath. The classical potential is the wrong object because it ignores fluctuation energy; the right object is the effective potential, whose minima are the true vacua and which surveys all candidate vacua at once; the right truncation is the loop expansion, because loop order is invariant under field shifts; one loop is a sum of polygon graphs that I resum into a single integral, which cures the infrared and, after renormalizing the coupling at an arbitrary off-singularity scale, gives a cutoff-free `V` with a tilted, log-modified quartic; in a single coupling the loop-induced minimum needs an exponentially large log and is an artifact, but with a *second* independent coupling the loop term balances the tree term at comparable magnitude with no large log, so the minimum is real; the minimum condition determines `λ` in terms of `e`, trading a dimensionless coupling for a dimensionful vacuum value (dimensional transmutation); the spectrum is a massive vector and a massive scalar with `m²(S)/m²(V) = (3/2π)(e²/4π)`; the renormalization group confirms the origin is a true maximum and removes the restriction on `λ`; and the same trace-log structure generalizes to any massless gauge theory. A massless, scale-free Lagrangian breaks its own symmetry, by quantum mechanics alone.

```python
import sympy as sp

phi, M, lam, e, vev = sp.symbols('phi M lambda e v', positive=True)
pi = sp.pi

# --- one-loop effective potential, massless self-interacting scalar (the template) ---
# tree + resummed-polygon loop, cutoff removed by renormalizing at scale M:
V_scalar = lam/sp.factorial(4)*phi**4 \
         + lam**2*phi**4/(256*pi**2)*(sp.log(phi**2/M**2) - sp.Rational(25,6))
# the loop-induced minimum needs ln(phi/M) ~ 1/lam  ->  outside one-loop validity: an artifact.

# --- one-loop effective potential, massless scalar electrodynamics ---
# radial mass^2 = lam*phi^2/2, Goldstone mass^2 = lam*phi^2/6, photon mass^2 = e^2*phi^2:
V_sqed = lam/sp.factorial(4)*phi**4 \
       + (5*lam**2/(1152*pi**2) + 3*e**4/(64*pi**2))*phi**4*(sp.log(phi**2/M**2) - sp.Rational(25,6))
# with lam ~ e^4 (forced by renormalizability), drop the order-e^8 scalar-loop term:
V_sqed = lam/sp.factorial(4)*phi**4 \
       + 3*e**4/(64*pi**2)*phi**4*(sp.log(phi**2/M**2) - sp.Rational(25,6))

# --- minimum: choose M = <phi> so the log vanishes there; read off lambda(e) ---
Vp = sp.diff(V_sqed, phi)
lam_min = sp.solve(sp.Eq(lam/6 - 11*e**4/(16*pi**2), 0), lam)[0]   # = 33 e^4 / (8 pi^2)

# final potential parametrized by e and <phi> alone (dimensional transmutation):
V_final = V_sqed.subs({lam: lam_min, M: vev})
V_final = sp.simplify(V_final)        # = 3 e^4/(64 pi^2) * phi^4 (ln(phi^2/<phi>^2) - 1/2)

# --- spectrum at the broken vacuum ---
mS2 = sp.simplify(sp.diff(V_final, phi, 2).subs(phi, vev))   # 3 e^4 <phi>^2 / (8 pi^2)
mV2 = e**2 * vev**2                                          # e^2 <phi>^2 (photon eats Goldstone)
ratio = sp.simplify(mS2/mV2)                                # 3 e^2 / (8 pi^2) = (3/2pi)(e^2/4pi)
depth = sp.simplify(V_final.subs(phi, vev))                 # -3 e^4 <phi>^4 / (128 pi^2)

# --- general one-loop master formula (functional determinant of the fluctuation operator) ---
# V1 = (1/64 pi^2) [ Tr W^2 ln W + 3 Tr M^4 ln M^2 - Tr (mm†)^2 ln(mm†) ]
#   W   = scalar field-dependent mass^2 matrix
#   M^2 = vector field-dependent mass^2 matrix (M^2_ab = g_a g_b (T_a phi, T_b phi))
#   mm† = fermion field-dependent mass^2 ;  + for bosons, - for fermions, weighted by (2J+1)
```
