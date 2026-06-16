Let me start from what actually goes wrong. I have a smooth convex-concave saddle problem, min over x and max over y of L(x,y), and I bundle the two variables into z=(x,y) and the gradients into one operator G(z) = [∇_x L ; -∇_y L], with the sign flip on the y-block built in so that a saddle point is just a zero of G. Because L is convex-concave, G is monotone — ⟨G(z1)-G(z2), z1-z2⟩ ≥ 0 — and I'll assume it's R-Lipschitz. What I want to make small is the gradient norm ‖∇L(z)‖² = ‖G(z)‖²; that's the thing that's actually zero at a solution and is meaningful even when I drift out of the convex-concave world into the GAN-like games I really care about. The duality gap is the classical measure, but it needs bounded domains to even be finite and I can't measure it easily, so I'm committing to the gradient norm and to the unconstrained problem on all of R^n × R^m.

The obvious thing to try is to descend in x and ascend in y simultaneously: z^{k+1} = z^k − α G(z^k). Let me sanity-check it on the simplest convex-concave problem there is, L(x,y)=xy. Then G(z) = [y, −x] = [[0,1],[−1,0]] z, which is a 90-degree rotation of z. So each step pushes me perpendicular to my current position vector. In continuous time ż = −G(z) is ż = [−y, x], and d/dt(½‖z‖²) = ⟨z, ż⟩ = ⟨[x,y],[−y,x]⟩ = −xy + xy = 0 — the distance to the origin is exactly conserved, the flow just circles the saddle point forever and never arrives. And the explicit Euler discretization is worse than neutral: ‖z^{k+1}‖² = ‖z^k − αG(z^k)‖² = ‖z^k‖² − 2α⟨z^k, G(z^k)⟩ + α²‖G(z^k)‖² = ‖z^k‖²(1+α²), because ⟨z^k, G(z^k)⟩ = ⟨[x,y],[y,−x]⟩ = 0. So the naive method spirals strictly outward and diverges on the easiest possible instance. That's the wall I have to get past before anything else: the rotational, cycling part of the dynamics.

The cure for the cycling is already known: extragradient. Korpelevich's idea is to not trust the gradient at where I am. Take a tentative look-ahead step to a predicted point, evaluate G there, and apply that corrected gradient back at the original point: z^{k+1/2} = z^k − αG(z^k), then z^{k+1} = z^k − αG(z^{k+1/2}). The intuition on the rotation: the look-ahead gradient G(z^{k+1/2}) "sees around the corner," picking up the curvature of the circular motion, so the corrected step bends inward instead of flying off tangentially. Let me confirm it actually contracts. With w = z − αG(z) and z^+ = z − αG(w), I want to compare ‖z−z*‖² with ‖z^+−z*‖². Write both through the common point w:
‖z−z*‖² = ‖z−w‖² + 2⟨z−w, w−z*⟩ + ‖w−z*‖²,
‖z^+−z*‖² = ‖z^+−w‖² + 2⟨z^+−w, w−z*⟩ + ‖w−z*‖².
Subtract, and the ‖w−z*‖² cancels:
‖z−z*‖² − ‖z^+−z*‖² = ‖z−w‖² − ‖z^+−w‖² + 2⟨z−z^+, w−z*⟩.
Now z−z^+ = α G(w), so the last term is 2α⟨G(w), w−z*⟩ = 2α⟨G(w)−G(z*), w−z*⟩ ≥ 0 by monotonicity, since G(z*)=0. Drop it as a nonnegative bonus. And ‖z^+−w‖² = ‖(z−αG(w)) − (z−αG(z))‖² = α²‖G(z)−G(w)‖² ≤ α²R²‖z−w‖² by Lipschitzness. So
‖z−z*‖² − ‖z^+−z*‖² ≥ (1−α²R²)‖z−w‖² = (1−α²R²)α²‖G(z)‖²,
using ‖z−w‖ = α‖G(z)‖. As long as α < 1/R this is a genuine, strictly positive decrease at every step, scaled by ‖G(z)‖². Beautiful — extragradient defeats the cycling.

But now I measure how fast it kills the gradient norm. Sum that inequality over i = 0..k. The left side telescopes to ‖z^0−z*‖² − ‖z^{k+1}−z*‖² ≤ ‖z^0−z*‖², and the right side is (1−α²R²)α² Σ_{i=0}^k ‖G(z^i)‖² ≥ (1−α²R²)α²(k+1) min_{i≤k}‖G(z^i)‖². So
min_{i=0..k} ‖G(z^i)‖² ≤ ‖z^0−z*‖² / [(1−α²R²)α²(k+1)] = O(R²/k).
Two things bother me about this. First, it's O(1/k), not faster. Second, and more annoying, it's a *best-iterate* bound: it controls the smallest gradient norm I've seen, with no promise the *last* iterate is the good one, so I'd have to carry the best-so-far. For GANs I want the last iterate to actually be near a solution. And there's a hard wall here — for the natural stationary algorithm class that contains EG, you provably cannot beat O(1/k) on the last-iterate gradient norm. Optimism (Popov) reuses the previous gradient as the prediction to save an evaluation, but it lives in the same O(1/k) class. So extragradient gives me a constant step and anti-cycling, but it's stuck at O(1/k) best-iterate. Wall.

What makes the *last* iterate converge, and selects a particular solution? The Halpern iteration. For a nonexpansive map T, instead of just iterating u ← T(u), you mix in a pull back toward the fixed starting anchor u^0: u_{k+1} = λ_{k+1} u^0 + (1−λ_{k+1}) T(u_k). If λ_k → 0 but the pulls don't die too fast, this converges — to the fixed point of T nearest u^0 in ℓ2. It's an implicitly regularized iteration: the anchor regularizes toward u^0 and quietly picks the minimum-norm solution. And it has a clean residual rate: with λ_k = 1/(k+1), the fixed-point residual ‖T(u_k) − u_k‖ = 2‖u^0 − u*‖/(k+1), an O(1/k) — and crucially that's a *last-iterate* statement. My G is monotone and Lipschitz, so I can build a nonexpansive T out of it (a resolvent, or an averaged step) whose residual controls ‖G‖; the Halpern residual rate then transfers to my gradient norm. So anchoring buys me exactly the property EG lacks: last-iterate convergence with an honest rate.

The natural move is to transplant the anchor onto plain gradient steps, which is what SimGD-with-anchoring does: z^{k+1} = z^k − ((1−p)/(k+1)^p) G(z^k) + ((1−p)γ/(k+1))(z^0 − z^k). Now I have a last-iterate gradient-norm rate, O(1/k^{2−2p}), and the anchor damps the bilinear rotation. But look at the gradient step size: it's forced to *diminish* like (1−p)/(k+1)^p, and p has to stay above 1/2, so the best I can push the rate to is just under O(1/k) — and the shrinking step makes the thing crawl. So anchoring gives me last-iterate convergence and solution selection but is shackled to a vanishing step and can't reach O(1/k²). Another wall, complementary to EG's.

Let me lay the two failures side by side, because the complementarity is the whole clue. Extragradient: defeats cycling, allows a *constant* step, but best-iterate, stuck at O(1/k). Anchoring: last-iterate, selects the solution, but needs a *shrinking* step and stalls below O(1/k). Each one supplies precisely the ingredient the other is missing. EG's look-ahead is what gives me a per-step decrease at a *fixed* step size — the (1−α²R²)α²‖G‖² descent — which is the very thing SimGD-A had to buy with a decaying step. Anchoring is what makes the last iterate, not just the best one, go to zero. So what if I don't choose between them — what if I run extragradient but plant the anchor pull inside *both* of its half-steps, and keep the step size constant?

z^{k+1/2} = z^k + β_k(z^0 − z^k) − α G(z^k),
z^{k+1}   = z^k + β_k(z^0 − z^k) − α G(z^{k+1/2}).

That's it as a candidate: the extragradient predictor-corrector, but every gradient step is offset by an anchoring term β_k(z^0 − z^k) pulling back toward the start, with the same anchor offset in both lines (note the offset uses z^k, the current point, in both the predictor and the corrector — I anchor relative to where I am, not relative to the half-step w). When β_k = 0 this is exactly extragradient; the anchor is the only new thing. Now I have to figure out the anchoring schedule β_k, prove it works, and find out how fast.

Before I prove anything, let me get the schedule from the continuous picture, because that's where the design choice lives. The anchored flow is ż(t) = −G(z(t)) − β(t)(z(t) − z^0): the operator drives toward a zero of G, and a time-decaying spring pulls back toward the start. There are two competing speeds in β(t). The contracting speed: the spring alone, ż = −β(t)(z − z^0), contracts the iterate toward z^0, and that contraction is what gives stability and kills the rotation. The vanishing speed: but I don't want to converge to z^0, I want a zero of G, so the spring *must* eventually die. If β vanishes too fast the flow isn't contracted enough and reverts to the slow cycling; if it vanishes too slowly the spring keeps dragging me toward z^0 and holds me back from the solution. Parametrize β(t) = γ/t^p and watch the tradeoff. With p > 1 the spring dies too early — flow barely contracted, slow. With p < 1 it dies too late — keeps dragging, slow. The sweet spot is p = 1, β(t) = 1/t, where contracting and vanishing speeds are matched and the flow lands fastest. So the anchored flow with β(t) = 1/t is the target, and its discrete shadow is an anchoring coefficient that decays like 1/k. The natural discrete choice is β_k = 1/(k+2): it's the 1/t schedule, and the +2 offset just keeps β_0 = 1/2 < 1 sensible at the start.

Let me check β(t) = 1/t actually accelerates on the rotation, since that's where everything dies or lives. For L = xy the anchored flow is ẋ = −y + (1/t)(x^0 − x), ẏ = x + (1/t)(y^0 − y). Multiply through by t to get a clean ODE in the scaled variables: d/dt(t·x) = t ẋ + x = −t y + x^0, and d/dt(t·y) = t ẏ + y = t x + y^0. Differentiate again: d²/dt²(t x) = −d/dt(t y) = −t x − y^0, and d²/dt²(t y) = d/dt(t x) = −t y + x^0. So t x and t y each satisfy a forced harmonic oscillator, giving t x(t) = c₁ cos t − c₂ sin t − y^0 and t y(t) = c₁ sin t + c₂ cos t + x^0; fitting the initial conditions,
x(t) = (y^0 cos t + x^0 sin t − y^0)/t,  y(t) = (y^0 sin t − x^0 cos t + x^0)/t.
The iterate decays like 1/t — so ‖z(t)‖² ~ 1/t², the squared gradient norm goes like 1/t². Contrast the EG-flavored (Moreau–Yosida-regularized) flow on the same xy, which only decays like exp(−λt/(1+λ²)) at a rate set by the small regularization λ. The anchor converts the EG flow's feeble exponential-in-λ damping into a clean polynomial 1/t² decay. That's the acceleration, and it's exactly on the bilinear instance the naive method diverged on. So β(t) = 1/t, hence β_k = 1/(k+2), is not arbitrary — it's the schedule that balances contraction against vanishing and turns rotation into 1/t² decay.

Now the real work: prove the discrete EAG converges, and at what rate. I'll look for a Lyapunov function that's nonincreasing along the iteration. Given the two pieces I expect to control — the squared gradient norm I want to bound, and an anchoring-flavored inner product — let me posit
V_k = A_k ‖G(z^k)‖² + B_k ⟨G(z^k), z^k − z^0⟩,
with coefficient sequences A_k, B_k to be pinned down. The first term is literally the quantity I'm trying to drive down (weighted up by a growing A_k so that "V bounded" forces ‖G‖² small); the second is the anchoring term, the inner product of the gradient with the displacement from the anchor. The whole game is to choose A_k, B_k (and along with them the step/anchor relationship) so that V_{k+1} ≤ V_k, and so that A_k grows fast enough that boundedness of V translates into a fast rate on ‖G(z^k)‖².

I need the algebraic identities the EAG update gives me. From z^{k+1/2} = z^k + β_k(z^0−z^k) − α_k G(z^k) and z^{k+1} = z^k + β_k(z^0−z^k) − α_k G(z^{k+1/2}) (I'll allow a possibly-varying α_k for now and specialize to constant later), subtract and rearrange:
z^k − z^{k+1} = β_k(z^k − z^0) + α_k G(z^{k+1/2}),
z^{k+1/2} − z^{k+1} = α_k(G(z^{k+1/2}) − G(z^k)),
z^0 − z^{k+1} = (1−β_k)(z^0 − z^k) + α_k G(z^{k+1/2}).
The first comes straight from the second update line; the second from subtracting the two update lines (the anchor and z^k cancel, leaving the gradient difference); the third from adding z^0 − z^k to the first. These three are the only structural facts I'll use.

Start with what monotonicity hands me for free: 0 ≤ ⟨z^k − z^{k+1}, G(z^k) − G(z^{k+1})⟩. So I can subtract any nonnegative multiple of this from V_k − V_{k+1} and keep a valid lower bound:
V_k − V_{k+1} ≥ V_k − V_{k+1} − (B_k/β_k)⟨z^k − z^{k+1}, G(z^k) − G(z^{k+1})⟩.
Write out V_k − V_{k+1} = A_k‖G(z^k)‖² + B_k⟨G(z^k), z^k−z^0⟩ − A_{k+1}‖G(z^{k+1})‖² − B_{k+1}⟨G(z^{k+1}), z^{k+1}−z^0⟩, substitute z^{k+1}−z^0 = −(z^0−z^{k+1}) using the third identity, and expand the monotonicity cross-term using the first identity. After cancellations — and here I impose the first coefficient relation, B_{k+1} = B_k/(1−β_k), which is exactly what makes the ⟨G(z^{k+1}), z^0−z^k⟩ pieces collect cleanly — I'm left with a quadratic form in the three gradients G(z^k), G(z^{k+1/2}), G(z^{k+1}):
V_k − V_{k+1} ≥ A_k‖G(z^k)‖² − A_{k+1}‖G(z^{k+1})‖² + α_k B_{k+1}⟨G(z^{k+1}), G(z^{k+1/2})⟩ − (α_k B_k/β_k)⟨G(z^{k+1/2}), G(z^k) − G(z^{k+1})⟩.

There's still a ‖G(z^{k+1/2})‖² that I'd like to appear with a good sign — it's the look-ahead point, the thing extragradient buys me — so I bring in Lipschitzness. From R-Lipschitz G and the second identity z^{k+1/2}−z^{k+1} = α_k(G(z^{k+1/2})−G(z^k)),
0 ≤ R²‖z^{k+1/2}−z^{k+1}‖² − ‖G(z^{k+1/2})−G(z^{k+1})‖² = α_k²R²‖G(z^k)−G(z^{k+1/2})‖² − ‖G(z^{k+1/2})−G(z^{k+1})‖².
Subtract A_k/(α_k²R²) times this from the running bound. Now I have a full quadratic form in (G(z^k), G(z^{k+1/2}), G(z^{k+1})), and I want to choose the coefficients so it's manifestly nonnegative. Two more relations make it collapse. Impose A_k = α_k B_k/(2β_k): then the coefficient of the cross term ⟨G(z^k), G(z^{k+1/2})⟩ becomes 2A_k − α_k B_k/β_k = 0 and that term vanishes entirely. And notice α_k B_{k+1} + α_k B_k/β_k = α_k B_k(1/(1−β_k) + 1/β_k) = α_k B_k/(β_k(1−β_k)) = 2A_k/(1−β_k), which fixes the coefficient of ⟨G(z^{k+1/2}), G(z^{k+1})⟩. Finally choose the A-recurrence
A_{k+1} = A_k(1 − α_k²R² − β_k²)/((1−α_k²R²)(1−β_k)²),
which is what's forced by α_{k+1} = α_k β_{k+1}(1−α_k²R²−β_k²)/(β_k(1−β_k)(1−α_k²R²)) together with A_{k+1} = α_{k+1}B_{k+1}/(2β_{k+1}). Plug all of these in and the whole thing simplifies to
V_k − V_{k+1} ≥ [A_k(1−α_k²R²)/(α_k²R²)]‖G(z^{k+1/2})‖² + [A_k(1−α_k²R²−β_k)²/(α_k²R²(1−α_k²R²)(1−β_k)²)]‖G(z^{k+1})‖² − [2A_k(1−α_k²R²−β_k)/(α_k²R²(1−β_k))]⟨G(z^{k+1/2}), G(z^{k+1})⟩.
This is a‖u‖² + b‖v‖² − 2c⟨u,v⟩ with u = G(z^{k+1/2}), v = G(z^{k+1}); with these coefficients, c² = ab, so it is exactly a signed square in inner-product form, or Young's inequality at equality. So V_k − V_{k+1} ≥ 0 — the Lyapunov function is nonincreasing. The extragradient look-ahead earned me the ‖G(z^{k+1/2})‖² term; monotonicity and Lipschitzness, weighted by exactly these coefficients, made everything line up into a sum of squares.

Now specialize the schedule and read off the rate. With β_k = 1/(k+2): B_{k+1} = B_k/(1−β_k) = B_k(k+2)/(k+1) telescopes from B_0 = 1 to B_k = k+1, growing *linearly*. And A_k = α_k B_k/(2β_k) = α_k(k+1)(k+2)/2, growing *quadratically* (as long as α_k stays bounded away from zero). Linear B and quadratic A — that's the structural reason an O(1/k²) rate is even possible: the gradient-norm term in V is weighted by something ~k², so V staying bounded squeezes ‖G(z^k)‖² down by ~1/k². Let me make that precise. I have the constant-α version in mind (the simpler, fixed-step EAG), but the cleanest rate proof is for a slightly varying α_k that decreases to a positive limit, so let me run that and note the constant-step version after.

V_k is nonincreasing, so V_k ≤ V_0 = A_0‖G(z^0)‖² + B_0⟨G(z^0), z^0−z^0⟩ = α_0‖G(z^0)‖² ≤ α_0 R²‖z^0−z*‖², using B_0 = 1, A_0 = α_0(1)(2)/2 = α_0, and ‖G(z^0)‖ = ‖G(z^0)−G(z*)‖ ≤ R‖z^0−z*‖. Now I lower-bound V_k to extract ‖G(z^k)‖². Start from the definition and use monotonicity to swap z^k for z*:
V_k = A_k‖G(z^k)‖² + B_k⟨G(z^k), z^k−z^0⟩ ≥ A_k‖G(z^k)‖² + B_k⟨G(z^k), z*−z^0⟩,
because ⟨G(z^k), z^k − z*⟩ = ⟨G(z^k)−G(z*), z^k−z*⟩ ≥ 0 lets me replace z^k−z^0 by z*−z^0 at the cost of subtracting a nonnegative quantity. Now the inner product ⟨G(z^k), z*−z^0⟩ could be negative; tame it with Young's inequality, ⟨G(z^k), z*−z^0⟩ ≥ −(A_k/(2B_k))‖G(z^k)‖²·... let me be careful: B_k⟨G(z^k), z*−z^0⟩ ≥ −(A_k/2)‖G(z^k)‖² − (B_k²/(2A_k))‖z^0−z*‖² (splitting the product so the ‖G‖² piece carries coefficient A_k/2). So
V_k ≥ A_k‖G(z^k)‖² − (A_k/2)‖G(z^k)‖² − (B_k²/(2A_k))‖z^0−z*‖² = (A_k/2)‖G(z^k)‖² − (B_k²/(2A_k))‖z^0−z*‖².
Substitute A_k = α_k(k+1)(k+2)/2 and B_k = k+1:
V_k ≥ (α_k/4)(k+1)(k+2)‖G(z^k)‖² − ((k+1)/(α_k(k+2)))‖z^0−z*‖².
The α_k's are decreasing to a limit α_∞ > 0, so replacing α_k by α_∞ where it helps and bounding (k+1)/(k+2) ≤ 1 only weakens the bound:
V_k ≥ (α_∞/4)(k+1)(k+2)‖G(z^k)‖² − (1/α_∞)‖z^0−z*‖².
Combine the upper and lower bounds on V_k: (α_∞/4)(k+1)(k+2)‖G(z^k)‖² ≤ V_k + (1/α_∞)‖z^0−z*‖² ≤ (α_0 R² + 1/α_∞)‖z^0−z*‖². Divide:
‖∇L(z^k)‖² = ‖G(z^k)‖² ≤ [4(1 + α_0 α_∞ R²)/α_∞²] · ‖z^0−z*‖²/((k+1)(k+2)).
There it is — O(R²‖z^0−z*‖²/k²), on the *last* iterate, no averaging, no best-so-far tracking. The linear-B-quadratic-A structure I engineered via β_k = 1/(k+2) is exactly what produced the (k+1)(k+2) in the denominator.

I owe myself the fact that the α_k sequence is well-behaved — decreasing but bounded away from zero, so α_∞ > 0 and the constant is finite. WLOG R = 1 (rescale α_k by R). The recurrence α_{k+1} = α_k(1 − (1/((k+1)(k+3)))·α_k²/(1−α_k²)) can be rewritten as
α_k − α_{k+1} = α_k³/((k+1)(k+3)(1−α_k²)),
which is manifestly positive for α_k ∈ (0,1), so the sequence is strictly decreasing. To show it doesn't crash to zero, suppose 0 < α_N < ρ < 1 with γ := ½(1/(N+1) + 1/(N+2))·ρ²/(1−ρ²) < 1 — and this holds for all N ≥ 0 whenever ρ < 3/4. Sum the decrements from N onward, using the induction hypothesis α_{N+j} < ρ in the denominator and the monotone bound α_{N+j} ≤ α_N to turn α_{N+j}³ into ρ²α_N:
α_N − α_{N+k+1} = Σ_{j=0}^k α_{N+j}³/((N+j+1)(N+j+3)(1−α_{N+j}²)) < (ρ²α_N/(1−ρ²)) Σ_{j=0}^∞ 1/((N+j+1)(N+j+3)).
That last sum telescopes — 1/((m+1)(m+3)) = ½(1/(m+1) − 1/(m+3)) — to ½(1/(N+1) + 1/(N+2)), so the whole thing is < γ α_N, giving (1−γ)α_N < α_{N+k+1} < α_N for all k. Hence α_k ↓ α_∞ ≥ (1−γ)α_N > 0. Concretely, α_0 = 0.618/R drives α_∞ ≈ 0.437/R, and plugging into the rate constant 4(1+α_0α_∞R²)/α_∞² gives ≈ 27. So EAG with α_0 = 0.618/R achieves ‖∇L(z^k)‖² ≤ 27R²‖z^0−z*‖²/((k+1)(k+2)).

What about the genuinely *constant*-step version — the one I'd actually run, a single fixed step α at every iteration with nothing to tune across steps? The Lyapunov skeleton is identical: same V_k = A_k‖G(z^k)‖² + B_k⟨G(z^k), z^k−z^0⟩, same B_k = k+1, same β_k = 1/(k+2). What changes is that with a fixed α I can no longer let the α-recurrence pick A_{k+1} for me; instead I combine the monotonicity inequality (weight (k+1)(k+2)) and the Lipschitz inequality (weight τ_k ≥ 0, a free parameter) and collect V_k − V_{k+1} into a single trace, V_k − V_{k+1} ≥ Tr(M_k S_k M_kᵀ) where M_k = [G(z^k) G(z^{k+1/2}) G(z^{k+1})] and S_k is a 3×3 tridiagonal matrix whose entries are explicit in α, k, A_k, τ_k. If S_k ⪰ 0 then Tr(M_k S_k M_kᵀ) = Tr(S_k M_kᵀ M_k) ≥ 0 because the PSD cone is self-dual under the trace inner product, so V_k is nonincreasing. The job is then to pick τ_k and an A_k-recurrence keeping A_k in an interval [ℓ_k, u_k] ≈ αk²/2 (still quadratic growth) while S_k ⪰ 0 — this is the performance-estimation idea of searching for Lyapunov coefficients meeting a semidefiniteness constraint while forcing the desired growth. The algebra is genuinely brutal — the S_k entries are degree-three polynomials in α and k that I won't pretend are clean — but the punchline is what matters: it goes through for α R small enough, with the step-size conditions
1 − 3αR − α²R² − α³R³ ≥ 0  and  1 − 8αR + α²R² − 2α³R³ ≥ 0,
which hold for α ∈ (0, 1/(8R)] (and in fact up to about 0.1265/R), and the rate is
‖∇L(z^k)‖² ≤ [4(1+αR+α²R²)/(α²(1+αR))] · ‖z^0−z*‖²/(k+1)²,
i.e. at α = 1/(8R) a constant of 260. So the constant-step EAG also gets the optimal O(1/k²) last-iterate rate, just with a fatter constant than the varying-step version. The fixed step is simpler to run; the varying step is what makes the *proof* clean.

I want to know this is actually optimal, not just good, because otherwise I'd keep hunting. So I need a matching lower bound: no first-order method can beat O(1/k²) on the squared gradient norm for R-smooth convex-concave problems. The clean route is through biaffine problems, the simplest convex-concave family. Take L(x,y) = ⟨Ax − b, y − c⟩, so ∇_x L = Aᵀ(y−c) and ∇_y L = Ax − b, and G is ‖A‖-Lipschitz; the saddle point solves Ax = b and Aᵀ(y−c) = 0. Translate so z^0 = 0. For any algorithm whose iterates stay in the span of the gradients it has queried (and even for the broader class that uses the x- and y-gradients separately), the iterates land in Krylov subspaces of A built from b and c. Take A symmetric with b = Ac; then both x^k and y^k live in the order-(k−1) Krylov subspace K_{k−1}(A; b) = span{b, Ab, ..., A^{k−1}b}. Now I've reduced the minimax problem to: solve the linear system Ax = b using only matrix-vector products with A — and that's a classical, fully-understood problem. Nemirovsky's lower bound, via Chebyshev-type matrix polynomials, says there's a symmetric A with ‖A‖ ≤ R and b ∈ range(A) such that for any x in that Krylov subspace,
‖Ax − b‖² ≥ R²‖x*‖²/(2⌊k/2⌋+1)²,
where x* is the minimum-norm solution. The gradient norm at z^k = (x^k, y^k) is ‖∇L(z^k)‖² = ‖Aᵀ(y^k−c)‖² + ‖Ax^k − b‖² = ‖Ay^k − b‖² + ‖Ax^k − b‖² (using A = Aᵀ, b = Ac so that c plays the role of a shifted solution), and applying the bound to each block,
‖∇L(z^k)‖² ≥ 2 · R²‖x*‖²/(2⌊k/2⌋+1)² = R²‖z^0 − z*‖²/(2⌊k/2⌋+1)² = Ω(R²‖z^0−z*‖²/k²).
So the O(1/k²) rate is optimal up to a constant — the upper and lower bounds match in order, and EAG sits at the optimum. (Incidentally, this is why EAG can break the O(1/k) last-iterate lower bound that pins extragradient: that lower bound is for *stationary* algorithms with fixed coefficients, and EAG's β_k = 1/(k+2) are non-stationary — the anchoring schedule itself is what escapes the lower-bound class.)

Let me step back and name what just happened with the acceleration, because it's worth being clear it isn't Nesterov in disguise. Nesterov's momentum *adds* inertia — it pushes the iterate along its recent direction. Anchoring does the opposite: the β_k(z^0 − z^k) term *pulls back* toward the start, damping motion, killing the oscillation. Two opposite-looking mechanisms both yielding O(1/k²), one for minimization error, one for the gradient norm in minimax — combined here with the extragradient look-ahead that handles the monotone rotation a pure gradient step can't. The look-ahead supplies the per-step decrease at a constant step; the anchor supplies last-iterate convergence and solution selection; together they break the O(1/k) ceiling.

One more thing I have to be honest about, because it changes how the method behaves the moment I feed it noisy information instead of exact gradients: each gradient query can return G(z) + ξ, or equivalently with a fixed step size the update line can receive an additive Gaussian perturbation after the exact G(z) call. The whole acceleration rests on the Lyapunov function decreasing every step, and that decrease was driven by monotonicity and Lipschitz identities that hold for the *exact* operator. With noisy gradients or update perturbations, each step injects an error into those identities, and because the gradient-norm term in V is weighted by A_k ~ k², the accumulated noise gets *amplified* by the same quadratic factor that gave me the acceleration. So the accelerated method does not enjoy free noise-robustness — exactly like stochastic Nesterov in convex minimization, the fast transient is real but error accumulates, and the gradient norm flattens out once noise dominates rather than continuing to 1/k². Stability requires the oracle variance to decay on the order of 1/k; with fixed-variance noise the error terms accumulate. So with stochastic perturbations, I should expect EAG's fast O(1/k²) transient first, then noise-dominated behavior set by σ; reducing that floor would need a separate variance-control mechanism layered on top, not anything in the update itself. The update is the same EAG step; the noise is injected around the operator calls.

Now let me write the method in code. The state carries the iterate z, the fixed anchor z^0, and the step index (the anchor coefficient needs k). Each step does two operator evaluations — the predictor at z, the corrector at the look-ahead w — and two independent additive Gaussian update perturbations, with the same anchor offset in both lines. I keep the step size τ = α constant (the simpler fixed-step form), and the anchoring coefficient β_k = 1/(k+2):

```python
import numpy as np


def init_state(problem, initial_z, hyperparameters):
    # carry the iterate, the fixed anchor z^0, and the iteration index k
    z0 = np.asarray(initial_z, dtype=float).reshape(2 * problem.dim)
    return {
        "z": z0,                 # current iterate z^k
        "anchor_z": z0.copy(),   # the anchor z^0 (never changes)
        "step_index": 0,         # k, for the anchoring coefficient
    }


def step(state, problem, hyperparameters):
    tau = float(hyperparameters["tau"])              # constant step size α
    z = state["z"]
    anchor_z = state["anchor_z"]
    k = int(state["step_index"])

    # anchoring coefficient β_k = 1/(k+2): the discretized 1/t anchored-flow schedule
    # (linear B_k = k+1, quadratic A_k ⇒ O(1/k^2)); with 0-based k the equivalent form is 1/(k+2)
    beta = 1.0 / (k + 2.0)

    # predictor (look-ahead), with anchor pull toward z^0 and an update perturbation
    # w = z^k + β_k (z^0 - z^k) - α G(z^k) + η
    g = problem.grad(z)
    w = z + beta * (anchor_z - z) - tau * g + problem.noise()

    # corrector: step from z^k using G(w), same anchor offset based on z^k
    # z^{k+1} = z^k + β_k (z^0 - z^k) - α G(w) + η'
    gw = problem.grad(w)
    z_next = z + beta * (anchor_z - z) - tau * gw + problem.noise()

    new_state = {"z": z_next, "anchor_z": anchor_z, "step_index": k + 1}
    # the iterate whose gradient norm is measured (last iterate)
    return new_state, z_next


def get_hyperparameters(problem_name):
    # constant step τ = α, inside the convergent range α ≤ 1/(8R) for each instance's R
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```

So the causal chain, end to end. The naive simultaneous gradient method diverges by spiralling outward on the bilinear rotation, because the saddle operator is a rotation and a forward step grows the radius. Extragradient defeats the rotation with a look-ahead correction and a constant step, and its monotonicity-plus-Lipschitz descent gives a per-step decrease of (1−α²R²)α²‖G‖² — but only O(1/k), and only on the best iterate, with a hard last-iterate ceiling for its algorithm class. The Halpern iteration showed that pulling back toward a fixed anchor makes the *last* iterate converge and implicitly selects the nearest solution at an O(1/k) residual rate; transplanting that anchor onto gradient steps (SimGD-A) gave a last-iterate gradient-norm rate but was shackled to a diminishing step and stalled below O(1/k). The two failures are complementary — EG allows a constant step but is best-iterate, anchoring is last-iterate but needs a shrinking step — so I planted the anchor inside both half-steps of extragradient and kept the step constant. The anchored-flow ODE told me the schedule: β(t) = 1/t balances the anchor's contracting speed against its vanishing speed, turning the rotation's slow exponential damping into 1/t² decay, so β_k = 1/(k+2). A Lyapunov function V_k = A_k‖G(z^k)‖² + B_k⟨G(z^k), z^k−z^0⟩, made nonincreasing by exactly tuned monotonicity and Lipschitz inequalities (with the look-ahead gradient earning a sum-of-squares), has B_k = k+1 linear and A_k ~ k² quadratic under that schedule, and boundedness of V then forces ‖∇L(z^k)‖² = O(R²‖z^0−z*‖²/k²) on the last iterate — for varying step with constant ≈27, for constant step with constant 260 under αR ≤ 1/8. A biaffine-to-Krylov reduction and Nemirovsky's matrix-Chebyshev lower bound show Ω(R²/k²), so the rate is optimal. With noisy gradients or update perturbations, the same update has EAG's fast transient until the k²-weighting that produces the acceleration also amplifies the injected noise.
