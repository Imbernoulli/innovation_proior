OK, let me think this through from scratch. I have a single intensity measurement — a far-field diffraction pattern, or a speckle-interferometry power spectrum — and from it I get the modulus of the Fourier transform of my object, `|F(u)|`, but not its phase. The object `f(x)` is something physical: light from a star, electron density in a crystal. So I know more than just `|F|`. I know `f(x)` is real. I know it's non-negative — it's an intensity distribution, it can't go below zero. And I know it sits inside a bounded region; it's a finite object on a dark background. That's the whole input: one Fourier modulus, plus "real, ≥0, finite support." I want the phase back, or equivalently `f(x)` itself.

The first thing I should be honest about is how this differs from the two-modulus problem, because there's a known iteration for that one and I'd love to reuse it. In electron microscopy or wavefront sensing you record *two* pictures of the same wave — a focused image and a diffraction pattern — so you measure `|f(x)|` *and* `|F(u)|`. Gerchberg and Saxton (1972) solved that by bouncing between the two domains: Fourier transform your current object estimate, replace the computed Fourier modulus with the measured `|F|` but keep the computed phase, inverse transform, replace the computed object modulus with the measured `|f|` keeping that phase, and loop. Four steps, two FFTs an iteration, and they proved a defined error goes down every iteration. It works because each domain hands you a *measured modulus* to snap onto.

But my problem has no measured `|f(x)|`. There's no focused image. The object plane gives me only inequalities — be real, be non-negative, be zero outside the support. So Gerchberg–Saxton's fourth step has nothing to reset the object modulus *to*. I can't just copy it.

Let me strip both problems down to the same skeleton and see what generalizes. In each domain I know *something* — a measured modulus, or a constraint — and the two domains are tied by an FFT. So the move is: transform to a domain, make the *minimum* change to the current field that satisfies whatever I know there, transform to the other domain, do the same. "Minimum change to satisfy what I know" — that's a projection onto the set of fields consistent with my knowledge in that domain. Gerchberg–Saxton is the special case where both domains' knowledge is a measured modulus. So let me generalize the fourth step to "project onto the object constraints, whatever they are." Call this the error-reduction algorithm.

I need the two projections explicitly. In the Fourier domain I know `|F(u)|`. The set of fields whose Fourier modulus equals `|F|` everywhere is, pixel by pixel after transforming, a circle of radius `|F(u)|` in the complex plane — any phase allowed, that one radius required. The nearest point on a circle of radius `a` to a complex number `G` keeps `G`'s phase and resets its modulus to `a`: write `G = |G|e^{iφ}`, minimize `|G − a e^{iθ}|² = |G|² + a² − 2a|G|cos(θ−φ)` over `θ`, the only `θ`-dependence is `−cos(θ−φ)`, minimized at `θ = φ`. So the Fourier projection is "keep the computed phase `φ`, set the modulus to `|F|`." Concretely: FFT the object, `G'_k(u) = |F(u)| exp(i·arg G_k(u))`, inverse FFT. Those are exactly the first three steps GS already does; I keep them verbatim.

The object projection is the new part. My object set is "real, ≥0, zero outside support `D`." That set is *convex* — it's the non-negative cone intersected with the subspace of functions vanishing outside `D`, both convex, intersection convex. The nearest point is found pointwise: outside `D`, set to 0; inside `D`, if the value is negative, clip to 0, else keep it. Let `g'_k(x)` be the inverse transform of the modulus-corrected spectrum, and let `γ` be the set of points where `g'_k` violates the constraints — where it's negative inside the support, or anywhere outside the support. Then the projected next estimate is

  `g_{k+1}(x) = g'_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = 0` for `x ∈ γ`.

That's the whole error-reduction iteration: FFT, reset Fourier modulus keeping phase, inverse FFT, zero out the constraint violations. Two FFTs, and it uses only `|F|` and the constraints.

Does it actually get better, or could it wander off? Let me try to prove the error can't grow, the way GS did, but in this generalized setting. I'll need to be careful with the DFT scaling because the comparison is between an object-domain sum and a Fourier-domain sum. Use the convention `F(u) = Σ_x f(x) exp(−i2πu·x/N)` and `f(x) = N^{-2} Σ_u F(u) exp(i2πu·x/N)` for the 2-D case, so Parseval reads `N^{-2} Σ_u |G(u)|² = Σ_x |g(x)|²`.

Define the Fourier-domain squared error at iteration `k` as the total squared amount by which the computed `G_k` had to be changed to satisfy the Fourier constraint: `E_{F,k}² = N^{-2} Σ_u |G_k(u) − G'_k(u)|²`. Since `G'_k` and `G_k` share a phase and differ only in modulus, this is `N^{-2} Σ_u (|G_k| − |F|)²` — the modulus mismatch. Now `G_k − G'_k` is the DFT of `g_k − g'_k` because the DFT is linear, so by Parseval

  `E_{F,k}² = N^{-2} Σ_u |G_k − G'_k|² = Σ_x |g_k − g'_k|².`

Compare that to the object-domain error `E_{O,k}² = Σ_x |g_{k+1}(x) − g'_k(x)|²`, the squared change the object projection made. `g_{k+1}` is *by definition the nearest* constraint-satisfying point to `g'_k`. And `g_k` — the previous estimate — also satisfies the object constraints (it came out of the previous object projection). So at every point, the distance from `g'_k` to its *nearest* constraint-satisfying neighbor `g_{k+1}` can't exceed the distance from `g'_k` to *some other* constraint-satisfying point `g_k`:

  `|g_{k+1}(x) − g'_k(x)| ≤ |g_k(x) − g'_k(x)|` for all `x`,

and summing the squares, `E_{O,k}² ≤ Σ_x |g_k − g'_k|² = E_{F,k}²`, i.e. `E_{O,k} ≤ E_{F,k}`.

Now the other half-step. By Parseval again, `E_{O,k}² = Σ_x |g_{k+1} − g'_k|² = N^{-2} Σ_u |G_{k+1} − G'_k|²`. The next Fourier projection makes `G'_{k+1}` the nearest field with modulus `|F|` to `G_{k+1}`. But `G'_k` also has modulus `|F|`, so the nearest such field can't be farther than `G'_k`: `Σ_u |G_{k+1} − G'_{k+1}|² ≤ Σ_u |G_{k+1} − G'_k|²`, which says `E_{F,k+1}² ≤ E_{O,k}²`. Chain the two:

  `E_{F,k+1} ≤ E_{O,k} ≤ E_{F,k}.`

The modulus error is monotone non-increasing, every iteration. Each step is either a Parseval-preserving Fourier hop or a genuine nearest-point projection, and a nearest-point projection can only shrink (or hold) the distance. Good — it can't diverge.

But monotone-decreasing is not the same as *solving* it. The error is bounded below by zero and non-increasing, so it converges to *some* limit. Nothing says that limit is zero, or that the field has reached a true intersection of the two sets. With *two convex* sets, alternating projection does land in the intersection (or the nearest pair). But my Fourier set isn't convex — it's a product of circles, and a circle is the canonical non-convex set. The object set is convex, but the pair isn't. So I should expect the trajectory to get *trapped*: a point where neither projection moves it much, yet the error has bottomed out at a positive value. Geometrically the two sets touch the trajectory's neighborhood in a way that pins it in a basin that isn't the global one.

And that's exactly the observed behavior. The error drops fast for the first thirty or so iterations and then crawls along a plateau — in a real run it sticks near 0.16, much later eases to 0.02, then 0.003, taking *thousands* of iterations. There's a characteristic signature too: it stagnates at a local minimum that shows up as a faint pattern of stripes laid across the reconstruction. The stripes aren't telling me the problem is unsolvable — the data do determine the object (for a complicated 2-D object with support and non-negativity the solution is essentially unique). They're telling me the *iteration* is stuck at a local, not global, minimum. Painfully slow convergence is the price of the non-convexity, and for the single-intensity problem that price is too high to pay.

Before I try to fix it, let me understand *why* it's slow, because that'll tell me what to change. Let me look at error-reduction as an optimizer. Treat the `N²` real values `g(x)` as parameters and minimize the Fourier squared error `B = E_F² = N^{-2} Σ_u (|G(u)| − |F(u)|)²`. What's the gradient? Differentiate, with `g(x)` real:

  `∂B/∂g(x) = 2 N^{-2} Σ_u (|G(u)| − |F(u)|) · ∂|G(u)|/∂g(x).`

Now `|G| = (G G*)^{1/2}`, so `∂|G|/∂g(x) = (1/2|G|)(G* ∂G/∂g + G ∂G*/∂g)`, and since `G(u) = Σ_x g(x)exp(−i2πu·x/N)`, `∂G/∂g(x) = exp(−i2πu·x/N)`. Substituting and grouping the two conjugate sums, and defining `G'(u) = |F(u)| G(u)/|G(u)|` — which is precisely the modulus-corrected spectrum from step two — the whole thing collapses, because the sums are inverse DFTs, to

  `∂B/∂g(x) = 2[g(x) − g'(x)],`

where `g'(x)` is the inverse transform of `G'`. So the gradient at every point is just twice "current estimate minus the error-reduction output," and computing it costs exactly the two FFTs I was already doing. Steepest descent says step `g̃(x) = g_k(x) − h ∂B/∂g = g_k(x) − 2h[g_k(x) − g'_k(x)]`. To pick `h`, expand `B` to first order in `g`: `B ≈ B_k + Σ_x ∂B_k/∂g(x)·[g(x) − g_k(x)]`, set it to zero, and using `Σ_x (∂B/∂g)² = 4 Σ_x [g_k − g'_k]² = 4 B_k`, the predicted zero of that tangent model is `g̃ − g_k = −(1/4)∂B = (1/2)[g'_k − g_k]`, i.e. `h = 1/4`. Along this particular same-phase Fourier move the mismatch scales as `(1−t)²`, so the tangent zero lands halfway to the point that actually makes the Fourier error vanish. Take the double-length step `h = 1/2`: then `g̃ = g'_k`, and at that point `B = 0` exactly, because `|G̃| = |F|`. Then project onto the object constraints, the fourth step. So error-reduction *is* steepest descent on `B` with a double-length step. That reframing tells me precisely why it stalls: it's gradient descent on a non-convex `B`, and on the plateau the gradient `2[g − g']` is tiny, so a fixed step barely moves. And every step ends by slamming into the object-constraint projection — I'm "constantly running into the object-domain constraints."

I could try smarter gradient methods — line-search the step `h_k`, or conjugate-gradient directions `D_k = g_k − g'_k + (B_k/B_{k-1})D_{k-1}`. These help: a bigger step on the plateau, conjugacy to avoid zig-zagging. But they all share the same structural annoyance — after every gradient step I must re-project onto the object constraints, so the search keeps colliding with the constraint set, and a fixed `h` is wrong (small gradient on the plateau wants a *large* `h`, but a large `h` elsewhere goes unstable). Adaptively tuning `h` is its own unsolved nuisance. What I really want is a method that satisfies the object constraints *inherently* while it drives the Fourier error down, instead of one that alternately fights them. Let me look for that.

Group the first three steps — FFT, reset Fourier modulus, inverse FFT — into a single black box. Feed it an input `g`, it returns an output `g' = IFT(|F| · FT(g)/|FT g|)`. The key property of this box is that *its output always has the correct Fourier modulus*. Whatever I put in, the output lies in the Fourier set `M`. So if the output *also* happens to satisfy the object constraints, the output is a solution — done. That flips my whole picture of what the input is for. In error-reduction and the gradient methods, the input `g_k` is my *current best estimate of the object*, and I refine it. But it doesn't have to be. The input is just a *driving function*: I get to choose it however I like to steer the *output* toward satisfying the object constraints. The input need not satisfy any object constraint at all. That's a lot of freedom I wasn't using.

So: how should I drive the input to push the output where I want? There's an empirical regularity about this nonlinear box — observed across these problems — that a small change `Δg` of the input produces, on average, a change of the output that's `≈ α Δg`, the same general direction, scaled by a roughly constant `α` (plus nonlinear terms I can't predict exactly). So if I *want* the output to change by some `Δg'`, a sensible input change is `β Δg'` for a constant `β` of order one. Now, what change of the output do I want? Where the output already satisfies the object constraints, I want no change. Where it *violates* them — at the points `γ` where `g'_k(x)` is negative inside the support or lies outside the support — I want to drive that output value to *zero* (that's what satisfying the constraint there means). So the desired output change is

  `Δg_k(x) = 0` for `x ∉ γ`,  `Δg_k(x) = −g'_k(x)` for `x ∈ γ`.

A logical next input is then `g_{k+1} = g_k + β Δg_k`:

  `g_{k+1}(x) = g_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = g_k(x) − β g'_k(x)` for `x ∈ γ`.

Call that the basic input–output algorithm. Notice what it does at the violating points: instead of slamming the output to zero (the projection), it nudges the *input* by `−β·output`, and lets the box respond next iteration. At the satisfied points it leaves the input alone. The input is no longer pretending to be an object estimate; it's a controller setpoint.

Let me also notice a property of the box that suggests a second variant. If I feed the output `g'` back in as the input, its output is *itself* — because `g'` already has the correct Fourier modulus, the box leaves it unchanged. So the output `g'` can always be regarded as having come from itself as input, regardless of what input actually produced it. That gives me license to build the next input out of `g'` rather than `g`. Replacing `g_k` by `g'_k` in the update:

  `g_{k+1}(x) = g'_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = g'_k(x) − β g'_k(x) = (1−β)g'_k(x)` for `x ∈ γ`.

Call that output–output. And look — if `β = 1`, the violating points get set to `(1−1)g' = 0` and the satisfied points stay at `g'`, which is *exactly error-reduction*. So error-reduction is just output–output with `β` pinned at 1. Since `β = 1` is generally *not* the best choice, error-reduction is a suboptimal special case of this more general family. That already suggests I'm in richer territory.

If I use output–output, I can already see a failure mode: the output on successive iterations can stop changing even though it's nowhere near a solution. Picture a point `x` inside the support where the output keeps coming out negative. Output–output sets the next input there from `g'_k` (or `(1−β)g'_k`), which is tied to the current bad output — there's no memory pushing it out, so it can sit negative forever, and the whole field locks. The trouble is that at the violating points the next input is built *only* from the present output, so a persistently-violating point has nothing accumulating to force it across.

What would force it across? I want a point that keeps violating to feel a *growing* push until it can't violate anymore. Basic input–output had that ingredient — it carried `g_k`, the previous input, and subtracted `β g'_k` from it. If a point stays violating, that input term `g_k − β g'_k` keeps getting decremented (or incremented) iteration after iteration, accumulating, so the input there drifts steadily until eventually the box's output must flip to non-negative. That's exactly the memory output–output lacks. But basic input–output, at the *satisfied* points, keeps the *input* `g_k` rather than the good output `g'_k` — which throws away the perfectly-good, constraint-satisfying, correct-modulus value the box just handed me.

So splice the two together: take the *good half* of output–output and the *escaping half* of basic input–output. At the satisfied points (`x ∉ γ`), keep the output `g'_k` — it already has the right Fourier modulus and satisfies the object constraints, so accept it. At the violating points (`x ∈ γ`), use the input-feedback term `g_k − β g'_k` — so a stubbornly-violating point accumulates a growing correction until it's forced non-negative. That is

  `g_{k+1}(x) = g'_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = g_k(x) − β g'_k(x)` for `x ∈ γ`,

the hybrid input–output algorithm. Upper line from output–output, lower line from basic input–output. The hybrid is specifically built to dodge the output–output stagnation: if a given point's output stays negative for more than one iteration, the corresponding input point grows larger and larger until that output value is forced to go non-negative. The feedback at the violating points is the escape mechanism; accepting the good output at the satisfied points is what keeps the progress already made.

Let me make sure I believe *why* this escapes a local minimum that traps error-reduction, beyond "memory." Error-reduction, at a violating point, *projects* — it replaces the value with the nearest feasible one (zero), a contractive move toward the constraint set. On a non-convex pair that contraction is exactly what lets the trajectory settle into a spurious basin: every move shrinks distance, so once you're in a basin you can't climb out. The hybrid, at violating points, does the opposite — it doesn't project the output, it pushes the *input* by `−β g'_k`, and `β` near one makes that an aggressive, *non-contractive* step that can carry the field clear past the constraint set. It's allowed to overshoot. Overshooting is precisely the thing a pure projection forbids and the thing needed to leave a local minimum.

I can see this even more sharply by writing the moves as projections and *reflections*. Let `P_M` be the Fourier projection (the box: keep phase, reset `|FT|` to `|F|`) and `P_S` the object projection (zero outside support, clip negatives). Error-reduction is `P_S P_M` — project onto one set, then the other, alternating projection. Now define the reflection across a set, `R = 2P − I`: reflecting a point across a set means going to the projection and then the same distance again past it. `P_X(x)` is the midpoint of `x` and `R_X(x)`. Consider, for the support-only constraint (drop non-negativity for a moment so `P_S` is linear), what the hybrid with `β = 1` computes. At points inside the support it returns `P_M(x)`; outside it returns `x − P_M(x)`. Inside-the-support-keep-`P_M`, outside-subtract-`P_M`: writing `1_D` for the support indicator, that's `1_D · P_M(x) + (1 − 1_D)(x − P_M(x))`. Expand using `P_S(z) = 1_D · z` (the support projection just multiplies by the indicator): regroup to `P_S(2P_M − I)(x) + x − P_M(x) = P_S R_M(x) + (I − P_M)(x)`. And since `P_S = (R_S + I)/2`, a couple of lines of algebra fold this into

  `x_{k+1} = ½(R_S R_M + I)(x_k).`

That's a clean object: reflect across the Fourier set, reflect the result across the object set, average with where you started. "Reflect–reflect–average." It is *not* alternating projection — it's an averaged double reflection. The reflections are what let it step *outside* each constraint set before averaging back, which is exactly the overshoot that lets it escape a basin where `P_S P_M` would be pinned. (For general `β` the same algebra gives `x_{k+1} = ½[R_S(R_M + (β−1)P_M) + I + (1−β)P_M](x_k)`, a relaxed version of the same reflect–reflect–average with `β` controlling how far the reflection reaches.) So the hybrid's superiority isn't a lucky heuristic: it's that averaged reflections traverse the non-convex landscape in a way alternating projections structurally cannot. With the *non-negativity* constraint switched back on, `P_S` is no longer linear, so this exact reflector form doesn't close — but the geometry is the same, and the update rule I derived from the input/output picture is the thing I actually run.

Now, what does this cost me in stability, and how do I use it? The hybrid is *not* monotone — that guarantee belonged to error-reduction, the contractive method, and I just deliberately gave up contraction to gain escape. So I should not interpret an early increase in the object error `E_O` as immediate failure; overshoot can temporarily move the output farther from the object constraints while moving the iterate out of the basin. The right error to watch isn't the Fourier `E_F` either: in input–output the input `g_k` is no longer an object estimate, so `E_F` of the input is meaningless. The meaningful metric is the object-domain error of the *output*, `E_O = √(Σ_{x∈γ} [g'_k(x)]² / Σ_x [g'_k(x)]²)` — how much the output violates the constraints, normalized by image energy. Once the hybrid has used non-contractive feedback to move away from a plateau, a few plain error-reduction iterations are the natural cleanup step: the contractive projector snaps the output back onto the object constraints without changing the Fourier-modulus step. So the working strategy is to *alternate*: run a block of hybrid input–output iterations to make progress and escape plateaus, then a few error-reduction iterations to consolidate the residual constraint violation, and repeat.

What about `β`? It's the feedback gain, ideally of order one. Push it up and the algorithm reacts faster — until it gets too large and goes unstable. Push it down and the motion is slower but steadier. A value near unity is the scale I want, and a conservative implementation default around `β = 0.8` leaves some margin below the aggressive setting.

Two more practical pieces and I can write it. First, the support mask. I get a bound on it for free: the autocorrelation is `IFT(|F|²)`, and the object's diameter is half the autocorrelation's diameter, so I threshold the autocorrelation at a small fraction of its peak to get a support region. It won't be tight — in 2-D the autocorrelation support doesn't pin the object support uniquely — so I keep the mask a bit *loose*. A too-tight, off-center mask is itself a stagnation trap: if the partially-reconstructed object isn't centered in the mask, applying the support constraint can chop off part of the object, which freezes the iteration. A good recipe is tight early for speed, then looser later so I never truncate the solution. Second, the starting input. A constant phase combined with the centro-symmetry of `|F|` can leave the iteration unable to break symmetry — and there's the twin-image degeneracy lurking, since `f(x)` and `f*(−x)` share `|F|`. So I seed with a *random* phase (or a randomized, thresholded, demagnified autocorrelation that has roughly the right size and shape), which breaks the symmetry and lets me restart from a fresh seed if I land in a bad basin. Reconstructing two or three times from different seeds and checking they agree is how I gain confidence the recovered solution is unique, up to the translation/inversion/twin symmetries `|F|` cannot break, rather than a spurious stagnation.

Let me write it as code I'd actually run. I keep a Fourier spectrum whose modulus has already been reset to the measured `|F|`; each loop inverse-transforms it to the output `g'_k`, applies the object-domain rule, then restores the measured Fourier modulus for the next loop.

```python
import numpy as np

def support_from_autocorrelation(mag, frac=0.04):
    # Autocorrelation = IFT(|F|^2); threshold it for a loose support mask.
    autoc = np.abs(np.fft.fftshift(np.fft.ifft2(mag ** 2)))
    return autoc > frac * autoc.max()

def object_domain_step(candidate, previous, mask, beta=0.8, mode=None):
    if mode is None:
        mode = "hybrid"
    if mode not in {"input-output", "output-output", "hybrid"}:
        raise ValueError("mode must be 'input-output', 'output-output', or 'hybrid'")

    mask = mask.astype(bool)
    gamma = ((candidate < 0) & mask) | (~mask)

    if mode in {"output-output", "hybrid"}:
        updated = candidate.copy()             # x not in gamma: accept g'_k
    else:
        updated = previous.copy()              # basic input-output keeps g_k there

    if mode in {"input-output", "hybrid"}:
        updated[gamma] = previous[gamma] - beta * candidate[gamma]
    else:
        updated[gamma] = candidate[gamma] - beta * candidate[gamma]
    return updated

def phase_retrieval(mag, mask=None, beta=0.8, steps=200, mode=None, init=None):
    if mask is None:
        mask = np.ones(mag.shape, dtype=bool)
    else:
        mask = mask.astype(bool)

    if init is None:
        spectrum = mag * np.exp(1j * 2 * np.pi * np.random.rand(*mag.shape))
    else:
        spectrum = mag * np.exp(1j * np.angle(np.fft.fft2(init)))

    previous = None
    for _ in range(steps):
        candidate = np.real(np.fft.ifft2(spectrum))  # g'_k, already in the Fourier set
        if previous is None:
            previous = candidate.copy()
        image = object_domain_step(candidate, previous, mask, beta, mode)
        spectrum = mag * np.exp(1j * np.angle(np.fft.fft2(image)))
        previous = image
    return previous

def object_error(output, mask):
    mask = mask.astype(bool)
    gamma = ((output < 0) & mask) | (~mask)
    energy = np.sum(output ** 2)
    if energy == 0:
        return 0.0
    return np.sqrt(np.sum(output[gamma] ** 2) / energy)
```

In the code, `spectrum` carries the measured Fourier modulus at the start of each loop, so `candidate` is the output `g'_k`. `gamma` is the violation set `γ`. Basic input-output keeps the previous input on satisfied points; output-output keeps the current output and becomes the error-reduction projector when `beta=1`; the hybrid keeps the current output where the constraints hold and applies the accumulating feedback `previous − beta * candidate` where they do not. A cleanup pass is just the same function with `mode="output-output", beta=1.0`. The random-phase spectrum breaks centro-symmetry, and the loose autocorrelation mask keeps me from truncating an off-center object.

So the causal chain, end to end: one Fourier modulus plus real/non-negative/support constraints; I read it as wanting a field in the intersection of a non-convex Fourier-modulus set and a convex object set, joined by an FFT; alternating nearest-point projections (error-reduction) provably never increase the modulus error — it's even a double-length-step steepest descent on `N^{-2}Σ(|G|−|F|)²` — but on the non-convex pair that monotone decrease plateaus at a striped local minimum, *painfully* slowly; reframing the three transform steps as a nonlinear box whose output always has the right modulus frees the input to be a *driving function* rather than an estimate; driving the input by `β` times the desired output change gives basic input–output, and the box's fixed-point property gives output–output, which freezes for lack of memory; splicing output–output's accept-the-good-output with basic input–output's accumulating feedback yields the hybrid `g_{k+1} = g'_k` on satisfied points, `g_k − β g'_k` on violators — an averaged reflect–reflect–average map that overshoots each constraint set and so escapes the basins a pure projection is pinned in; and because that escape costs monotonicity, I alternate hybrid blocks with a few error-reduction iterations, seed randomly, and keep the support loose.
