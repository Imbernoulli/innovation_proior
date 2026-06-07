OK, let me think this through from scratch. I have a coherent wave — in an electron microscope, say — and I can record two pictures of it: a focused image, and its diffraction pattern. The thing is, a photographic plate only registers how *bright* the wave is, the intensity. It throws away the phase. So from the image I learn the amplitude `|f(x)|` of the wavefront, and from the diffraction pattern I learn the amplitude `|F(u)|` of its Fourier transform, because the back focal plane of a lens holds the Fourier transform of the front. Two amplitudes, one Fourier-transform relationship `F = FT(f)` tying them. And the phase — the part that actually encodes where the atoms are — is gone from both.

If I had the phase in either plane I'd be finished. Knowing `f(x) = |f(x)| e^{iφ(x)}` completely, I just transform and I have the whole wave; knowing `F(u)` completely, I inverse-transform. So the entire game is: fill in the missing phase. Can I just search for it? The field is, what, 128×128 complex numbers, so there are N² ≈ 16000 phase values to pick, each a real number on a circle. There is no searching that. I need structure.

I know two things about the unknown wave. I know its modulus in the object plane is `|f|`. And I know its modulus in the Fourier plane is `|F|`. Nothing else. Let me make that into sets. Let `M_o` be the set of all complex fields `g(x)` whose modulus equals the measured `|f(x)|` everywhere — phase totally free. Let `M_F` be the set of all fields whose *Fourier transform* has modulus `|F(u)|` everywhere — phase free over there. The true wave is in both. So I want a point in the intersection `M_o ∩ M_F`. The phase problem just turned into: find the meeting point of two sets.

Now, what does a single one of these sets look like, pointwise? Fix a pixel. `M_o` says: this complex number must have modulus `a = |f(x)|`. That's a *circle* of radius `a` in the complex plane — I'm free to be anywhere on it (any phase) but nowhere off it. So `M_o` is a product of circles, one per pixel. Same for `M_F`, only the circles live after a Fourier transform.

The natural move with two sets you want to meet is to bounce between them. Take a current guess, snap it onto the first set, snap that onto the second, repeat. But "snap onto the set" needs a definition — and the cleanest one is: go to the *nearest* point of the set. So let me actually compute the nearest point of a fix-the-modulus set. I have some complex number `x` and I want the closest `z` with `|z| = a` fixed. Write `z = a e^{iθ}`, minimize `|x − a e^{iθ}|²` over `θ`. Expand: `|x|² + a² − 2a|x|cos(θ − arg x)`. The only `θ`-dependence is `−2a|x|cos(θ − arg x)`, minimized when `cos = 1`, i.e. `θ = arg x`. So the nearest point keeps the phase of `x` and just resets the modulus to `a`:

  `P(x) = a · x/|x| = a · e^{i·arg x}.`

If `x = 0`, the phase is undefined and every point on the radius-`a` circle is equally close; in code I can choose the zero angle convention. If `a = 0`, the projection is just zero.

That's beautiful and it's exactly the operation I can do with the data I have — I don't *know* the phase, but whatever phase my current estimate has, I keep it, and I overwrite the amplitude with the amplitude I *do* know. So projecting onto `M_o` means "keep computed phase, set modulus to `|f|`," and projecting onto `M_F` means "keep computed phase, set modulus to `|F|`" — but the second one lives in the Fourier domain, so I have to transform there and back.

So the iteration writes itself. Start with the object-plane amplitude and a guessed phase: `x_k = |f| · e^{iφ_{k-1}}`. Fourier transform to the diffraction plane: `X_k = FT(x_k) = |X_k| e^{iψ_k}`. Now I'm sitting in `M_F`'s domain holding a field with the wrong modulus `|X_k|` but some phase `ψ_k`. Project: keep `ψ_k`, overwrite the modulus with the *measured* `|F|`: `Y_k = |F| · e^{iψ_k}`. Inverse transform back to the object plane: `y_k = IFT(Y_k) = |y_k| e^{iφ_k}`. Project onto `M_o`: keep `φ_k`, overwrite modulus with `|f|`. That gives the next estimate `x_{k+1} = |f| e^{iφ_k}`, and I loop. Four steps — transform, reset modulus, transform back, reset modulus — two FFTs an iteration. That's the whole algorithm and it only ever uses `|f|`, `|F|`, and the FFT.

I should ask: does this actually get better, or could it wander? Let me try to prove the modulus mismatch can't grow. I need to be explicit about the DFT scaling, because the error comparison depends on it. Use the phase-retrieval convention `F_m = (1/N)Σ_n f_n exp(-2πinm/N)` and `f_n = Σ_m F_m exp(2πinm/N)`, so Parseval is `Σ|f|² = NΣ|F|²`. Define the diffraction-plane error as the total squared gap between the modulus I computed and the modulus I'm forced to, `E_k² = N Σ_n (|X_k[n]| − |F[n]|)²`. And the object-plane error after the inverse transform is `e_k² = Σ_n (|y_k[n]| − |f[n]|)²`.

Start from `E_k`. By construction `Y_k = |F| e^{iψ_k}` has the *same phase* `ψ_k` as `X_k`, so `|X_k| − |Y_k|` are real and `||X_k| − |F||` is exactly `|X_k − Y_k|` pulled out by the common phase factor: `E_k² = N Σ |X_k − Y_k|²`. Now `X_k − Y_k` is the DFT of `x_k − y_k` (the DFT is linear, and `X_k=FT x_k`, `Y_k=FT y_k`). Parseval's identity says `N Σ|X_k − Y_k|² = Σ|x_k − y_k|²`. So

  `E_k² = Σ_n |x_k[n] − y_k[n]|².`

Write it out with `x_k = |f| e^{iφ_{k-1}}` and `y_k = |y_k| e^{iφ_k}`. Each term is `|f|² + |y_k|² − 2|f||y_k|cos(φ_{k-1} − φ_k)`. Since `cos ≤ 1`, dropping it to 1 can only make the cross term *more* negative — wait, I want a lower bound on `E_k`, so I use `cos(φ_{k-1}−φ_k) ≤ 1`, giving `−2|f||y_k|cos ≥ −2|f||y_k|`, hence

  `E_k² ≥ Σ (|f|² + |y_k|² − 2|f||y_k|) = Σ (|f| − |y_k|)² = e_k².`

So `e_k ≤ E_k`. The object-plane reset can't be worse than where the diffraction reset left me. Makes sense: `e_k` compares the same two moduli `|f|` and `|y_k|` but with the *best possible* phase alignment (the projection chose `arg y_k`), whereas `E_k` had whatever phase mismatch `φ_{k-1}−φ_k` happened to be — the projection is the nearest point, so it can only have shrunk the distance.

Now the next half-step. `x_{k+1} = |f| e^{iφ_k}` shares the phase `φ_k` with `y_k = |y_k| e^{iφ_k}`, and `|x_{k+1}| = |f|`. So `x_{k+1}` is the closest point with modulus `|f|` to `y_k`, and

  `Σ|y_k − x_{k+1}|² = Σ(|y_k| − |f|)² = e_k².`

Transform `x_{k+1}` forward to `X_{k+1}`. The next Fourier projection `Y_{k+1}` is the closest field with modulus `|F|` to `X_{k+1}`. But `Y_k` also has modulus `|F|`, so the projection cannot be farther than reusing the old `Y_k`:

  `NΣ|X_{k+1} − Y_{k+1}|² ≤ NΣ|X_{k+1} − Y_k|².`

The left side is `E_{k+1}²`. Parseval turns the right side into `Σ|x_{k+1} − y_k|²`, which is exactly `e_k²`. So `E_{k+1} ≤ e_k`. Chain it:

  `E_{k+1} ≤ e_k ≤ E_k.`

The error is monotone non-increasing, every iteration, forever. Each of the four steps is either a Fourier-domain hop with the matching Parseval scaling or a nearest-point projection. The scalar mismatch cannot diverge. That's a real guarantee, and it's the reason this thing is trustworthy enough to just run.

Now — does monotone-decreasing mean it *solves* the problem? Stare at this for a second. No. Monotone and bounded below by zero means the error has a limit, not that the field sequence has reached a true intersection. The sets `M_o` and `M_F` are products of *circles* — non-convex. With convex sets, alternating projection finds the intersection (or the closest pair) and you're safe. With non-convex sets, the iteration can settle into a fixed point where neither projection moves you appreciably and yet you are *not* at a true intersection — the error has bottomed out at a positive value. Geometrically: the nearest point on a circle is unique and well-behaved, but two families of circles linked by an FFT can trap a trajectory in a basin that isn't the global one. So I should expect: fast progress for a few iterations, then long plateaus where the error barely budges, possibly stalling well short of zero.

And that is exactly what shows up in practice — the error drops sharply over the first tens of iterations and then crawls along a plateau, sometimes a second and third plateau, occasionally leaving a faint stripe pattern frozen across the reconstruction. That stripe is the signature of a local minimum, not of the problem being unsolvable. There's a subtle failure mode too: if I initialize with a *constant* phase and the two amplitude data sets happen to be centrosymmetric, the DFT and IDFT never change the phase pattern at all and the error never decreases — the iteration is stuck from step zero. That's a good argument for seeding with a *random* phase: it breaks that symmetry, and it also lets me restart from a different seed to climb out of a bad basin. The plateau is the price of the non-convexity, and any honest use of this iteration plans around it (restart, or splice in a feedback step that can overshoot a constraint and escape — but that's a different track).

Let me now turn the same machine around, from *retrieval* to *synthesis*, because that's where it really earns its keep. Instead of two measured pictures, I have hardware. A spatial light modulator: a grid of pixels, each of which retards the phase of the wave passing through (or off) it, but does *not* change its amplitude. The beam hitting it has a fixed amplitude profile — call it `|A|`, uniform if the beam is flat-top, a Gaussian bump if it isn't. I cannot touch `|A|`; I can only set the per-pixel phase `φ_j`. A lens after the SLM Fourier-transforms the field, and I get to choose what intensity pattern lands in that Fourier plane: a picture, or a constellation of focal spots.

This is the *same two-modulus problem*. The object plane is the SLM: its modulus is fixed at `|A|` (the illumination), phase free — that's my `M_o`, except now the amplitude isn't a measurement, it's a hardware constraint. The Fourier plane is the target: I *want* its modulus to be the target amplitude `|T|` — that's my `M_F`. Find a field in both: a phase-only SLM field whose transform has the target modulus. So I run the identical loop. Seed a random phase on the SLM, build `A·e^{iφ}`, FFT to the target plane, keep the computed phase but reset the amplitude to the target `|T|`, IFFT back, throw away the amplitude the inverse transform produced and force it back to the fixed illumination `|A|` (this is the phase-only constraint biting — I am *not allowed* to keep the amplitude the math wants), read off the new phase, repeat. The "keep phase, reset to `|A|`" step is precisely what makes the output a legal phase-only hologram. If I could keep the amplitude the IFFT gave me, I'd just inverse-transform the target once and be done — but I can't, the SLM has no amplitude knob, and that single fact is the whole reason this needs to be iterative.

Now I aim it at a spot array — `M` bright points where I want to trap particles, all meant to be *equally* bright. The obvious target modulus is: amplitude 1 at each of the `M` spot locations, 0 elsewhere. Run GS. And here's the wall. The spots do *not* come out equal. Some are bright, some are dim, and there are faint extra spots — ghost orders — where I asked for nothing. The uniformity, measured as `u = 1 − (I_max − I_min)/(I_max + I_min)`, is poor when I wanted it near 1. Why? Because a phase-only mask has limited freedom: it's one real number per pixel, and forcing the transform to have *equal* amplitude at a structured set of points sets up interference conditions the mask simply can't satisfy exactly. The leftover energy has to go somewhere, so it scatters into uneven spot heights and ghosts. Asking for uniform spots, with phase only, gives me non-uniform spots. That's not a bug in GS; it's GS landing in a phase-only compromise, which can easily be non-uniform.

So let me think about what I actually want versus what I've been asking for. I've been *demanding* that the target amplitudes all equal 1, and accepting whatever delivered intensities `I_m = |V_m|²` fall out. But I don't care about the target amplitudes — they're a fiction I invented. What I care about is that the *delivered* `I_m` are equal. So flip it: treat the per-spot target amplitudes as *free weights* `w_m` and tune them so that the *outcome* is uniform. If a spot is coming out too dim, it means the algorithm isn't sending it enough power — so make that spot *more attractive* by raising its target weight. If a spot is too bright, lower its weight. Feedback.

Quantitatively: let `⟨|V|⟩` be the average delivered amplitude over the spots. I want every `|V_m|` to equal `⟨|V|⟩`. A spot with `|V_m| < ⟨|V|⟩` is underserved; the ratio `⟨|V|⟩/|V_m| > 1` is how much I'm short. So multiply that spot's weight by exactly that ratio:

  `w_m^{(k)} = w_m^{(k-1)} · ⟨|V|⟩ / |V_m|.`

Why multiplicative, why ratio-to-the-mean, why this exact form? Multiplicative keeps weights positive — an amplitude can't go negative — and it compounds gently. Ratio-to-the-mean rather than to some absolute brightness level means I'm correcting the *spread* directly; the efficiency is then something I monitor rather than a level I bake into the update. And — this is the clincher — look at the fixed point. When the spots finally *are* uniform, `|V_m| = ⟨|V|⟩` for all `m`, the ratio is 1, and the weights stop changing. The uniform state is exactly the resting state of this update. Any rule with that fixed point and positivity would do; this is the simplest one. (If I'd rather write it in intensities, `I_m = |V_m|²`, the same rule reads `w_m ← w_m · √(I_target/I_m)` with `I_target = ⟨|V|⟩²` for this update — the square root is just because intensity is amplitude squared; it's the identical update.)

I fold this one line into the GS loop. Each iteration I still do the GS phase bookkeeping — forward-propagate, read each spot's complex field `V_m`, keep its phase `θ_m = arg V_m` (that's the GS "keep computed phase" move, applied per spot), and on the way back I build the SLM field from the *weighted* target amplitudes `w_m` instead of from flat 1's. Then I refresh the weights from how the spots actually came out. If the delivered amplitudes become equal, the ratio becomes 1 everywhere and the weights freeze. This is weighted Gerchberg–Saxton, and it's just GS with one feedback line that says "reweight the targets until the outputs are even."

Let me write it as code I'd actually run. The phase-retrieval / image-synthesis version is the cleanest — a target image, a flat illumination, numpy FFTs:

```python
import numpy as np

def synthesize_phase(target_amplitude, illumination=1.0, n_iter=30):
    # target_amplitude: desired Fourier-plane amplitude, normalized 0..1
    h, w = target_amplitude.shape
    phase = np.random.rand(h, w)             # random seed breaks centrosymmetry / lets us restart
    u = np.empty((h, w), dtype=complex)
    I = None
    for _ in range(n_iter):
        u = illumination * np.exp(1j * phase)        # M_o: amplitude fixed at |A|, keep phase
        U = np.fft.fftshift(np.fft.fft2(u))          # lens: SLM plane -> target plane
        I = np.abs(U) ** 2                           # delivered intensity (for metrics)
        psi = np.angle(U)                            # phase the transform produced...
        U = target_amplitude * np.exp(1j * psi)      # M_F: keep that phase, reset amp to target
        u = np.fft.ifft2(np.fft.ifftshift(U))        # lens: target plane -> SLM plane
        phase = np.angle(u)                          # ...and keep ONLY the phase (drop amplitude)
    return phase, I                                  # phase = the hologram to write on the SLM
```

Every line is a step I derived: `exp(1j*phase)` with fixed illumination is the projection onto `M_o`; `fft2/fftshift` is the lens; `target_amplitude * exp(1j*psi)` is the nearest-point projection onto `M_F` (keep phase, reset modulus); `ifft2`; `np.angle(u)` discards the amplitude the inverse transform wanted, enforcing phase-only. The random seed is the symmetry-breaker.

Weighted GS is the same loop with the feedback line. I carry a weight image `w` over the target spots; before re-imposing the target amplitude I scale it by how short each point fell:

```python
def synthesize_balanced_phase(target_amplitude, n_iter=30):
    def normalization(a):
        a_min = a.min()
        a_max = a.max()
        return (a - a_min) / (a_max - a_min + 1e-12)

    h, w = target_amplitude.shape
    mask = (target_amplitude == 1)           # the spots we care about
    phase = np.random.rand(h, w)
    weights = target_amplitude.astype(float).copy()
    prev_w = target_amplitude.astype(float).copy()
    u = np.empty((h, w), dtype=complex)
    I = None
    for _ in range(n_iter):
        u = np.exp(1j * phase)               # flat illumination |A| = 1
        U = np.fft.fftshift(np.fft.fft2(u))
        I = np.abs(U) ** 2
        Inorm = normalization(I)
        psi = np.angle(U)
        # feedback: too-dim spots get heavier, too-bright lighter; fixed point = uniform
        weights[mask] = np.sqrt(target_amplitude[mask] / np.maximum(Inorm[mask], 1e-12)) * prev_w[mask]
        weights = normalization(weights)
        prev_w = weights.copy()
        U = weights * np.exp(1j * psi)       # impose the WEIGHTED target amplitude, keep GS phase
        u = np.fft.ifft2(np.fft.ifftshift(U))
        phase = np.angle(u)
    return phase, I
```

The only change from GS is the weight update and using `weights` in place of `target_amplitude` when I re-impose the modulus. In the binary-image numpy version, the feedback line is the same ratio idea written against the normalized delivered intensity on the target pixels: dim pixels have small `Inorm`, so `sqrt(target_amplitude/Inorm)` raises their weights; bright pixels get a smaller multiplier. `* prev_w` makes it multiplicative, and the min-max normalization keeps the level bounded so the update keeps correcting relative spread.

For a genuine spot *array* it's often cleaner to skip the full grid FFT and propagate directly to the `M` spot coordinates, since I only have a handful of points. With flat illumination, each spot `m` carries a complex field `V_m = (1/N) Σ_j exp(i(φ_j − Δ_j^m))`; with a non-flat pupil I multiply each term by the known illumination amplitude. Here `Δ_j^m = (2π/λf)(x_j x_m + y_j y_m) + (z_m π/λf²)(x_j² + y_j²)` is the known ramp that focuses light to `(x_m,y_m,z_m)` — a tilt for the lateral position, a Fresnel quadratic for the axial one. Backward propagation builds the SLM phase from the weighted, phased spots, `φ_j = arg Σ_m w_m e^{i(Δ_j^m + θ_m)}`; forward propagation reads each `V_m` back out; then keep `θ_m = arg V_m` and update `w_m`:

```python
def synthesize_spot_phase(delta, illumination, n_iter=30):
    # delta[j,m] is the propagation phase from SLM pixel j to spot m;
    # illumination is the fixed per-pixel |A|.
    N, M = delta.shape
    theta = np.random.rand(M) * 2*np.pi      # random per-spot phase offset (the SR seed)
    w = np.ones(M)                           # equal weights to start
    phi = np.zeros(N)
    I = np.zeros(M)
    for _ in range(n_iter):
        # backward: superpose single-spot ramps, weighted and phased, keep only phase -> hologram
        field = (w * np.exp(1j*theta))[None, :] * np.exp(1j*delta)      # (N_pix, M)
        phi = np.angle(field.sum(axis=1))                              # phi_j = arg sum_m ...
        # forward: propagate the hologram back to each spot
        A = illumination * np.exp(1j*phi)
        V = (A[:, None] * np.exp(-1j*delta)).sum(axis=0) / N           # V_m
        I = np.abs(V)**2
        theta = np.angle(V)                                            # GS: keep computed phase
        w = w * (np.mean(np.abs(V)) / np.abs(V))                       # WGS: reweight toward the mean
    e = np.sum(I)                                                      # efficiency
    u = 1 - (I.max() - I.min())/(I.max() + I.min())                   # uniformity
    return phi, e, u
```

`w = w * mean(|V|)/|V|` is the weight update verbatim; `theta = angle(V)` is GS keeping the per-spot phase; the `sum over m` then `arg` is the phase-only projection back onto the SLM; `e` and `u` are the metrics I monitor.

So the causal chain, start to finish: a detector keeps intensity and loses phase, leaving me two moduli — object and Fourier — joined by an FFT; I read that as wanting a point in the intersection of two fixed-modulus sets; the nearest point on a fixed-modulus set keeps the phase and resets the amplitude, so alternating projection becomes transform / reset-modulus / transform-back / reset-modulus; Parseval plus the nearest-point property prove the modulus error never increases; but the sets are non-convex, so a decreasing error can still plateau away from zero, which is why I seed randomly and can restart; turning the object-plane modulus into a hardware-fixed illumination makes this a phase-only hologram synthesizer; aimed at an equal-amplitude spot array it delivers *unequal* spots because phase-only can't split power evenly; so I stop demanding equal targets and instead reweight them by `w_m ← w_m·⟨|V|⟩/|V_m|`, whose fixed point is exactly the uniform array — weighted Gerchberg–Saxton.
