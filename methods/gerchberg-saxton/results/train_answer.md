I would solve the phase-retrieval and phase-only synthesis problems with the Gerchberg–Saxton algorithm and its weighted extension, because both can be written as alternating projections between two fixed-modulus sets joined by a Fourier transform. A detector records intensity, not phase, so in the image plane I only know the modulus |f| of the wavefront and in the diffraction plane I only know the modulus |F| of its Fourier transform, with F = FT(f). The phase carries the structural information I actually want, but it has been discarded in both planes. The synthesis setting is mathematically identical: a spatial light modulator, kinoform, or diffractive optical element retards phase pixel by pixel while the illumination amplitude |A| is fixed by hardware, so I must choose a phase mask whose Fourier transform produces a desired intensity pattern. In both cases amplitude is dictated in at least one plane, leaving only phase free, and a direct inverse transform of the desired field will not satisfy the amplitude constraint.

Existing simpler ideas each fail in a different way. A direct inverse transform assumes I can set both amplitude and phase in the modulator plane, which is false for a phase-only device. Random superposition builds a phase mask as the argument of a sum of single-spot phase ramps with random offsets; it is fast because it is a single pass, but it offers no control over how power divides, so the resulting spots are uneven and accompanied by ghost orders. Plain iterative Fourier-transform schemes propagate back and forth and reset the modulus in each plane, but they often stagnate because the fixed-modulus constraint at each pixel defines a circle in the complex plane, and the intersection of two such non-convex sets linked by an FFT can trap the iteration in a local fixed point. Hybrid input–output methods help escape plateaus for the single-modulus-plus-support astronomy problem, but they are not designed for phase-only synthesis and do not enforce uniform power splitting across a multi-spot array. What I need is an iteration that uses only the two known moduli and an FFT, is cheap enough for 128 by 128 fields and larger, provably does not make the mismatch worse, and can be extended to deliver uniform intensity across an array of focal spots.

I cast the problem as finding a field in the intersection of two sets. Let M_o be the set of all complex fields whose modulus equals the measured or constrained amplitude in the object plane, and let M_F be the set of all fields whose Fourier transform has modulus equal to the measured or desired amplitude in the Fourier plane. The true field lies in M_o ∩ M_F. Pointwise, each set is a product of circles: at a single pixel the condition |z| = a forces z onto a circle of radius a in the complex plane, leaving only the phase free. The nearest point on such a circle to any complex number x keeps the phase of x and resets its modulus to a, that is, P(x) = a x / |x| = a e^{i arg x}. This projection is the cheapest possible enforcement of the modulus constraint because it preserves all phase information in the current estimate and only adjusts magnitude.

A single Gerchberg–Saxton iteration therefore consists of four elementary steps. I start with the object-plane amplitude multiplied by the current guessed phase, x_k = |f| e^{i φ_{k-1}}. I propagate to the Fourier plane with an FFT, X_k = FT(x_k) = |X_k| e^{i ψ_k}. I keep the computed Fourier phase ψ_k but replace the modulus with the known or desired |F|, producing Y_k = |F| e^{i ψ_k}. I inverse transform back to the object plane, y_k = IFT(Y_k) = |y_k| e^{i φ_k}, and keep only the phase while forcing the modulus back to |f|, giving the next iterate x_{k+1} = |f| e^{i φ_k}. Two FFTs suffice per iteration, and the algorithm never needs anything beyond the two moduli and the FFT.

The error behavior is controlled by the fact that each half-step is either a unitary Fourier transform, which preserves squared distance up to the Parseval scaling, or a nearest-point projection, which cannot increase distance to the target set. With the convention F_m = (1/N) Σ_n f_n e^{-2π i n m / N} and Parseval's identity Σ |f|^2 = N Σ |F|^2, the diffraction-plane error E_k^2 = N Σ (|X_k| - |F|)^2 and the object-plane error e_k^2 = Σ (|y_k| - |f|)^2 satisfy E_{k+1} ≤ e_k ≤ E_k. Thus the modulus mismatch is monotone non-increasing. This is a genuine guarantee, but it is not a guarantee of reaching the global solution: because the fixed-modulus sets are non-convex, the iteration can plateau at a positive error corresponding to a local fixed point. A constant phase initialization can even be stuck forever under centrosymmetric data, so I seed the iteration with a random phase and am prepared to restart from fresh seeds if the plateau is unsatisfactory. The characteristic low-contrast stripe pattern sometimes seen in reconstructions is the signature of such a local minimum, not of problem non-uniqueness.

When I turn the same iteration to phase-only synthesis, the object-plane modulus becomes the fixed illumination profile |A| rather than a measurement, and the Fourier-plane modulus becomes the desired target amplitude. The loop remains identical: form A e^{i φ}, FFT, keep the Fourier phase and impose the target modulus, inverse FFT, and keep only the resulting phase as the next hologram. The step that discards the amplitude produced by the inverse transform is precisely what enforces the phase-only hardware constraint. For a general image target this produces a usable hologram after a few dozen iterations.

For an array of focal spots meant to be equally bright, plain Gerchberg–Saxton still gives non-uniform delivered intensities and ghost spots. The reason is fundamental: a phase-only mask has only one real degree of freedom per pixel, and asking it to split power evenly among a structured set of equal-amplitude targets is over-constrained. The residual energy must go somewhere, so some spots become brighter than others and extra diffraction orders appear. My response is to stop demanding equal target amplitudes and instead treat them as adaptive weights that are tuned until the delivered intensities are uniform. This is Weighted Gerchberg–Saxton.

In Weighted Gerchberg–Saxton, let V_m be the complex field delivered at spot m and let I_m = |V_m|^2 be its intensity. I define the average delivered amplitude over the spots as ⟨|V|⟩ and update each weight multiplicatively by w_m ← w_m · ⟨|V|⟩ / |V_m|. The multiplicative form keeps weights positive and compounds gently; ratio-to-the-mean corrects the spread directly; and the fixed point is exactly the desired uniform state, because when |V_m| equals the mean for every m the ratio is one and the weights stop changing. Equivalently, in intensities, the update reads w_m ← w_m · √(I_target / I_m) with I_target = ⟨|V|⟩^2. I insert this update between forward propagation and modulus replacement, using the weighted amplitudes w_m in place of the flat target. The same monotone-error reasoning applies, because the weighting only changes the target modulus being projected onto; the phase-projection steps are unchanged. The practical metrics are efficiency e = Σ_m I_m, the fraction of power landing in the wanted spots, and uniformity u = 1 - (max I_m - min I_m) / (max I_m + min I_m), which equals one when all spots are identical.

```python
import numpy as np

def synthesize_phase(target_amplitude, illumination=1.0, n_iter=30):
    '''Gerchberg–Saxton phase-only synthesis for a target Fourier-plane amplitude.'''
    h, w = target_amplitude.shape
    phase = np.random.rand(h, w)
    I = None
    for _ in range(n_iter):
        u = illumination * np.exp(1j * phase)
        U = np.fft.fftshift(np.fft.fft2(u))
        I = np.abs(U) ** 2
        psi = np.angle(U)
        U = target_amplitude * np.exp(1j * psi)
        u = np.fft.ifft2(np.fft.ifftshift(U))
        phase = np.angle(u)
    return phase, I


def synthesize_balanced_phase(target_amplitude, n_iter=30):
    '''Weighted Gerchberg–Saxton: drive target spot intensities to uniformity.'''
    def normalization(a):
        a_min = a.min()
        a_max = a.max()
        return (a - a_min) / (a_max - a_min + 1e-12)

    h, w = target_amplitude.shape
    mask = (target_amplitude == 1)
    phase = np.random.rand(h, w)
    weights = target_amplitude.astype(float).copy()
    prev_w = weights.copy()
    I = None
    for _ in range(n_iter):
        u = np.exp(1j * phase)
        U = np.fft.fftshift(np.fft.fft2(u))
        I = np.abs(U) ** 2
        Inorm = normalization(I)
        psi = np.angle(U)
        weights[mask] = np.sqrt(
            target_amplitude[mask] / np.maximum(Inorm[mask], 1e-12)
        ) * prev_w[mask]
        weights = normalization(weights)
        prev_w = weights.copy()
        U = weights * np.exp(1j * psi)
        u = np.fft.ifft2(np.fft.ifftshift(U))
        phase = np.angle(u)
    return phase, I


def synthesize_spot_phase(delta, illumination, n_iter=30):
    '''Weighted GS for focal-spot arrays via direct Fourier-optics propagation.
    delta[j, m] is the propagation phase from SLM pixel j to spot m.'''
    N, M = delta.shape
    theta = np.random.rand(M) * 2 * np.pi
    w = np.ones(M)
    phi = np.zeros(N)
    I = np.zeros(M)
    e = u = None
    for _ in range(n_iter):
        field = (w * np.exp(1j * theta))[None, :] * np.exp(1j * delta)
        phi = np.angle(field.sum(axis=1))
        A = illumination * np.exp(1j * phi)
        V = (A[:, None] * np.exp(-1j * delta)).sum(axis=0) / N
        I = np.abs(V) ** 2
        theta = np.angle(V)
        w = w * (np.mean(np.abs(V)) / np.abs(V))
        e = np.sum(I)
        u = 1 - (I.max() - I.min()) / (I.max() + I.min())
    return phi, e, u
```
