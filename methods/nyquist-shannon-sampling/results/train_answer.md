I will present the Nyquist-Shannon sampling theorem as the canonical statement about reconstructing a continuous-time bandlimited signal from its discrete-time samples. The theorem is usually named the Nyquist-Shannon sampling theorem, and I will refer to it that way throughout. It forms one of the bridges between analog and digital signal processing, and it explains why, if we sample a signal fast enough, we do not lose information about the original waveform.

A signal is called bandlimited if there is some finite frequency B such that its Fourier transform is zero for all frequencies whose absolute value exceeds B. In other words, all of the signal's energy lives inside the band from minus B to plus B. The Nyquist-Shannon sampling theorem says that such a signal can be perfectly reconstructed from its samples, provided the samples are taken uniformly at a rate that is strictly greater than twice the maximum frequency B. The critical rate, 2B, is called the Nyquist rate, and the corresponding frequency B is called the Nyquist frequency. The reconstruction is carried out by the Whittaker-Shannon interpolation formula, which convolves the sampled impulse train with a sinc kernel whose bandwidth matches the signal band.

To be more precise, suppose x of t is a continuous-time signal whose Fourier transform X of f is zero whenever the absolute value of f is greater than B. Sample x at times t equals n over fs, where n is an integer and fs is the sampling rate. If fs is strictly greater than 2B, then the original signal can be recovered exactly by the formula x of t equals the sum over all integers n of x of n over fs times sinc of fs times t minus n, where sinc of u is defined as sine of pi u over pi u, with the continuous extension sinc of zero equals one. This infinite sum converges for every t, and it reproduces x of t perfectly under the stated bandlimit and sampling assumptions.

The reason the theorem works can be understood by looking in the frequency domain. Sampling in time corresponds to convolving the spectrum with shifted copies of the sampling spectrum, creating spectral replicas centered at integer multiples of the sampling frequency fs. If fs is greater than 2B, the replicas do not overlap, so an ideal lowpass filter can isolate the original baseband spectrum. If fs is less than or equal to 2B, the replicas overlap, and the baseband spectrum becomes corrupted. This corruption is called aliasing, and it is generally irreversible without additional prior knowledge about the signal.

The strict inequality fs greater than 2B is important. If fs equals exactly 2B and the highest frequency component is nonzero at the boundary, the replicas can still touch or overlap, depending on the exact spectral shape, so perfect reconstruction is not guaranteed unless extra conditions hold. In practice, engineers often use an anti-aliasing filter before sampling to ensure the signal is effectively bandlimited to less than half the sampling rate, leaving a small guard band. The sampling frequency is then chosen comfortably above 2B to allow realizable filters with finite transition bands.

The Nyquist-Shannon sampling theorem has shaped countless technologies. It underpins digital audio, where audio signals are bandlimited to about 20 kHz and then sampled at 44.1 kHz or 48 kHz. It underpins digital photography and image sensors, where spatial bandlimits determine the required pixel density. It underpins telecommunications, radar, medical imaging, and essentially any field where physical signals are converted into digital form. Without the theorem, the design of analog-to-digital converters and digital-to-analog converters would lack a rigorous foundation.

The theorem also clarifies the limits of sampling. If a signal is not bandlimited, no finite sampling rate can guarantee perfect reconstruction from samples alone. That is why real systems use prefiltering. Conversely, if we know additional structure about a signal, such as sparsity in some transform domain, compressed sensing techniques can sometimes recover signals from fewer samples than the Nyquist rate would suggest, but that is a separate set of assumptions and not a violation of the Nyquist-Shannon theorem itself.

I want to make the reconstruction concrete. The Whittaker-Shannon interpolation formula can be implemented numerically for a finite set of samples. While a finite sum is only an approximation of the infinite ideal reconstruction, it is still useful for illustration and verification. In the code below, I construct a bandlimited signal by summing a few sinusoids whose frequencies are below half the chosen sampling rate, sample it uniformly, and then reconstruct the continuous waveform from those samples using the sinc interpolation formula. I then verify that the reconstruction matches the original signal at a dense grid of points and report the maximum absolute error. I also demonstrate aliasing by sampling the same signal below the Nyquist rate and showing that the reconstructed waveform differs from the original.

```python
import numpy as np

def sinc_reconstruct(samples, fs, t_out):
    """Whittaker-Shannon interpolation using a finite sample window."""
    n = np.arange(len(samples))
    t_n = n / fs
    diff = fs * (t_out[:, None] - t_n[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_matrix = np.sinc(diff)
    return sinc_matrix @ samples

# Build a bandlimited signal: sum of sinusoids below B = 5 Hz.
B = 5.0
fs = 25.0  # fs > 2B, comfortably above Nyquist rate of 10 Hz.
t_max = 4.0  # Wider sample window to reduce truncation at reconstruction edges.
t_fine = np.linspace(0.25, 0.75, 1001)  # Interior interval avoids edge effects.
x_fine = np.sin(2 * np.pi * 1.0 * t_fine)
x_fine = x_fine + 0.5 * np.sin(2 * np.pi * 3.0 * t_fine)
x_fine = x_fine + 0.3 * np.sin(2 * np.pi * 4.5 * t_fine)

# Uniform sampling over the wider window.
n_samples = int(np.floor(fs * t_max)) + 1
t_samples = np.arange(n_samples) / fs
x_samples = np.sin(2 * np.pi * 1.0 * t_samples)
x_samples = x_samples + 0.5 * np.sin(2 * np.pi * 3.0 * t_samples)
x_samples = x_samples + 0.3 * np.sin(2 * np.pi * 4.5 * t_samples)

# Reconstruct and compare.
x_recon = sinc_reconstruct(x_samples, fs, t_fine)
max_error = np.max(np.abs(x_recon - x_fine))
print(f"Sampling rate: {fs} Hz (Nyquist rate: {2*B} Hz)")
print(f"Max reconstruction error: {max_error:.2e}")

# Demonstrate aliasing by sampling below the Nyquist rate.
fs_alias = 8.0  # Below Nyquist rate 10 Hz.
n_alias = int(np.floor(fs_alias * t_max)) + 1
t_alias = np.arange(n_alias) / fs_alias
x_alias_samples = np.sin(2 * np.pi * 1.0 * t_alias)
x_alias_samples = x_alias_samples + 0.5 * np.sin(2 * np.pi * 3.0 * t_alias)
x_alias_samples = x_alias_samples + 0.3 * np.sin(2 * np.pi * 4.5 * t_alias)
x_alias_recon = sinc_reconstruct(x_alias_samples, fs_alias, t_fine)
alias_error = np.max(np.abs(x_alias_recon - x_fine))
print(f"Aliased sampling rate: {fs_alias} Hz")
print(f"Max aliased reconstruction error: {alias_error:.2e}")
```

When this script runs, the reconstruction at the higher sampling rate returns a very small error, reflecting the numerical truncation of the infinite sinc sum rather than any fundamental information loss. The reconstruction at the lower sampling rate returns a much larger error, confirming that aliasing corrupts the recovered waveform. This numerical demonstration aligns with the theoretical guarantee of the Nyquist-Shannon sampling theorem: sample a bandlimited signal above twice its maximum frequency, and the samples contain enough information to rebuild the original signal exactly.
