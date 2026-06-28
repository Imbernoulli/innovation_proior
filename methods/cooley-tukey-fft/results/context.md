# Context: Computing the Discrete Fourier Transform of n Samples

## Research question

Given $N$ equally spaced samples $A(0), A(1), \dots, A(N-1)$ of a signal, compute all $N$ discrete Fourier coefficients

$$X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}, \qquad W = e^{-2\pi i / N}, \qquad j = 0, 1, \dots, N-1.$$

$W$ is a primitive $N$-th root of unity, so $W^N = 1$. The transform is the fundamental tool for extracting the frequency content of a sampled signal, and for computing discrete convolutions and correlations, which by the convolution theorem reduce to a pointwise product in the transform domain.

Read literally, the formula is a multiplication of the data vector $A$ by the $N \times N$ matrix whose $(j,k)$ entry is $W^{jk}$. That matrix multiplication takes $N^2$ complex multiplications and about $N^2$ additions. The sizes that matter in practice — spectroscopy, radar, seismic exploration, X-ray scattering, harmonic analysis of tides and orbits, time-series with thousands to millions of points — make the operation count the central quantity: at $N = 10^6$, $N^2 = 10^{12}$ operations. The question is whether the DFT can be computed in fundamentally fewer than $N^2$ operations.

Two concrete applications around 1963 set the scale. The route into the computation runs through Richard Garwin, who brings John Tukey's factor-$N$ split idea from President Kennedy's Scientific Advisory Council. First, nuclear-test-ban verification: a credible US/Soviet test-ban regime requires detecting Soviet underground tests *without* on-site inspection, and the proposed scheme is to ring the Soviet Union with seismometers and analyze the resulting seismological time series — separating a clandestine explosion from an earthquake means computing spectra of long records from many stations. Second, the application handed forward as the immediate programming target is a three-dimensional solid-state calculation: determining periodicities in the spin orientations of helium-3 in a crystal requires a 3-D discrete Fourier transform on a grid that is a power of two along each axis. In both, the controlling quantity is raw operation count.

## Background

The continuous trigonometric series goes back to Euler, who gave formulas for the coefficients of a series representation of a function, and to Clairaut (1754), who published what appears to be the earliest explicit formula for computing series coefficients from equally spaced samples — a cosine DFT. Lagrange extended this to finite sine series. Clairaut and Lagrange were driven by orbital mechanics: given a finite set of equally spaced observations of a periodic phenomenon, recover the harmonic coefficients. In modern terms, an even periodic $f$ of period one is represented as $f(x) = \sum_k a_k \cos 2\pi k x$, and forcing $f$ to equal the observed samples at $x_n = n/N$ makes the $\{a_k\}$ exactly the cosine DFT of the observations. From its origin, the object of interest is the *interpolation* of periodic data.

The structure lives entirely in the matrix $[W^{jk}]$. Its entries are not arbitrary: each is a power of one root of unity $W$, and the exponent only matters modulo $N$ because $W^N = 1$. Three facts about roots of unity carry weight:

- **Periodicity.** $W^{m+N} = W^m$. The exponents wrap.
- **Even powers.** If $N$ is even, $W^2 = e^{-2\pi i/(N/2)}$ is itself a primitive $(N/2)$-th root of unity.
- **Half-index value.** $W^{N/2} = e^{-\pi i} = -1$, so $W^{j+N/2} = -W^j$.

These mean the $N^2$ matrix entries take far fewer than $N^2$ distinct values.

The nineteenth-century computers (human ones) attacked this practically. Their methods grouped terms in the trigonometric series sharing a common multiplicative coefficient, so a block of $n$ multiplications and $n-1$ additions collapsed to a single multiply and $n-1$ adds. This cut constant factors.

A diagnostic example fixes scale. By hand, a 64-point Fourier analysis was a substantial undertaking; in the 1940s, doubling tricks were valued because they cut the labor of going from a 32-point to a 64-point analysis to little more than the 32-point work itself, and because the redundant recomputation they exposed also served as a running accuracy check. The empirical fact on the table is concrete: the cost of a hand or machine DFT grows like $N^2$.

## Baselines

**Direct evaluation (the matrix multiply).** Compute each $X(j)$ as $\sum_k A(k) W^{jk}$ independently. Precompute the powers of $W$ once. Cost: $N^2$ complex multiplications. Correct, simple, easy to store, and the universal fallback.

**Term-grouping / common-factor schemes (Carlini 1828 for $n=12$, Hansen 1835 for $n=64$, Archibald Smith 1846, used by Kelvin and by G. H. Darwin for tidal analysis).** Collect terms in the series that multiply the same $\cos$ or $\sin$ value and add them before multiplying. This removes repeated multiplications by equal trigonometric values and was the standard hand technique through the nineteenth century. It typically only computed harmonics up to the fourth, because measurement noise swamped higher ones, so these were fixed-size tabulated recipes.

**Gauss's interpolation factorization (1805, published posthumously).** For orbit interpolation, a composite number of equally spaced samples can be arranged as a product of shorter cyclic grids. The trigonometric notation is different, but the substance is a decomposition of one long finite Fourier calculation into shorter ones, with phase corrections between the stages. It remained within interpolation tables for orbital data.

**Runge's doubling (Runge 1903/1905; Runge & König).** A length-$2N$ transform is assembled from two length-$N$ transforms with about $N$ extra operations. It builds $2N$ from $N$ and was presented as a doubling rule.

**Stumpff's doubling and tripling (Stumpff 1939).** Extends Runge with a tripling rule alongside doubling, and on one page suggests the generalization to an arbitrary multiple. Framed as discrete multiplier rules (×2, ×3).

**The Danielson–Lanczos doubling lemma (Danielson & Lanczos 1942, for X-ray scattering from liquids).** They observe that a length-$N$ DFT equals the sum of two length-$N/2$ DFTs, one formed from the even-indexed samples and one from the odd-indexed samples, combined with a power of $W$. Because the lemma reproduces a length-$N$ transform from two of length $N/2$, it can be applied again to each half. They present and use it as a doubling/halving device tied to powers of two.

**Good's prime-factor algorithm (Good 1958).** A factorization of composite $N$. When $N = N_1 N_2$ with $N_1, N_2$ *coprime*, the Chinese Remainder Theorem gives an index map under which the length-$N$ DFT becomes a true two-dimensional $N_1 \times N_2$ DFT with **no** cross-term multiplier between the stages. It applies to coprime factors.

## Evaluation settings

The natural yardstick is the operation count — complex multiplications and additions — as a function of $N$, against the $N^2$ baseline, for the sizes that arise in practice. The relevant size regimes are highly composite $N$, especially $N = 2^L$ (a power of two); general composite $N = \prod r_i$; and prime $N$, where no nontrivial factorization exists. Correctness is checked by agreement with the direct matrix evaluation on the same input to floating-point tolerance. The application settings are harmonic analysis of equally spaced samples (orbital and tidal data, spectroscopy, X-ray scattering) and discrete convolution/correlation via the transform domain. Inputs are complex sequences of length $N$; the inverse transform is the same computation with the conjugate root $W^{-1} = e^{+2\pi i/N}$ and an overall $1/N$ normalization.

## Code framework

The deliverable is a single self-contained C++17 program. It reads from stdin: `N` (a power of two) followed by `N` complex samples as `re im` pairs. It writes to stdout the `N` transform coefficients, one `re im` pair per line, using fixed six-decimal formatting. The skeleton supplies the I/O shell and leaves the computation slot open.

```cpp
#include <bits/stdc++.h>
using namespace std;

using cd = complex<double>;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int N;
    if (!(cin >> N)) return 0;
    if (N < 1 || (N & (N - 1))) {
        cerr << "length must be a power of two\n";
        return 1;
    }
    vector<cd> A(N);
    for (int i = 0; i < N; ++i) {
        double re, im;
        cin >> re >> im;
        A[i] = cd(re, im);
    }
    vector<cd> X(N);
    // TODO: compute the N output coefficients from A.
    cout << fixed << setprecision(6);
    for (int j = 0; j < N; ++j)
        cout << X[j].real() << ' ' << X[j].imag() << '\n';
    return 0;
}
```
