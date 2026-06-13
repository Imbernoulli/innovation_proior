# Context: Computing the Discrete Fourier Transform of n Samples

## Research question

Given $N$ equally spaced samples $A(0), A(1), \dots, A(N-1)$ of a signal, compute all $N$ discrete Fourier coefficients

$$X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}, \qquad W = e^{-2\pi i / N}, \qquad j = 0, 1, \dots, N-1.$$

$W$ is a primitive $N$-th root of unity, so $W^N = 1$. The transform is the fundamental tool for extracting the frequency content of a sampled signal, and for computing discrete convolutions and correlations, which by the convolution theorem reduce to a pointwise product in the transform domain.

Read literally, the formula is a multiplication of the data vector $A$ by the $N \times N$ matrix whose $(j,k)$ entry is $W^{jk}$. That matrix multiplication takes $N^2$ complex multiplications and about $N^2$ additions. For the sizes that matter in practice — spectroscopy, radar, seismic exploration, X-ray scattering, harmonic analysis of tides and orbits, time-series with thousands to millions of points — the quadratic cost is the bottleneck. At $N = 10^6$, $N^2 = 10^{12}$ operations is the difference between roughly seconds and roughly weeks of machine time. The question is whether the DFT can be computed in fundamentally fewer than $N^2$ operations.

Two concrete applications around 1963 make the wall sharp rather than abstract. The route into the computation runs through Richard Garwin, who brings John Tukey's factor-$N$ split idea from President Kennedy's Scientific Advisory Council. First, nuclear-test-ban verification: a credible US/Soviet test-ban regime requires detecting Soviet underground tests *without* on-site inspection, and the proposed scheme is to ring the Soviet Union with seismometers and analyze the resulting seismological time series — separating a clandestine explosion from an earthquake means computing spectra of long records from many stations, a transform volume that the computers of the day cannot process by direct evaluation. Second, the application handed forward as the immediate programming target is a three-dimensional solid-state calculation: determining periodicities in the spin orientations of helium-3 in a crystal requires a 3-D discrete Fourier transform on a grid that is a power of two along each axis, where the per-axis lengths again push the direct $N^2$ cost past what the available machines can deliver. In both, the bottleneck is not accuracy but raw operation count, and only a sub-quadratic algorithm makes the computation feasible at all.

## Background

The continuous trigonometric series goes back to Euler, who gave formulas for the coefficients of a series representation of a function, and to Clairaut (1754), who published what appears to be the earliest explicit formula for computing series coefficients from equally spaced samples — a cosine DFT. Lagrange extended this to finite sine series. Clairaut and Lagrange were driven by orbital mechanics: given a finite set of equally spaced observations of a periodic phenomenon, recover the harmonic coefficients. In modern terms, an even periodic $f$ of period one is represented as $f(x) = \sum_k a_k \cos 2\pi k x$, and forcing $f$ to equal the observed samples at $x_n = n/N$ makes the $\{a_k\}$ exactly the cosine DFT of the observations. So from its origin, the object of interest is the *interpolation* of periodic data, and the cost of computing it by hand was always the practical wall.

The structure to exploit lives entirely in the matrix $[W^{jk}]$. Its entries are not arbitrary: each is a power of one root of unity $W$, and the exponent only matters modulo $N$ because $W^N = 1$. Three facts about roots of unity carry all the weight:

- **Periodicity.** $W^{m+N} = W^m$. The exponents wrap.
- **Even powers.** If $N$ is even, $W^2 = e^{-2\pi i/(N/2)}$ is itself a primitive $(N/2)$-th root of unity.
- **Half-index value.** $W^{N/2} = e^{-\pi i} = -1$, so $W^{j+N/2} = -W^j$.

These mean the $N^2$ matrix entries take far fewer than $N^2$ distinct values, and the same partial sums recur across different output indices. That redundancy is the latent inefficiency a fast method must convert into reuse.

The nineteenth-century computers (human ones) attacked this practically. Their methods grouped terms in the trigonometric series sharing a common multiplicative coefficient, so a block of $n$ multiplications and $n-1$ additions collapsed to a single multiply and $n-1$ adds. This cut constant factors but did not change the order of growth — it is not a sub-quadratic method, because it never decomposes a long DFT into genuinely shorter DFTs.

A diagnostic example fixes scale. By hand, a 64-point Fourier analysis was a substantial undertaking; in the 1940s, doubling tricks were valued not only because they cut the labor of going from a 32-point to a 64-point analysis to little more than the 32-point work itself, but because the redundant recomputation they exposed also served as a running accuracy check. The empirical fact on the table before any fast algorithm exists is therefore concrete: the cost of a hand or machine DFT grows like $N^2$, and the root-of-unity structure means most of that arithmetic is recomputation.

## Baselines

**Direct evaluation (the matrix multiply).** Compute each $X(j)$ as $\sum_k A(k) W^{jk}$ independently. Precompute the powers of $W$ once. Cost: $N^2$ complex multiplications. Correct, simple, easy to store, and the universal fallback. Its gap is exactly its order: it never reuses a partial sum from one output row in another, although the periodicity of $W$ guarantees enormous reuse is available.

**Term-grouping / common-factor schemes (Carlini 1828 for $n=12$, Hansen 1835 for $n=64$, Archibald Smith 1846, used by Kelvin and by G. H. Darwin for tidal analysis).** Collect terms in the series that multiply the same $\cos$ or $\sin$ value and add them before multiplying. This removes repeated multiplications by equal trigonometric values and was the standard hand technique through the nineteenth century. It typically only computed harmonics up to the fourth, because measurement noise swamped higher ones, so these were fixed-size tabulated recipes, not general procedures. The gap: the savings are a bounded constant factor; the work still grows as $N^2$ because no shorter DFT is ever formed.

**Gauss's interpolation factorization (1805, published posthumously).** For orbit interpolation, a composite number of equally spaced samples can be arranged as a product of shorter cyclic grids. The trigonometric notation is different, but the substance is a decomposition of one long finite Fourier calculation into shorter ones, with phase corrections between the stages. Its gap for the present machine problem is historical and practical: it remained buried in interpolation tables and was not available as a general programming recipe for complex DFTs, arbitrary composite lengths, and regular machine storage.

**Runge's doubling (Runge 1903/1905; Runge & König).** A length-$2N$ transform can be assembled from two length-$N$ transforms with about $N$ extra operations. This is a real order-reducing step — it forms shorter DFTs and recombines them — but only in the doubling direction: it builds $2N$ from $N$, so it is naturally limited to power-of-two-style growth and was presented as a doubling rule rather than a general factorization.

**Stumpff's doubling and tripling (Stumpff 1939).** Extends Runge with a tripling rule alongside doubling, and on one page suggests the generalization to an arbitrary multiple. Still framed as discrete multiplier rules (×2, ×3) rather than a single recursive decomposition for general composite $N$.

**The Danielson–Lanczos doubling lemma (Danielson & Lanczos 1942, for X-ray scattering from liquids).** The cleanest of the doubling derivations. They observe that a length-$N$ DFT equals the sum of two length-$N/2$ DFTs, one formed from the even-indexed samples and one from the odd-indexed samples, combined with a power of $W$. Because the lemma reproduces a length-$N$ transform from two of length $N/2$, it can be applied again to each half. The gap they leave open: they present and use it as a doubling/halving device tied to powers of two, and stop short of a single statement covering general composite $N$ and the bookkeeping (index reordering, in-place butterflies) that makes it a machine algorithm.

**Good's prime-factor algorithm (Good 1958).** A different factorization of composite $N$. When $N = N_1 N_2$ with $N_1, N_2$ *coprime*, the Chinese Remainder Theorem gives an index map under which the length-$N$ DFT becomes a true two-dimensional $N_1 \times N_2$ DFT with **no** cross-term multiplier between the stages. Elegant and multiplication-light, but it works only for coprime factors, so it cannot recurse on $N = 2^L$ (the factors $2, 2, \dots$ are not coprime), and the coprime-only restriction means it does not give a single uniform recursion for arbitrary composite sizes.

## Evaluation settings

The natural yardstick is the operation count — complex multiplications and additions — as a function of $N$, against the $N^2$ baseline, for the sizes that arise in practice. The relevant size regimes are highly composite $N$, especially $N = 2^L$ (a power of two), where any decomposition can be applied uniformly all the way down; general composite $N = \prod r_i$; and the adversarial case of prime $N$, where no nontrivial factorization exists. Correctness is checked by agreement with the direct matrix evaluation on the same input to floating-point tolerance. The application settings that motivate the work are harmonic analysis of equally spaced samples (orbital and tidal data, spectroscopy, X-ray scattering) and discrete convolution/correlation via the transform domain. Inputs are complex sequences of length $N$; the inverse transform is the same computation with the conjugate root $W^{-1} = e^{+2\pi i/N}$ and an overall $1/N$ normalization.

## Code framework

The primitives already exist: complex arithmetic, a routine to form powers of a root of unity, and the direct transform as both a baseline and a small-$N$ correctness oracle. The open slots are a faster transform, whatever size validation it requires, and the inverse wrapper.

```python
import cmath

def root_of_unity(N, sign=-1):
    # principal N-th root W = exp(sign * 2 pi i / N); sign=-1 forward, +1 inverse
    return cmath.exp(sign * 2j * cmath.pi / N)

def dft_direct(A, sign=-1):
    # baseline O(N^2) transform; also the trusted small-N base case
    N = len(A)
    W = root_of_unity(N, sign)
    return [sum(A[k] * W**(j * k) for k in range(N)) for j in range(N)]

def _validate_length(n):
    # TODO: enforce whatever size condition the faster transform needs.
    pass

def transform(A, sign=-1):
    # TODO: fill the faster transform slot; it must agree with dft_direct.
    pass

def inverse_transform(X):
    # TODO: run the conjugate-root transform and apply the normalization.
    pass
```
