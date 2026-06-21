# Context — Statistical Communication

## Research question

A communication engineer wants to send a message produced at one point and reproduce it, exactly or
approximately, at another. By the late 1940s the engineering toolbox for doing this — modulation schemes
such as PCM and PPM that trade bandwidth for signal-to-noise ratio — had grown rich, but there was no
*general* theory that answered the underlying quantitative questions. Three questions in particular had no
agreed answer:

1. **How much "information" is there to send?** Messages have meaning, but meaning is private to the
   communicators and varies with their shared language; it cannot be the basis of an engineering measure.
   What can be measured, and in what units?
2. **How much can a source be compressed?** Real sources — English text, for instance — are highly
   redundant: letters are unequal in frequency and strongly correlated. How far can that redundancy be
   squeezed out, and is there a hard floor?
3. **What can a *noisy* channel deliver?** Noise corrupts symbols, so the output no longer determines the
   input. Intuition said that to drive the error probability toward zero you must add ever more redundancy
   and so let the transmission rate fall toward zero. Is that true? Is there a well-defined maximum rate at
   which a given channel can carry information, and if so can it be approached *with* arbitrarily small error?

## Background

**The logarithmic measure (Nyquist, Hartley).** The first quantitative handle came from telegraphy.
Nyquist (1924, "Certain Factors Affecting Telegraph Speed") found that the speed at which "intelligence"
can be sent over a circuit, for a fixed signaling rate, obeys

    W = K log m,

where m is the number of distinct current values a signal element may take and K a constant. The logarithm
appears because doubling the number of code values multiplies the number of distinguishable characters, but
adds only a fixed increment of transmitting power.

Hartley (1928, "Transmission of Information") made the central conceptual move: a
measure of information must be "based on physical as contrasted with psychological considerations." He
argued that communication is a sequence of *selections* from an agreed set of symbols; each selection
eliminates the alternatives not chosen ("Apples are red" — the first word eliminates other objects, etc.). The
*meaning* of the symbols — the "psychological factors" — varies with the parties and must be discarded;
what remains is the number of possible symbol sequences. With s symbols available at each of n independent
selections there are s^n sequences. Requiring the information to be additive in the number of selections,
H = Kn, and requiring that two systems with the same number of possible sequences (s1^{n1} = s2^{n2})
carry the same information, forces K = K0 log s, hence

    H = n log s = log(s^n),

"the logarithm of the number of possible symbol sequences." Setting n = 1, one selection from s equally
likely symbols carries log s of information.

**Entropy in statistical mechanics (Boltzmann, Gibbs).** In the kinetic theory of gases the quantity

    S = −k Σ_i p_i log p_i

— with p_i the probability of the system being in cell i of its phase space — measures the disorder or
uncertainty of a statistical state; it is the quantity in Boltzmann's H-theorem and in Gibbs's statistical
mechanics (as in Tolman's *Principles of Statistical Mechanics*). It is maximal for the uniform distribution
and additive over independent subsystems. This form is part of the standing mathematical culture of the time.

**Signals and noise as stochastic processes (Wiener, Kolmogorov).** Contemporaneously, Wiener and (earlier)
Kolmogorov treated signal and noise as random processes and solved the problem of optimal linear
filtering/prediction: extract the best estimate of a signal from noisy data in the mean-square sense. This
established the *probabilistic* modeling of communication waveforms and the continuum/spectral machinery.

**The diagnostic facts about real sources.** Natural-language sources are conspicuously redundant: in English
the letters are far from equiprobable, and successive letters are strongly dependent (digram, trigram
statistics). One can *see* this by generating random text from successively higher-order letter statistics —
the output grows steadily more English-like as more structure is included. This redundancy is a pre-existing,
measurable fact about sources, not a property of a particular encoding scheme.

## Baselines

- **Nyquist's W = K log m (1924).** Establishes the logarithm as the natural measure of signaling capability,
  as transmission *speed* for equiprobable code values; deterministic, no source model, no noise.
- **Hartley's H = log(s^n) = n log s (1928).** The first explicit "amount of information" stripped of meaning,
  via additivity over selections; assumes equally likely and independent symbols.
- **Boltzmann/Gibbs entropy S = −k Σ p log p.** A standard mathematical form from statistical mechanics, with
  its standard properties (maximized at uniform, additive over independent systems) — living in physics.
- **Wiener–Kolmogorov filtering/prediction.** Probabilistic, optimal in mean-square error, the state of the art
  for getting signal out of noise; an estimation theory for recovering waveforms from noisy observations.

## Evaluation settings

The natural yardsticks are the standard objects of communication engineering, all pre-existing: a discrete
source emitting symbols from a finite alphabet (the canonical example being English text and its
letter/word statistics, including printed-English redundancy estimates); a discrete channel specified by its
transition probabilities (the binary symmetric channel with a given crossover probability as the textbook
case, and noiseless constrained channels such as telegraph codes with symbol-duration constraints); the
continuous (band-limited, white-Gaussian-noise) channel parameterized by bandwidth W, signal power and
noise power; and figures of merit such as description length per symbol, transmission rate versus
error frequency, and bandwidth-versus-signal-to-noise trade-offs. These are the settings within
which a quantitative theory of information and coding limits would be stated and checked.

## Code framework

A finite-alphabet source-coding harness already has the statistical primitives it needs: samples, empirical
frequencies, a finite source alphabet, binary labels, and an expected code length. What remains open is filled
in below.

```python
from collections import Counter

def empirical_distribution(symbols):
    """Relative frequencies of symbols in a sample."""
    n = len(symbols)
    if n == 0:
        raise ValueError("sample must be non-empty")
    return {s: c / n for s, c in Counter(symbols).items()}

def per_symbol_measure(distribution):
    """A per-symbol number summarizing the source distribution."""
    # TODO
    pass

def binary_code(distribution):
    """A binary label for each positive-probability symbol."""
    # TODO
    pass

def average_length(code, distribution):
    """Expected binary symbols per source symbol."""
    # TODO
    pass

def source_coding_check(symbols):
    """Relate the per-symbol measure to the average length of a binary code."""
    # TODO
    pass
```
