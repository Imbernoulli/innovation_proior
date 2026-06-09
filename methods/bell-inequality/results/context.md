# Context — turning the EPR completeness debate into an experiment

## Research question

For a pair of quantum systems prepared together and then separated to great distance, quantum mechanics predicts that the outcomes of measurements on the two systems are correlated, and in special cases perfectly correlated. Einstein, Podolsky and Rosen took this to mean the quantum state cannot be the whole story: if I can predict with certainty the result of a measurement on the far system by measuring the near one, and if nothing I do to the near system can physically disturb the far one, then the far result must already be fixed — an element of reality — even though the wave function does not contain it. The proposed cure is to *supplement* the wave function with additional ("hidden") variables that restore both determinism and locality.

The question that matters is whether this is a genuine physical issue or a matter of taste. Is there any conceivable measurement whose outcome would come out one way if the world is governed by such supplemented variables and a different way if it is governed by quantum mechanics? A satisfactory resolution would have to settle the matter by experiment, not by philosophical argument — and to do so for *any* such hidden-variable account, not one specific model, and in a form that makes clear which apparatus assumptions an experiment has to control.

## Background

**The EPR argument (Einstein, Podolsky, Rosen 1935).** EPR set down a sufficient criterion of reality: "If, without in any way disturbing a system, we can predict with certainty the value of a physical quantity, then there exists an element of physical reality corresponding to this physical quantity." They then exhibit two systems I and II that interact for a while and separate. The joint wave function can be expanded in eigenfunctions of an observable A of system I, with the coefficients themselves eigenfunctions of some observable P of system II; expanded instead in eigenfunctions of a *different*, noncommuting observable B of system I, the coefficients are eigenfunctions of a noncommuting observable Q of system II. Measuring A on I lets one predict P on II with certainty; measuring B on I lets one predict Q on II with certainty. Since the choice on I cannot disturb the distant II, both P and Q are elements of reality of II — yet no quantum state assigns simultaneous sharp values to noncommuting P and Q. Conclusion: the wave-function description is incomplete.

**Bohm's spin version (1951) and Bohm–Aharonov (1957).** The original EPR used continuous position/momentum, awkward to picture. Bohm recast it with two spin-½ particles in the singlet state, flying apart toward two Stern–Gerlach magnets. Each particle is deflected up or down; which one is unpredictable, but the two are always opposite when the magnets are aligned. This is the form everyone reasons about: discrete ±1 outcomes, a single adjustable knob (the magnet orientation) on each side. Bohm and Aharonov stressed that a decisive version would change the magnet settings *during the flight* of the particles, so that no slower-than-light signal could coordinate the two ends.

**The quantum correlations of the singlet.** For the spin singlet |ψ⟩ = (1/√2)(|↑↓⟩ − |↓↑⟩), the joint expectation of spin components along directions **a** and **b** is ⟨ψ|(σ₁·**a**)(σ₂·**b**)|ψ⟩ = −**a**·**b** = −cos θ, where θ is the angle between the settings. At θ = 0 this is −1 (perfect anticorrelation: aligned magnets always give opposite deflections); at θ = π it is +1; at θ = π/2 it is 0. This smooth cosine, and in particular the perfect anticorrelation at θ = 0, is the load-bearing empirical prediction — established quantum mechanics, knowable before any hidden-variable analysis.

**von Neumann's "impossibility" (1932) and its standing.** Against hidden variables stood a celebrated theorem of von Neumann, widely read as a proof that no hidden-variable completion of quantum mechanics is mathematically possible — and so as having closed the EPR question in advance. Its essential assumption is that the expectation functional is *additive over all observables*: Exp(R + S) = Exp(R) + Exp(S), required to hold even when R and S are represented by *noncommuting* operators and so are not simultaneously measurable. For quantum-mechanical ensembles this additivity is a true (if peculiar) fact. von Neumann demanded it also of the hypothetical dispersion-free (hidden-variable) states, and from it derived a contradiction. Jauch and Piron (1963) and the corollary drawn from Gleason (1957) reach the same negative conclusion from analogous assumptions about commuting projection operators and their values.

**A concrete single-spin counterexample exists.** One can build, explicitly, a hidden-variable scheme that reproduces all quantum predictions for a single spin-½ system: let a real parameter λ together with the spinor fix every measurement outcome, averaging over λ to recover the quantum statistics. Such a model exists and is consistent — so something in the "impossibility" hypotheses must be too strong, and the standing of those theorems against *every* hidden-variable theory is in doubt.

## Baselines

**von Neumann 1932 (the standing no-go result).** Idea: characterize states by an expectation functional Exp on observables; assume linearity/additivity of Exp over arbitrary Hermitian operators; show no dispersion-free Exp exists, so no hidden variables. Gap: the additivity axiom is imposed on individual hidden states as well as on quantum ensembles, and it is far from obvious that an individual hidden state owes the same property; the single-spin counterexample above already shows the theorem's hypotheses are too strong somewhere.

**Jauch–Piron 1963 / Gleason-corollary.** Idea: work with projection operators (yes/no observables, eigenvalues 0,1); assume additivity of expectation values for *commuting* projections plus a logical-looking conjunction axiom (if ⟨a⟩=⟨b⟩=1 then ⟨a∧b⟩=1); derive a contradiction with dispersion-free states. Gleason's theorem gives a sharper version in dimension ≥ 3. Gap: the same worry recurs — assumptions tailored to quantum ensembles are pressed onto the individual hidden states whose existence is in question, so it is unclear which class of hidden-variable theories is genuinely excluded.

**Bohm's 1952 hidden-variable theory.** Idea: an explicit, deterministic, hidden-variable completion of wave mechanics (particle positions as the hidden variables, guided by the wave function) that reproduces all quantum predictions — a living counterexample to the claim that hidden variables are impossible. Its trajectory equations are, however, grossly *nonlocal*: when the joint wave function is non-factorable, the trajectory of particle 1 depends on the distant analyzing field acting on particle 2. Gap: it shows hidden variables *can* exist, but at the price of explicit action at a distance — leaving open whether *every* hidden-variable account must be nonlocal, or whether a *local* one is possible.

**The "naive realist" model (the obvious local picture).** Idea: the source emits each pair with oppositely oriented magnetic axes (chosen at random); each Stern–Gerlach magnet deflects a particle up or down according to whether its axis points more nearly along or against the local field. This reproduces the perfect anticorrelation at aligned settings trivially and locally. Gap: whether it also matches the quantum prediction away from the aligned and orthogonal settings has not been worked out — only the special cases have been checked.

## Evaluation settings

The natural physical realization is Bohm's gedanken experiment: a source emitting correlated pairs toward two well-separated analyzers, each with a single adjustable orientation, each yielding a binary outcome. Two laboratory embodiments exist. Wu–Shaknov (1950) measured the polarization correlation of the 0.5-MeV γ-rays from positronium annihilation, via Compton scattering — but Compton scattering is only a statistically weak index of linear polarization, so it cannot force a sharp binary outcome suited to a decisive test. Kocher–Commins (1967) measured the polarization correlation of the visible photon pair from the 6¹S₀–4¹P₁–4¹S₀ cascade of calcium, using Polaroid-type linear polarizers and standard coincidence counting; this gives genuine two-channel (emerge/not-emerge) outcomes and is the right kind of setup, but as performed it placed the polarizers only at relative orientations 0° and 90°. The yardstick quantities are the coincidence-detection rate R(a,b) as a function of the two polarizer orientations, the single-arm rates with one polarizer removed, and the rate R₀ with both removed; from these one forms the correlation P(a,b) of the ±1 outcomes. The relevant control is whether the two settings can be changed fast enough, and the arms separated far enough, that no light-speed signal can pass from one analyzer to the other within the measurement — the condition Bohm and Aharonov insisted on.

## Code framework

A bare correlation-experiment harness. We can simulate any candidate local model and compute, from its outcome functions, the correlations it predicts at chosen settings — and separately compute the quantum prediction — so the two can be compared. The empty slot is below.

```python
import numpy as np

# Quantum prediction for the singlet (established, used as the yardstick)
def quantum_correlation(theta):
    # E(a, b) for the spin singlet as a function of the angle between settings
    return -np.cos(theta)

class LocalHiddenVariableModel:
    """A candidate local model: a shared variable lambda, and two LOCAL
    outcome functions, each depending only on its own setting."""
    def sample_lambda(self, n):
        raise NotImplementedError  # TODO: the source's distribution rho(lambda)
    def outcome_A(self, a, lam):
        raise NotImplementedError  # TODO: +-1, depends on a and lam only (not b)
    def outcome_B(self, b, lam):
        raise NotImplementedError  # TODO: +-1, depends on b and lam only (not a)

def correlation_of_model(model, a, b, n=10**6):
    lam = model.sample_lambda(n)
    A = model.outcome_A(a, lam)
    B = model.outcome_B(b, lam)
    return float(np.mean(A * B))

def distinguish_local_from_quantum(*correlations):
    """Given correlations measured at several settings, decide whether they
    are compatible with the local hidden-variable form above."""
    # TODO
    pass
```
