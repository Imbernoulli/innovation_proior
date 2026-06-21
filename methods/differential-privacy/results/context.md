# Context: privacy-preserving analysis of statistical databases

## Research question

A trusted curator holds a database of sensitive records — one row per person — and wants to release useful aggregate statistics (counts, histograms, means, covariances, graph properties) without compromising the privacy of any individual whose data is in it. The question is foundational and definitional before it is algorithmic: **what should "preserving privacy" mean here, precisely enough that one can prove a mechanism satisfies it?** The setting involves an adversary who may hold arbitrary auxiliary information (other databases, newspapers, prior medical studies, gossip) and may combine several released answers, or one released answer with outside knowledge. At the same time the definition has to coexist with the release of genuine statistical information: releasing nothing, or pure noise, is trivially private and useless.

## Background

**The statistical-disclosure-control tradition.** Protecting respondents while publishing tabulations is an old problem in official statistics and, since the late 1970s, in computer science. A detailed catalogue of the techniques explored through 1989 is the survey of Adam and Wortmann (1989), which organizes the field into three families: **query restriction** (refuse queries on too-small sets, limit query overlap, audit a query log and disallow compromising queries), **input/data perturbation** (randomly modify the underlying records, e.g. data swapping or adding a fixed random perturbation to each entry, then answer from the modified data), and **output perturbation** (compute the true answer, then report a noisy or rounded version). The classical antecedent closest in spirit to noise-based release is Denning's work.

**Combination attacks.** Two large allowed queries that differ in a single record can be subtracted to reveal that record's value. Restricting query overlap or auditing the log addresses only a small number of queries, and the auditor's *refusals* themselves carry information — a refusal, combined with the answers to permitted queries, can finish a compromise; deciding when to refuse is NP-hard. Removing names and identifiers is a separate matter: a handful of attributes (gender, approximate age, ZIP, marital status) re-identifies most people, and a rare attribute identifies them outright — the linkage attacks demonstrated by Sweeney, which rekindled computer scientists' interest in the problem.

**The reconstruction result (Dinur and Nissim, 2003).** Model a database as bits d₁,…,dₙ ∈ {0,1}; a statistical query is a subset q ⊆ [n] answered by Σᵢ∈q dᵢ plus perturbation. Dinur and Nissim give a **polynomial-time reconstruction algorithm**: from answers to enough (random) subset-sum queries whose perturbation is only o(√n), an adversary recovers a candidate database within Hamming distance εn of the real one — all but a tiny fraction of the private bits. They name the failure they rule out **blatant non-privacy**: a bounded adversary recovering a 1−ε fraction of the entries. The quantitative consequence: to resist unrestricted queries one must add perturbation of magnitude **Ω(√n)** (linear in n against a computationally unbounded adversary). The technique is essentially list-decoding / a Goldreich–Levin-style recovery from noisy linear measurements. Noise is *necessary*, and its *magnitude* is what governs whether reconstruction succeeds.

**The Dalenius desideratum.** Dalenius (1977) articulated a goal: *access to a statistical database should not enable one to learn anything about an individual that could not be learned without access.* This is the database analogue of the cryptographic notion of **semantic security** that Goldwasser and Micali (1982) defined for encryption five years later — nothing learnable about a plaintext from the ciphertext that could not be learned without it. There is a disanalogy between the two settings. Semantic security holds because the ciphertext is *useless* to anyone but the key-holder; the auxiliary-information generator has no idea what ciphertext the eavesdropper will see. A statistical database is *designed* to convey information, and there is no decryption key separating the legitimate user from the adversary — they are one and the same. Whoever knows the data knows roughly what the user will learn. The Terry-Gross-height example makes this concrete: release the average height of Lithuanian women, hand the adversary "Terry Gross is two inches shorter than that average," and Terry Gross's exact height is disclosed — whether or not Terry Gross is in the database at all.

**Privacy by randomized process (Warner, 1965).** Warner's randomized response collects statistics on stigmatized behavior: each respondent flips a coin and, depending on the outcome, either answers truthfully or answers a fixed random way, so that any individual "Yes" carries plausible deniability — it would have occurred with non-trivial probability regardless of the truth — while the population proportion remains an unbiased, recoverable estimate because the noise process is known. The distribution of an individual's reported answer is only mildly shifted by their true bit. The scheme also illustrates that randomization is built in: any non-trivial deterministic mechanism that ever distinguishes two databases distinguishes two that differ in a single row, and an adversary who knows the database is one of those two learns the differing row.

**The cryptographic lens.** A definition that must hold against arbitrary side information naturally borrows the worst-case, adversary-explicit style of cryptography: state precisely what the adversary may know and may do, and bound a worst-case quantity rather than an average. The cryptographic measures themselves — statistical/total-variation distance; computational indistinguishability — are available starting points.

## Baselines

- **Query restriction / auditing.** Allow only queries over large sets, cap query overlap, or audit the query log and refuse compromising queries.

- **k-anonymity (Sweeney) and de-identification.** Generalize/suppress quasi-identifiers so each released record is indistinguishable from at least k−1 others; strip names and SSNs. A syntactic property of the released table.

- **Input perturbation (Agrawal–Srikant and the data-swapping/fixed-perturbation line).** Randomly modify records, then answer from the modified data; privacy quantified by the magnitude of noise added to a value, or by estimator variance. Evfimievsky et al. study what a distribution-aware adversary can infer; the original input distribution can be reconstructed from the perturbed data.

- **Output perturbation (Denning; varying-perturbation and rounding schemes).** Add noise to (or round) the true answer; increase variance on repeated queries. The amount of noise was tied to the query dimension or to ad-hoc averages, so high-dimensional releases (histograms, contingency tables, covariances) used noise scaled to the dimension. Dinur–Nissim places the family in the context of the Ω(√n) noise bound.

- **Semantic-security-style "learn nothing about an individual" (the Dalenius goal).** The aspirational baseline definition, formalized as a relaxed semantic security.

- **Statistical / computational indistinguishability (imported from cryptography).** Bound total-variation distance or computational advantage between the views induced by two databases. Total-variation distance is an average-case measure.

## Evaluation settings

The natural yardsticks are about *what a curator must answer and how robustly*:
- **Query classes:** subset-sum / counting queries ("how many records satisfy predicate P?"), their weighted (linear) and fractional forms; **histograms** and **contingency tables** (partition the universe into d bins, count each — high-dimensional output); **means and covariance matrices** of per-record feature vectors; and **graph/holistic functionals** over a database whose rows are edges (minimum cut, minimum spanning-tree weight, distance-to-a-property).
- **Database model:** n rows from a domain D (typically {0,1}^d or ℝ^d), with the histogram representation x ∈ ℕ^|X| and the ℓ₁ / Hamming notion of "differs in one record" as the unit of change.
- **Adversary model:** a probabilistic interactive machine with arbitrary auxiliary information and (in the strongest setting) unbounded computation; interactive (query–response transcripts, possibly adaptive) versus non-interactive (publish a sanitized object once) release.
- **Quantities to track:** the amount one record can move a query, the magnitude of added noise as a function of that, and the accuracy of the answer as a function of the database size.

## Code framework

The starting scaffold is a trusted-curator harness around a true query plus a randomized response rule. The data pipeline, the query evaluation, and a generic centered-noise interface are standard.

```python
import numpy as np

# --- common primitives ---

def true_answer(database, f):
    """Curator computes the exact (non-private) query value f(database).
    f maps a database to a real number or a real vector."""
    return f(database)

def neighboring(x, y):
    """Two databases are neighbors iff they differ in a single individual's
    record under the chosen row-level convention. In a row representation this
    is one-row Hamming distance; in a histogram representation a replacement
    can move one count down and another count up."""
    return differs_in_one_record(x, y)         # representation-dependent

def centered_noise_sample(scale, size, rng):
    """Draw centered i.i.d. noise with the given scale. The exact law and
    calibration are open choices."""
    raise NotImplementedError("choose the noise law and scale")


# --- the slot the contribution fills ---

def private_mechanism(database, f, calibration, rng):
    """Answer f on the database by perturbing the true answer with a randomized
    response. What guarantee this is meant to satisfy, and how to calibrate the
    perturbation so it holds, is the open problem."""
    z = true_answer(database, f)
    pass  # TODO
```
