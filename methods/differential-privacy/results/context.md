# Context: privacy-preserving analysis of statistical databases

## Research question

A trusted curator holds a database of sensitive records — one row per person — and wants to release useful aggregate statistics (counts, histograms, means, covariances, graph properties) without compromising the privacy of any individual whose data is in it. The question is foundational and definitional before it is algorithmic: **what should "preserving privacy" even mean here, precisely enough to prove a mechanism satisfies it?** A satisfactory answer has to survive an adversary with arbitrary auxiliary information (other databases, newspapers, prior medical studies, gossip) and unbounded ingenuity, and it has to be robust to the obvious failure that has plagued every prior ad-hoc defense: the attacker combines several "harmless" released answers, or one released answer plus outside knowledge, to pin down a single person's record. The solution must also leave room for utility — releasing nothing, or pure noise, is trivially private and useless — so the definition has to coexist with the release of genuine statistical information.

A second, sharper requirement: the guarantee should hold *regardless of what the adversary already knows*. Most classical measures of disclosure are average-case or assume a particular prior over the data; an attacker who happens to know "Terry Gross is two inches shorter than the average Lithuanian woman" can turn an innocuous released average into an exact personal fact. A real definition cannot wave this away — it must hold in the worst case over side information.

## Background

**The statistical-disclosure-control tradition.** Protecting respondents while publishing tabulations is an old problem in official statistics and, since the late 1970s, in computer science. A detailed catalogue of the techniques explored through 1989 is the survey of Adam and Wortmann (1989), which organizes the field into three families: **query restriction** (refuse queries on too-small sets, limit query overlap, audit a query log and disallow compromising queries), **input/data perturbation** (randomly modify the underlying records, e.g. data swapping or adding a fixed random perturbation to each entry, then answer from the modified data), and **output perturbation** (compute the true answer, then report a noisy or rounded version). The classical antecedent closest in spirit to noise-based release is Denning's work. None of these came with a rigorous adversary model or a definition of success that an attacker had to defeat; each was a heuristic, broken in turn.

**Why the heuristics break — combination attacks.** Query restriction fails the moment two large allowed queries differ in a single record: subtract the two sums and the lone record's value falls out. Restricting query overlap or auditing the log helps only for very few queries, and the auditor's *refusals* themselves leak — a refusal, combined with the answers to permitted queries, can finish a compromise; deciding when to refuse is itself NP-hard. Removing names and identifiers does not anonymize: a handful of "innocuous" attributes (gender, approximate age, ZIP, marital status) re-identifies most people, and a rare attribute identifies them outright — the linkage attacks demonstrated by Sweeney, which rekindled computer scientists' interest in the problem.

**The reconstruction barrier (Dinur and Nissim, 2003).** This is the load-bearing diagnostic finding. Model a database as bits d₁,…,dₙ ∈ {0,1}; a statistical query is a subset q ⊆ [n] answered by Σᵢ∈q dᵢ plus perturbation. Dinur and Nissim give a **polynomial-time reconstruction algorithm**: from answers to enough (random) subset-sum queries whose perturbation is only o(√n), an adversary recovers a candidate database within Hamming distance εn of the real one — i.e. all but a tiny fraction of the private bits. They name the failure they rule out **blatant non-privacy**: a bounded adversary recovering a 1−ε fraction of the entries. The consequence is a hard quantitative law: to have any hope of privacy against unrestricted queries one must add perturbation of magnitude **Ω(√n)** (linear in n against a computationally unbounded adversary). The technique is essentially list-decoding / a Goldreich–Levin-style recovery from noisy linear measurements. So noise is *necessary*, its *magnitude* is what matters, and too little noise is catastrophic — but Dinur–Nissim say "here is what privacy is *not*" without saying what it *is*.

**The Dalenius desideratum and its collapse.** Dalenius (1977) articulated the goal everyone implicitly wanted: *access to a statistical database should not enable one to learn anything about an individual that could not be learned without access.* This is the database analogue of the cryptographic notion of **semantic security** that Goldwasser and Micali (1982) defined for encryption five years later — nothing learnable about a plaintext from the ciphertext that could not be learned without it. The disanalogy is fatal. Semantic security is achievable because the ciphertext is *useless* to anyone but the key-holder; the auxiliary-information generator "has no idea" what ciphertext the eavesdropper will see. But a statistical database is *designed* to convey information, and there is no decryption key separating the legitimate user from the adversary — they are one and the same. Whoever knows the data knows roughly what the user will learn, and can plant auxiliary information that combines with the release to breach privacy. The Terry-Gross-height example makes it concrete: release the average height of Lithuanian women, hand the adversary "Terry Gross is two inches shorter than that average," and Terry Gross's exact height is disclosed — **whether or not Terry Gross is in the database at all.** The Dalenius goal, formalized as a relaxed semantic security, is therefore *impossible* whenever the release has any utility. An absolute "you learn nothing about me" promise cannot be kept, so the promise must be reformulated.

**Privacy by randomized process (Warner, 1965).** A much older idea points the way toward *how* a private mechanism behaves. Warner's randomized response collects statistics on stigmatized behavior: each respondent flips a coin and, depending on the outcome, either answers truthfully or answers a fixed random way, so that any individual "Yes" carries plausible deniability — it would have occurred with non-trivial probability regardless of the truth — while the population proportion remains an unbiased, recoverable estimate because the noise process is known. Privacy here is a property of the *response process*: the distribution of an individual's reported answer is only mildly shifted by their true bit. This is the seed of "privacy as a property of the mechanism's output distribution," and it foreshadows that randomization is not optional — any non-trivial deterministic mechanism that ever distinguishes two databases distinguishes two that differ in a single row, and an adversary who knows the database is one of those two learns the differing row.

**The cryptographic lens.** A definition that must hold against arbitrary side information naturally borrows the worst-case, adversary-explicit style of cryptography: state precisely what the adversary may know and may do, and bound a worst-case quantity rather than an average. The cryptographic measures themselves (statistical/total-variation distance; computational indistinguishability) are starting points, but with a twist forced by the utility requirement — a simple hybrid argument shows useful release *requires* non-negligible leakage, so the standard "negligible advantage" bar cannot be imported unchanged.

## Baselines

- **Query restriction / auditing.** Allow only queries over large sets, cap query overlap, or audit the query log and refuse compromising queries. *Gap:* two permitted large-set sums that differ in one record reveal that record by subtraction; refusals themselves leak; the auditing decision is NP-hard. Robust only for a handful of queries.

- **k-anonymity (Sweeney) and de-identification.** Generalize/suppress quasi-identifiers so each released record is indistinguishable from at least k−1 others; strip names and SSNs. *Gap:* a syntactic property of the released table, not a semantic guarantee about what an adversary can infer; vulnerable to linkage with outside datasets and to homogeneity within a group. Does not model auxiliary information at all.

- **Input perturbation (Agrawal–Srikant and the data-swapping/fixed-perturbation line).** Randomly modify records, answer from the modified data; privacy "measured" by the magnitude of noise added to a value, or by estimator variance. *Gap:* the noise-magnitude measure ignores what an adversary who knows the data distribution can infer (Evfimievsky et al.); large variance does not imply privacy (a high-variance estimator d̃ᵢ = dᵢ + E·e with E a large even number leaks dᵢ exactly by parity); the original input distribution can itself be reconstructed.

- **Output perturbation (Denning; varying-perturbation and rounding schemes).** Add noise (or round) the true answer; increase variance on repeated queries. *Gap:* no definition the noise is provably sufficient for; the amount of noise was tied to the query dimension or to ad-hoc averages, so high-dimensional releases (histograms, contingency tables, covariances) demanded noise proportional to the dimension, destroying utility — and even then there was no proof against a combination/reconstruction attack. Dinur–Nissim shows the whole family collapses below Ω(√n) noise.

- **Semantic-security-style "learn nothing about an individual" (the Dalenius goal).** The aspirational baseline definition. *Gap:* provably impossible against auxiliary information when the release has utility, and the breach can hit individuals not even in the database — so it cannot be the definition.

- **Statistical / computational indistinguishability (imported from cryptography).** Bound total-variation distance or computational advantage between the views induced by two databases. *Gap:* total-variation distance is an average-case measure — it can be tiny (e.g. 1/n) while a particular transcript reveals a specific individual; and the cryptographic "negligible leakage" standard is incompatible with statistical utility.

## Evaluation settings

The natural yardsticks are about *what a curator must answer and how robustly*:
- **Query classes:** subset-sum / counting queries ("how many records satisfy predicate P?"), their weighted (linear) and fractional forms; **histograms** and **contingency tables** (partition the universe into d bins, count each — high-dimensional output); **means and covariance matrices** of per-record feature vectors; and **graph/holistic functionals** over a database whose rows are edges (minimum cut, minimum spanning-tree weight, distance-to-a-property).
- **Database model:** n rows from a domain D (typically {0,1}^d or ℝ^d), with the histogram representation x ∈ ℕ^|X| and the ℓ₁ / Hamming notion of "differs in one record" as the unit of change.
- **Adversary model:** a probabilistic interactive machine with arbitrary auxiliary information and (in the strongest setting) unbounded computation; interactive (query–response transcripts, possibly adaptive) versus non-interactive (publish a sanitized object once) release.
- **Quantities to track:** a worst-case per-outcome leakage parameter (to be defined), the ℓ₁ amount one record can move a query, the magnitude of added noise as a function of that, and the accuracy of the answer as a function of the leakage parameter and the database size.

## Code framework

The starting scaffold is a trusted-curator harness around a true query plus an unspecified randomized response rule. The data pipeline, the query evaluation, and a generic centered-noise interface are standard; the privacy logic is the empty slot.

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
    calibration are still open choices."""
    raise NotImplementedError("choose the noise law and scale")


# --- the slot the contribution fills ---

def query_property_of_the_mechanism(M, x, y, outcomes):
    """The to-be-defined worst-case quantity comparing the output distribution
    of mechanism M on neighboring databases x, y across all outcomes.
    This is the object the privacy definition will pin down."""
    pass  # TODO: define privacy as a property of M's output distribution on neighbors

def noise_scale_from_query(f):
    """How much noise must be added is, somehow, a property of f alone.
    What property of f, and what scale, is exactly what must be derived."""
    pass  # TODO: identify the inherent quantity of f that determines the noise

def private_mechanism(database, f, calibration, rng):
    """Answer f on the database under the to-be-defined privacy guarantee at
    a calibrated level, by perturbing the true answer with a randomized response."""
    z = true_answer(database, f)
    pass  # TODO: add noise at the scale derived above so the guarantee holds

def compose(mechanisms, database, calibration_list, rng):
    """Run a sequence of private mechanisms on the same database and account
    for the total privacy cost. How the per-query guarantees aggregate is to
    be derived."""
    pass  # TODO: derive how the guarantee degrades under repeated release
```
