# Context: thresholds, expectation thresholds, and the gap between them

## Research question

Fix a finite ground set $X$ with $|X|=n$. For $p\in[0,1]$ let $\mu_p$ be the product measure on the power set $2^X$ that includes each element independently with probability $p$, so $\mu_p(A)=p^{|A|}(1-p)^{n-|A|}$, and write $X_p$ for a random set with this distribution. A family $\mathcal F\subseteq 2^X$ is **increasing** (a monotone property) if $B\supseteq A\in\mathcal F\Rightarrow B\in\mathcal F$. For nontrivial $\mathcal F$ (not $\emptyset$, not $2^X$) the function $p\mapsto\mu_p(\mathcal F)$ is continuous and strictly increasing, so there is a unique **threshold** $p_c(\mathcal F)$ with $\mu_{p_c}(\mathcal F)=1/2$. As $p$ crosses $p_c$, a random set abruptly goes from almost never having the property to almost always having it.

The central problem: thresholds are notoriously hard to pin down, and historically each interesting property has required its own ad hoc argument. The driving question is whether there is a single, computable quantity that predicts $p_c(\mathcal F)$ — up to a small, universal factor — for **every** increasing property simultaneously, replacing case-by-case threshold hunting with one general principle.

There is an obvious candidate for a lower bound, coming from the first moment. Say $\mathcal G\subseteq 2^X$ is a **cover** of $\mathcal F$ if every member of $\mathcal F$ contains some member of $\mathcal G$, i.e. $\mathcal F\subseteq\langle\mathcal G\rangle:=\bigcup_{S\in\mathcal G}\{T:T\supseteq S\}$. Call $\mathcal F$ **$p$-small** if it admits a cover $\mathcal G$ that is "cheap,"
$$\sum_{S\in\mathcal G}p^{|S|}\le \tfrac12 ,$$
and define the **expectation threshold** $q(\mathcal F)$ to be the largest $p$ for which $\mathcal F$ is $p$-small. The name is exact: $\sum_{S\in\mathcal G}p^{|S|}=\mathbb E\big[\,|\{S\in\mathcal G: S\subseteq X_p\}|\,\big]$ is the expected number of cover-sets that a $p$-random set contains. If that expectation is $\le 1/2$ then by the union bound $\mu_p(\mathcal F)\le\mu_p(\langle\mathcal G\rangle)\le\sum_{S}p^{|S|}\le1/2$, so $q(\mathcal F)\le p_c(\mathcal F)$ is automatic. The expectation threshold is the most naive estimate of the threshold and is usually the easiest thing to compute.

So the precise question is: **how far above its expectation threshold can a property's true threshold be?** A bound of the form $p_c(\mathcal F)\le K\,q(\mathcal F)\log\ell(\mathcal F)$, with $K$ universal and $\ell(\mathcal F)=\max\big(2,\text{ size of the largest minimal member of }\mathcal F\big)$, would say the naive estimate is always within a logarithmic factor of the truth — and would instantly reproduce many hard, property-specific threshold results. A solution has to achieve this *for every increasing family at once*, with no structure assumed beyond monotonicity.

## Background

**Thresholds and why the log factor is unavoidable.** The study of thresholds began with Erdős and Rényi's work on random graphs. The original Erdős–Rényi notion is coarse — $p^*(n)$ is *a* threshold for $\mathcal F_n$ if $\mu_p(\mathcal F_n)\to0$ for $p\ll p^*$ and $\to1$ for $p\gg p^*$ — and Bollobás and Thomason showed every increasing family has such a threshold, with $p_c(\mathcal F_n)$ serving as one. Two canonical examples calibrate the gap between $p_c$ and $q$. For the property "$G_{n,p}$ contains a perfect matching," and for "$G_{n,p}$ contains a Hamilton cycle," the expectation threshold is on the order of $1/n$ (it is governed by expected counts of the relevant local structures), while the true threshold is $\log n/n$ (proved for matchings by Erdős–Rényi, for Hamiltonicity by Pósa and Korshunov). The factor $\log n$ separating them is real and comes from a coupon-collector obstruction: having a perfect matching or a Hamilton cycle forces minimum degree $\ge1$ (resp. $\ge2$), and isolated/low-degree vertices persist until $p\approx\log n/n$. So no bound better than a single $\log$ factor is possible in general; the question is only whether a single $\log$ factor *always* suffices.

**The "expectation threshold" conjecture.** Kahn and Kalai (2007) conjectured precisely that $p_c(\mathcal F)\le K\,q(\mathcal F)\log n$ for a universal $K$ (later understood to hold with the smaller $\ell(\mathcal F)$ in place of $n$). They considered it so strong that they wrote it "would probably be more sensible to conjecture that it is *not* true," yet found no counterexample. The conjecture, if true, would subsume seminal results that each cost a long, problem-specific argument — for instance the threshold for perfect matchings in random $r$-uniform hypergraphs (Johansson–Kahn–Vu) and for the appearance of a given bounded-degree spanning tree (Montgomery) — and would explain *why* those thresholds sit where they do, rather than just computing them.

**Sunflowers and "robust" hitting.** A parallel development concerns sunflowers. A collection $S_1,\dots,S_r$ is an $r$-sunflower if all pairwise intersections equal the common core $K=\bigcap_i S_i$; the Erdős–Rado lemma forces a sunflower once a $w$-uniform family is large enough. Alweiss, Lovett, Wu and Zhang (2021), building on a robust/approximate notion of sunflower studied by Rossman and by Li–Lovett–Zhang, proved sharply improved bounds. The mechanism behind their main lemma is what matters here: a family that is sufficiently **spread** — meaning no small set is contained in an outsized fraction of its members, $|\mathcal H\cap\langle S\rangle|\le\kappa^{-|S|}|\mathcal H|$ for a "spreadness" $\kappa$ — is, with high probability, **hit** by a modest random set (the random set contains a whole member). Their proof is an encoding/specification argument: to bound the number of "bad" configurations in which a random portion $W$ fails to make progress toward containing a member, one specifies the configuration in a few steps (first a set $Z$ built from $W$ and the relevant edge, then a small "core," then the remainder), each step contributing a counting factor controlled by the spread hypothesis.

**Diagnostic facts about covers and spread.** Two structural facts are worth isolating, since they shape every attempt. First, $p$-smallness is genuinely a $\{0,1\}$ (integer) condition on the cover: a set is either in $\mathcal G$ or not. Second, the spread condition is a statement about a *measure* (uniform measure on a family, or a probability measure on $2^X$): $\nu$ is $q$-spread if $\nu(\langle S\rangle)\le q^{|S|}$ for all $S$, and a $\kappa$-spread family is exactly one whose uniform measure is $\kappa^{-1}$-spread. Spreadness is the precise hypothesis under which the sunflower-style encoding argument runs, because it is what bounds the number of members through a fixed small core.

## Baselines

**Property-by-property threshold determination.** The state of the art for a long time was: to locate $p_c(\mathcal F)$ for a particular $\mathcal F$, invent an argument tailored to $\mathcal F$. Pósa rotations for Hamiltonicity, absorbers and second-moment refinements for matchings, the intricate Johansson–Kahn–Vu argument for hypergraph matchings. Core idea: exploit the specific combinatorial structure.

**The first-moment / expectation-threshold lower bound.** Core idea: a cheap cover certifies that a $p$-random set is unlikely to have the property, via the union bound $\mu_p(\mathcal F)\le\sum_{S\in\mathcal G}p^{|S|}$. Math: $q(\mathcal F)\le p_c(\mathcal F)$, always.

**The fractional relaxation (Frankston–Kahn–Narayanan–Park, 2021).** Core idea: relax the cover from a set system to a nonnegative *weighting*. Say $\mathcal F$ is **weakly $p$-small** if there is $\lambda:2^X\to\mathbb R_{\ge0}$ with $\sum_{S\subseteq F}\lambda_S\ge1$ for all $F\in\mathcal F$ and $\sum_S\lambda_S p^{|S|}\le1/2$; the **fractional expectation threshold** $q_f(\mathcal F)$ is the largest such $p$. Restricting $\lambda$ to $\{0,1\}$ recovers $p$-smallness, so $q(\mathcal F)\le q_f(\mathcal F)\le p_c(\mathcal F)$. Math/algorithm: because $q_f$ is defined by a linear program, LP duality converts "$\mathcal F$ is not weakly $p$-small" into a concrete object — a spread probability measure supported on $\mathcal F$ — which serves as the starting point for an ALWZ-style encoding argument (refined by separating typical from atypical configurations). With this, FKNP proved $p_c(\mathcal F)\le K\,q_f(\mathcal F)\log\ell(\mathcal F)$.

**The "equate $q$ and $q_f$" program (Talagrand).** Core idea: since the fractional version is settled, it would suffice to show the integer and fractional expectation thresholds are always within a constant factor, $q(\mathcal F)\ge q_f(\mathcal F)/L$; this would make the integer conjecture equivalent to the proved fractional one. Math: it asks that weakly $p$-small implies $(p/L)$-small — a rounding statement turning fractional cover-weights into an integer cover. Talagrand singled out two "test cases," each of which required a nontrivial dedicated argument to settle (DeMarco–Kahn; Frankston–Kahn–Park).

## Evaluation settings

This is a pure existence/quantitative-bound question, so the "yardstick" is the set of benchmark instances against which any general threshold bound is measured for tightness and for whether it recovers known results:

- **Calibration instances where the gap is exactly $\log$:** perfect matchings and Hamilton cycles in $G_{n,p}$ ($q\sim1/n$, $p_c\sim\log n/n$). A general bound must not lose more than this single $\log$ factor.
- **Hard property-specific thresholds to be recovered:** perfect matchings in random $r$-uniform hypergraphs (Shamir's problem), and containment of a given bounded-degree spanning tree in $G_{n,p}$. A general theorem is judged by whether these fall out as easy corollaries.
- **The parameter regime:** increasing families on a finite $X$ of size $n$, with members bounded by $\ell=\ell(\mathcal F)$; the relevant asymptotics are $\ell\to\infty$, and the target relationship is $p_c$ versus $q\log\ell$, with a universal constant and an exceptional probability that should tend to $0$ as $\ell\to\infty$.
- **Metric of success:** a universal constant $K$ such that $p_c(\mathcal F)\le K\,q(\mathcal F)\log\ell(\mathcal F)$ for every nontrivial increasing $\mathcal F$ (equivalently, a high-probability hitting statement for families that are not $p$-small).

## Code framework

This is a theorem to be proved, not an algorithm to be coded; the field-appropriate "scaffold" is the proof harness — the standard objects and the generic argument shape that already exist, with the one creative step left as an empty slot.

**Standing objects (already available).**
- Ground set $X$, $|X|=n$; product measure $\mu_p$ and the random set $X_p$; fixed-size random subsets $X_m\sim\binom{X}{m}$, related to $\mu_p$ by a concentration bound on $|X_p|$.
- An increasing family $\mathcal F$ and its set of minimal elements $\mathcal H$ (a hypergraph), with $\langle\mathcal H\rangle=\mathcal F$; $\mathcal H$ is $\ell$-bounded.
- The cover cost functional $\mathrm{cost}(\mathcal G)=\sum_{S\in\mathcal G}p^{|S|}$ and the notion "$\mathcal H$ is/ is not $p$-small."
- Standard tools: the union bound / first moment, Markov's inequality, binomial-coefficient ratio estimates, and an additive-Chernoff concentration bound for $|X_p|$.

**Generic argument harness (pre-method shape).**
```
# Goal: if H is an ell-bounded hypergraph that is NOT p-small, then a random
# set of size ~ (p log ell) n lands in <H> with probability 1 - o_{ell->inf}(1).

def reduce_to_hypergraph(F):
    H = minimal_elements(F)          # ell(F)-bounded; <H> = F
    # convert mu_p sampling to fixed-size X_m sampling via concentration on |X_p|
    return H                          # prove the hypergraph statement for H

# ---- the one creative step lives here ----
def argument(H, W):
    """Given the not-p-small hypergraph H and a random set W ~ binom(X, w),
    relate W to membership in <H>. The object(s) to define from (H, W),
    and how they connect a random set to the cover cost, are to be designed."""
    pass                             # TODO: the construction this proof will define

def run_proof(H):
    # sample a random set of size ~ (p log ell) n and apply `argument`;
    # turn the bare hypothesis "H is NOT p-small" (no cheap cover exists)
    # into "the random set lands in <H> with high probability".
    pass                             # TODO: assemble the argument into the conclusion
```
The final proof fills `argument` (and any lemma it relies on) with an explicit construction and `run_proof` with the global conclusion; the reduction and the standing objects are exactly the generic pieces above.
