# Antecedents & attribution of Algorithm R / reservoir sampling

Sources retrieved this run:
- Vitter 1985, "Random Sampling with a Reservoir", ACM TOMS 11(1):37-57 (refs/vitter1985.txt, refs/vitter1985.pdf).
  - p.39: "Algorithm R (which is a reservoir algorithm due to Alan Waterman)".
  - Sec 2, Definition 1: a reservoir algorithm puts the first n records in a reservoir; the
    invariant is that after each record is processed a true random sample of size n can be
    extracted. Theorem 1 (Sec 3): EVERY one-pass algorithm for this problem is a reservoir
    algorithm; the (t+1)-st record must be chosen with prob n/(t+1).
  - Eq (2.1): avg reservoir entries = n(1 + H_N - H_n) = n(1 + ln(N/n)); this is also the
    lower bound on time (Sec 3). Algorithm R is O(N); Algorithms X/Y/Z generate the skip
    variate S(n,t) faster; Algorithm Z is O(n(1+log(N/n))) = optimum.
  - Eq (3.1): F(s) = Prob(S<=s) = 1 - (t+1-n)^{(s+1) falling}/(t+s+1)^{(s+1) falling}
    (telescoping product of (1 - n/(t+m)) over the s skipped records). Confirms the skip
    derivation in reasoning.md.

- Knuth, TAOCP Vol 2 (Seminumerical Algorithms), Sec 3.4.2, "Algorithm R". (Antecedent.)
  - Algorithm R is attributed by Knuth to Alan G. Waterman, who sent it to Knuth ~1975 as an
    improvement over the inferior 1st-edition Algorithm 3.4.2R (dating from 1962). It appeared
    in the 2nd edition, 1981. (markkm.com/blog/reservoir-sampling, web-retrieved.)
  - Fan, Muller & Rezucha (1962) described a related but distinct reservoir method earlier;
    McLeod & Bellhouse (1983) rediscovered Algorithm R independently. Vitter (1985) cites
    Waterman.

- Li 1994, "Reservoir-sampling algorithms of time complexity O(n(1+log(N/n)))", ACM TOMS
  20(4). (Algorithm L — the skip/threshold form used for the final clean code.) Pseudocode
  verified verbatim against Wikipedia "Reservoir sampling":
    W := exp(log(random())/k)            # W = U^{1/k}
    while i <= n:
        i := i + floor(log(random())/log(1-W)) + 1
        if i <= n: R[randomInteger(1,k)] := S[i]; W := W * exp(log(random())/k)

- Efraimidis & Spirakis 2006 weighted reservoir (A-Res, A-ExpJ): key k_i = u_i^{1/w_i},
  keep the k largest keys; single-draw selection prob = w_i / sum_j w_j (refs/efraimidis_enc.txt).

## Self-account search result (HARD GATE item 2)
No dedicated Vitter first-person retrospective / "how I discovered Algorithm Z" essay,
award lecture, or interview was found this run (web search returned only the primary paper,
reprints, and third-party explainers). GAP: no author self-account exists / could be located.
The discovery trace is therefore reconstructed from the primary source (Vitter 1985, whose
Sec 2-3 themselves narrate the motivating reasoning: R's per-record waste -> the skip
variate S(n,t) -> Algorithm Z) plus the Knuth/Waterman antecedent and the Li/Efraimidis
follow-on forms.
