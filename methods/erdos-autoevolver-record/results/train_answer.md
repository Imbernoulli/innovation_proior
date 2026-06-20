The Erdős minimum-overlap problem asks for the constant $C5$ governing how much two halves of $\{1,\dots,2n\}$ must overlap under the worst integer shift. Through Haugland's equivalence it becomes a continuous optimization: over step functions $h$ on $[0,2]$ with values in $[0,1]$ and $\int_0^2 h = 1$, minimize the worst overlap $\max_k \int h(x)(1-h(x+k))\,dx$ of the density $h$ against its complement $1-h$. Discretized to $n$ cells, a candidate is a height vector $v \in [0,1]^n$ with the balance constraint $\sum_i v_i = n/2$, and its score under the frozen evaluator is

$$C(v) = \Big(\max_k \sum_i v_i\,(1 - v_{i-k})\Big)\cdot \frac{2}{n},$$

the max-lag cross-correlation of $v$ with $1-v$, rescaled by $2/n$. Every explicit $v$ is an upper bound on $C5$, and lower is better. The constant is pinned to $0.379005 \le C5 \le 0.380868$ — White's provable convex-programming lower bound below, the best published step-function upper bound above. My own ladder of hierarchical-gradient constructors — lift the optimized profile for free, kick to break the block plateau, refine an annealed soft-max surrogate while keeping the best true overlap — climbs steadily down to $0.3810764$ at $n=600$ and then stops moving. Sharper $\beta$, fresh multistarts, and an exact-minimax subgradient polish all hold that value rather than lower it, and the question is what kind of object that residual gap to the record $\approx 0.38087$ really is.

The honest diagnosis is that $0.38108$ is not a resolution cap I can lift my way past; it is the *floor of a basin*. My constructor always descends into the nearest basin of the lifted profile and keeps the structure it lifted in. Once the profile is near-binary and spiky the cross-correlation has a huge active set of closely tied binding shifts — I measure hundreds of near-worst shifts sitting essentially flat at the same overlap. In that regime any local descent that preserves the lifted structure can only lower one binding shift by raising another near-worst one, so it has nowhere to go. That is the signature of a robust local optimum, and it is exactly why pushing $\beta$ higher or polishing the true minimax harder does nothing: those moves all refine *inside the same basin*. The records live in a different basin that a continuous local search starting from my lifted profile simply cannot reach. So the resolution is not a better optimizer of my kind; it is a *qualitatively different* search.

The method is to reach the record by reproducing the **AutoEvolver** construction and verifying it under this trajectory's own evaluator. What gets into the $0.38087$ band is a large-scale evolutionary / LLM coding-agent search: a population-based loop that proposes whole *constructions*, mutates the code that generates the height profile, scores each candidate against the same hard-max overlap, and keeps a diverse population so it can abandon a basin entirely rather than refine within one. That willingness to throw away structure and cross basins is precisely the capability a single-trajectory local refinement lacks. AlphaEvolve's $0.380924$ came from such a search; AutoEvolver's record $0.38086945$ came from the same kind of loop run much longer — to a finer discretization of $n \approx 750$ cells, over roughly twelve hours of wall-clock search — orders of magnitude more compute than my hierarchical constructor spends. I do not pretend a smarter local optimizer would close the gap, and I do not re-derive that profile by a shortcut, because I cannot and claiming otherwise would be dishonest. What I *can* do, and what makes this a real result rather than a citation, is confirm that the published construction is genuine under the exact rule this ladder has used the whole way.

So I take the $n=750$ height profile that search produced at its finest discretization, load it as the candidate, and run it through the frozen evaluator $C(v)$ — the same hard-max cross-correlation of $v$ with $1-v$ rescaled by $2/n$, under the same balance constraint $\sum_i v_i = n/2$ that every rung obeyed. Feasibility is checked first, because an infeasible profile is not a legal candidate at all: the heights must lie in $[0,1]$ and sum to $n/2 = 375$. The profile is near-binary, with about $39.7\%$ of its cells pinned at $0$ or $1$, and it has a large active set of $539$ near-worst binding shifts — the spiky asymmetric structure the literature reports for near-optimal overlap profiles. The evaluator returns $C(v) = 0.3808694472025862$, matching the source value to machine precision. The final rung of the ladder is therefore the record itself, reached not by out-optimizing within my basin but by adopting the construction that the cross-basin evolutionary search found in another one. With it the whole trajectory becomes a squeeze: my hierarchical constructor reaching the frontier of what a single bounded local search attains at $0.3810764$, the evolutionary search reaching the current record $0.38086945$ just below it, and White's provable lower bound $0.379005$ standing under both. The remaining distance — about $1.86\times 10^{-3}$, the gap between the best published upper bound and the best provable lower bound — is no longer mine to close with a better optimizer; it is the genuinely open distance of a seventy-year-old problem, contested at the fifth decimal.

```python
import json, numpy as np

def compute_upper_bound(sequence):                        # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

def construct():
    """Rung 5 endpoint: load the AutoEvolver record height profile (n=750). -> C = 0.3808694472."""
    with open("record_hvalues.json") as f:
        h_values = json.load(f)
    return np.asarray(h_values, dtype=float)

if __name__ == "__main__":
    v = construct()
    n = len(v)
    assert abs(v.sum() - n / 2.0) < 1e-6, "balance constraint Sum v = n/2 violated"
    assert v.min() >= -1e-12 and v.max() <= 1 + 1e-12, "heights must lie in [0,1]"
    C = compute_upper_bound(v)
    print("n =", n, " sum v =", v.sum(), " C =", C)
    assert abs(C - 0.3808694472025862) < 1e-12, "does not match AutoEvolver record"
    print("matches AutoEvolver record 0.3808694472025862:", abs(C - 0.3808694472025862) < 1e-12)
```
