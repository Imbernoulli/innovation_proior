# An average-case depth hierarchy theorem for Boolean circuits

## Problem

For Boolean circuits over {AND, OR, NOT}, depth is a worst-case resource: for each d there is a linear-size depth-d formula that no subexponential depth-(d−1) circuit computes *exactly*. The goal here is the average-case strengthening: an explicit linear-size depth-d formula that no subexponential depth-(d−1) circuit even *approximates* — agreement at most ½ + o(1).

## Main theorem

**Theorem (average-case depth hierarchy).** Let 2 ≤ d ≤ c√(log n)/log log n for an absolute constant c > 0, and let BalancedSipser_d be the explicit n-variable read-once monotone depth-d formula defined below. Then any circuit C of depth ≤ d−1 and size ≤ S = 2^{n^{1/(6(d−1))}} agrees with BalancedSipser_d on at most (½ + n^{−Ω(1/d)})·2ⁿ inputs.

This follows from two incomparable lower bounds, each of which already implies it (a depth-(d−1) circuit is a depth-d circuit with bottom fan-in 1, and is also a depth-d circuit; if its top alternation matches BalancedSipser_d it falls under the first theorem, otherwise the second):

**Theorem (first main bound — small bottom fan-in).** For 2 ≤ d ≤ c√(log n)/log log n, any depth-d circuit C of size ≤ S = 2^{n^{1/(6(d−1))}} and bottom fan-in ≤ (log n)/(10(d−1)) satisfies Pr_X[BalancedSipser_d(X) ≠ C(X)] ≥ ½ − n^{−Ω(1/d)} for X uniform on {0,1}ⁿ.

**Theorem (second main bound — opposite alternation).** For 2 ≤ d ≤ c√(log n)/log log n, any depth-d circuit C of size ≤ S = 2^{n^{1/(6(d−1))}} whose top gate is opposite to BalancedSipser_d's (OR vs AND) satisfies Pr_X[BalancedSipser_d(X) ≠ C(X)] ≥ ½ − n^{−Ω(1/d)}.

**Corollaries.** (1) With probability 1, a random oracle A satisfies Σ_d^{P,A} ⊊ Σ_{d+1}^{P,A} for all d — the polynomial hierarchy is infinite relative to a random oracle. (2) There is a monotone f with Inf(f) = O(log n) that no depth-d(n) circuit of size ≤ exp((log n)^{ω(1)}) (with d(n) = ω(1)) approximates to ½ + o(1) — no approximate converse to the Linial–Mansour–Nisan / Boppana total-influence bound holds.

**Near-optimality.** For monotone f, Bshouty–Tamon forces correlation ≥ ½ + Ω(1/n) with a single variable; and the Hajnal et al. discriminator lemma rules out a d-vs-(d−1) hierarchy at correlation ½ + n^{−ω(1)}. So ½ + n^{−Θ(1)} is best possible.

## The hard function

BalancedSipser_d is the depth-d read-once monotone alternating formula, depth-regular, all n leaves at depth d, bottom-adjacent (depth d−1) gates AND (so the root is OR for even d, AND for odd d), with fan-in sequence w_0, …, w_{d−1}:
- bottom fan-in w_{d−1} := m, and p := 2^{−m} (probability a bottom AND is satisfied by a uniform input);
- middle fan-in w_k := w := ⌊m·2^m/log e⌋ for 1 ≤ k ≤ d−2 (chosen so each layer is self-balancing: pw = m·ln2, so (1−p)^w ≈ 2^{−m} = p);
- top fan-in w_0 := smallest integer with (1−t_1)^{q w_0} ≤ ½ (so the projected top gate has bias ≈ ½).

Here n = ∏_k w_k ≈ (m·2^m/log e)^{d−1}/log e, so m ≈ (log n)/d. The formula is balanced: Pr_X[BalancedSipser_d(X) = 1] = ½ ± Õ(w^{−1/12}).

## Key idea: random projections

A **restriction** ρ ∈ {0,1,∗}^n fixes each variable or keeps it alive (x_i ↦ x_i). A **projection** instead works over variables grouped into blocks 𝒳 = {x_{a,i} : a ∈ A, i ∈ [ℓ]}, with a smaller space 𝒴 = {y_a : a ∈ A}: it fixes x_{a,i} or, when ρ_{a,i} = ∗, maps it to the single new variable y_a. So all alive variables of a block **collide** onto one fresh y_a. Restrictions are the special case ℓ = 1, 𝒴 ≡ 𝒳.

Collisions do three jobs that no restriction can do at once. Let Sipser^{(k)} be BalancedSipser truncated to depth k (each depth-k gate a fresh variable).
- **Target peels cleanly:** AND(y_a,…,y_a) = OR(y_a,…,y_a) = y_a, so `proj Sipser^{(k)} ≡ Sipser^{(k−1)}` — projection walks down the Sipser hierarchy by exactly one layer (Property 2 by construction).
- **Approximator collapses extra:** a gate containing x_{a,i} and x̄_{a,j} becomes y_a ∧ ȳ_a = 0, dead for free (extra simplification in the switching lemma, Property 1).
- **Completion to uniform survives the correlations** (Property 3): the per-stage biases t_k and the per-block keep-alive q_a are solved exactly so the correlated projection still reproduces uniform inputs.

## The projection sequence Ψ

Base parameters: λ := (log w)^{3/2}/w^{5/4} (the "force a block to all-∘" probability, with λ/q³ = w^{1/4}) and q := √p = 2^{−m/2} (keep-alive scale, giving alive sets of size ≈ qw = Θ̃(√w)). Biases solved by completion-to-uniform:
- t_{d−1} := (p − λ)/q, and t_{k−1} := ((1−t_k)^{qw} − λ)/q for k = d−1, …, 2.
- These stay controlled: t_k = q ± q^{1.1} for all k (downward induction giving |t_k q − p| ≤ (2m)^{d−1−k}λ).

**Initial projection R_init** (over the n bottom variables): independently per bottom block a, draw the m-bit block as {1}^m w.p. λ; as {∗_{1/2},1_{1/2}}^m∖{1}^m w.p. q; as {0_{1/2},1_{1/2}}^m∖{1}^m w.p. 1−λ−q.

**Subsequent projection R(τ)** (adaptive — depends on the previous outcome τ): per block a with alive set S_a = τ_a^{−1}(∗), if the gate is determined or S_a is not k-acceptable, re-randomize alive coords {•_{t_k},∘_{1−t_k}}^{S_a}; otherwise draw all-∘ w.p. λ; {∗_{t_k},∘_{1−t_k}}^{S_a}∖{∘} w.p. q_a; {•_{t_k},∘_{1−t_k}}^{S_a}∖{∘} w.p. 1−λ−q_a, with q_a := ((1−t_k)^{|S_a|} − λ)/t_{k−1} (so (1−t_k)^{|S_a|} = λ + q_a t_{k−1}). S is **k-acceptable** if |S| = qw ± w^{β(k,d)}, β(k,d) := 1/3 + (d−k−1)/(12d) ∈ [1/3, 5/12).

Ψ(f) := proj_{ρ^{(2)}} ⋯ proj_{ρ^{(d)}} f with ρ^{(d)} ← R_init, ρ^{(k)} ← R(ρ̂^{(k+1)}) (ρ̂ = lift: the value the depth-(k−1) gates take, ∘/∗/• per Definition of lift). Ψ maps 2^n → 2^{w_0}.

## Proof in four pieces

**Property 3 — Ψ completes to uniform.** For all f, g: Pr_{X~unif}[f(X) ≠ g(X)] = Pr_Y[(Ψf)(Y) ≠ (Ψg)(Y)], where Y is t_1-biased (d even) or (1−t_1)-biased (d odd). Proof: block by block, the choice of t_{d−1} forces λ + q t_{d−1} = p, making both Pr[X_a = 1^m] and Pr[X_a = Z] equal 2^{−m} (uniform); the choice of q_a forces (1−t_k)^{|S_a|} = λ + q_a t_{k−1}, making the stage regenerate the t_k-biased product from the t_{k−1}-biased one; induct from k = 2 up.

**Property 1 — approximator simplifies (projection switching lemma).** For a width-r DNF/CNF F:
- Pr_{ρ←R_init}[proj_ρ F not a depth-s DT] = (O(r·2^r·w^{−1/4}))^s;
- Pr_{ρ←R(τ)}[proj_ρ F not a depth-s DT] = (O(r·e^{r t_k/(1−t_k)}·w^{−1/4}))^s.

Proof (Razborov/Beame/Thapen encoding, generalized to projections): the canonical projection decision tree ProjDT(F↾ρ) queries the merged variables y_a; a bad ρ (deep path) is encoded as θ(ρ) = (ρσ, π′, encode(η), encode(γ)) where σ satisfies the hit terms, the "set non-occurring alive coords to 0" convention makes σ undoable without extra bits, and θ is an injection. The weight ratio per differing block is ζ((ρσ))/ζ(ρ) = Ω(w^{1/4})·((1−t_k)/t_k)^{Δ_a} (here λ/q³ = w^{1/4}); summing over Hamming weight of the new-1's encoding ϑ_4 gives Σ_i C(rs,i)(t_k/(1−t_k))^i = (1+t_k/(1−t_k))^{rs} = (e^{r t_k/(1−t_k)})^s; union over the small ϑ_2, ϑ_3 finishes. Applied bottom-up (with trimming of too-wide bottom gates in the opposite-alternation case): Ψ(C) is a depth-n^{1/4(d−1)} decision tree (or (1/S)-close to a width-n^{1/4(d−1)} CNF/DNF of matching top gate) w.p. 1 − exp(−Ω(n^{1/6(d−1)})).

**Property 2 — target retains structure.** `proj_ρ Sipser^{(k)} ≡ Sipser^{(k−1)}↾ρ̂` (lift Fact), so chaining gives Ψ(BalancedSipser_d) ≡ Sipser_d^{(1)}↾ρ̂^{(2)} — a single OR/AND of fan-in w_0. With **typicality** (each block k-acceptable, ≥ w_{k−2} − w^{4/5} lifted survivors) which (i) holds initially and (ii) bootstraps (typical ⟹ next typical) each w.p. 1 − e^{−Ω̃(w^{1/6})} via Chernoff (β(k,d) widens toward the root precisely to outrun the per-stage error, using d ≤ c log w/log log w), all d−1 lifts are typical w.p. 1 − d·e^{−Ω̃(w^{1/6})}. On that event the surviving top OR has |S(ρ̂)| ≈ q w_0 survivors, so by the choice of w_0, E_Ψ[bias(Sipser_d^{(1)}↾ρ̂^{(2)}, Y)] ≥ ½ − Õ(w^{−1/12}).

**Bottoming out.** Base-case correlation lemma (extending O'Donnell–Wimmer): a width-r CNF F and a restricted top OR_S satisfy Pr_Y[OR_S(Y) ≠ F(Y)] ≥ bias(OR_S, Y) − r t_1 (two cases: every clause has a negated S-literal ⟹ F(0^S)=1; else a monotone clause ⟹ Pr[F=1] ≤ r t_1). Assembling, with r = n^{1/4(d−1)}:

Pr_X[BalancedSipser_d ≠ C] = E_Ψ[Pr_Y[(OR↾ρ̂^{(2)})(Y) ≠ (Ψ C)(Y)]]
  ≥ E_Ψ[bias(OR↾ρ̂^{(2)}, Y)] − r t_1 − Pr_Ψ[Ψ(C) not a depth-r DT]
  ≥ (½ − Õ(w^{−1/12})) − r t_1 − exp(−Ω(n^{1/6(d−1)}))
  ≥ ½ − n^{−Ω(1/d)},

since r t_1 = n^{1/4(d−1)}·Θ̃(w^{−1/2}) = n^{−Ω(1/d)}. The opposite-alternation case is identical up to an extra negligible 1/S term. ∎

## Why each choice is what it is

| Choice | Reason |
|---|---|
| projections, not restrictions | restrictions get only 2 of {P1, P2, P3} (R(p): 1,3; blockwise: 1,2); collisions get all 3 |
| q = √p | alive sets of size ≈ qw = Θ̃(√w): small enough to switch C, large enough that Sipser gates survive d peels |
| λ = (log w)^{3/2}/w^{5/4} | λ/q³ = w^{1/4} = the per-level weight gap in the switching lemma; also closes completion algebra |
| t_{d−1}=(p−λ)/q, q_a=((1−t_k)^{\|S_a\|}−λ)/t_{k−1} | solved (not chosen) by demanding completion to uniform per block |
| adaptive R(τ) | per-block alive sizes \|S_a\| vary; q_a must react to keep completion exact |
| β(k,d)=1/3+(d−k−1)/12d | typicality must bootstrap: widen the deviation window toward the root to outrun accumulated error, but keep β < ½ for Chernoff |
| w = ⌊m2^m/log e⌋ | self-balances each layer (pw = m·ln2) |
| w_0 = least int with (1−t_1)^{q w_0} ≤ ½ | makes the final projected top gate ≈½-biased (Property 2's bias half) |
