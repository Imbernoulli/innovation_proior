# Super-polynomial lower bound for multilinear formulas computing the permanent and determinant

## Problem

An arithmetic formula is a binary tree of `+` and `×` gates over input variables and field constants; its size is the number of nodes. A formula is **multilinear** if the polynomial at every gate is multilinear (each variable to power ≤ 1). The permanent and determinant of an n×n matrix are multilinear, and their natural formulas are multilinear. The goal: an unconditional super-polynomial lower bound on the size of multilinear formulas for these polynomials, over any field — a regime where no super-polynomial bound is known for general formulas.

## Key idea

Attach to a multilinear polynomial f over two variable sets Y = {y_1,...,y_m}, Z = {z_1,...,z_m} the **partial-derivative matrix** M_f: rows indexed by multilinear Y-monomials, columns by multilinear Z-monomials, with M_f(p,q) = the coefficient of p·q in f. This is the commutative/multilinear analogue of Nisan's non-commutative coefficient matrix. Its rank behaves well under gates (sub-additive over `+`, multiplicative over `×` when variable sets are disjoint) and is capped by 2^{(smaller side)}.

Naively "small multilinear formula ⇒ low-rank M_f" is **false**: f = ∏_i (y_i + z_i) has a linear-size formula yet M_f is a permutation matrix (full rank 2^m), because the Y/Z split aligns with the factors. The fix is to **randomize the split**. Under a structured random assignment of the matrix variables, (i) every small formula is forced to be *k-weak* — on every "central" (heavy-child) path some gate is *k-unbalanced* (sees ≥ k more of one side than the other), which forces a rank deficit — while (ii) the permanent restricts to ∏_i(y_i+z_i) and the determinant to ∏_i(y_i−z_i), both of full rank 2^m. Deficit vs. full rank ⇒ contradiction.

## Definitions

- **Syntactic multilinear:** at every product gate, the two sons have disjoint variable sets. Any multilinear formula reduces to a syntactic one of the same size and same output (set the repeated variable to 0 in the son that doesn't contain it).
- For a node v over Y∪Z: Y_v, Z_v are the y-/z-variables in φ_v; **b(v) = avg(|Y_v|,|Z_v|)**, **a(v) = min(|Y_v|,|Z_v|)**, **d(v) = b(v) − a(v)**.
- **k-unbalanced:** d(v) ≥ k. **central path** (leaf→v): each parent u and chosen child u' satisfy b(u) ≤ 2·b(u'); one always exists since b(u) ≤ b(u_1)+b(u_2). **k-weak node:** every central path reaching it is k-unbalanced; **k-weak formula:** root is k-weak.

## Rank facts

For M_v := M_{φ_v}, with sons v_1, v_2:
1. rank(M_v) ≤ 2^{a(v)}.
2. (`+` gate) rank(M_v) ≤ rank(M_{v_1}) + rank(M_{v_2}).
3. (`×` gate, syntactic multilinear) rank(M_v) = rank(M_{v_1})·rank(M_{v_2}).

## Lemma A (k-weak ⇒ rank deficit)

If v is k-weak then **rank(M_v) ≤ |φ_v|·2^{b(v) − k/2}**.

*Proof.* k-weak ⇒ b(v) ≥ k. Induct on |φ_v|.
- *Leaf:* rank ≤ 1 ≤ 2^{b(v)−k/2} (since b(v)−k/2 ≥ k/2 ≥ 0).
- *v k-unbalanced:* a(v) = b(v)−d(v) ≤ b(v)−k, so by (1) rank ≤ 2^{b(v)−k} ≤ |φ_v|·2^{b(v)−k/2}.
- *Product, v not k-unbalanced:* b(v) = b(v_1)+b(v_2); WLOG b(v) ≤ 2b(v_1), so the heavy child v_1 is k-weak. Induction: rank(M_{v_1}) ≤ |φ_{v_1}|2^{b(v_1)−k/2}; (1): rank(M_{v_2}) ≤ 2^{a(v_2)} ≤ 2^{b(v_2)}; (3): rank(M_v) ≤ |φ_{v_1}|2^{b(v_1)+b(v_2)−k/2} = |φ_{v_1}|2^{b(v)−k/2} ≤ |φ_v|2^{b(v)−k/2}.
- *Plus, v not k-unbalanced:* b(v) ≤ b(v_1)+b(v_2); WLOG b(v) ≤ 2b(v_1) ⇒ v_1 k-weak.
  - If b(v) ≤ 2b(v_2): v_2 also k-weak; b(v) ≥ b(v_1), b(v) ≥ b(v_2); induction + (2): rank(M_v) ≤ (|φ_{v_1}|+|φ_{v_2}|)2^{b(v)−k/2} ≤ |φ_v|2^{b(v)−k/2}.
  - If b(v) > 2b(v_2): then b(v_2) < b(v)/2 ≤ b(v)−k/2 (as b(v) ≥ k), so by (1) rank(M_{v_2}) ≤ 2^{b(v_2)} ≤ 2^{b(v)−k/2}; v_1 k-weak with b(v) ≥ b(v_1) gives rank(M_{v_1}) ≤ |φ_{v_1}|2^{b(v)−k/2}; (2): rank(M_v) ≤ (|φ_{v_1}|+1)2^{b(v)−k/2} ≤ |φ_v|2^{b(v)−k/2}. ∎

## The structured random assignment A

X = {x_{i,j}}_{i,j∈[n]}, m = ⌈n^{1/3}⌉. Choose disjoint row indices q_1,...,q_m, r_1,...,r_m (all distinct) and disjoint column indices s_1,...,s_m, t_1,...,t_m (all distinct). For each i, on the 2×2 block (rows q_i,r_i; cols s_i,t_i), with prob ½ assign

  [[x_{q_i,s_i}, x_{q_i,t_i}],[x_{r_i,s_i}, x_{r_i,t_i}]] ← [[y_i, z_i],[1, 1]],

with prob ½ assign ← [[y_i, 1],[z_i, 1]]. Fill an arbitrary perfect matching of leftover rows/cols with 1, all else with 0. Each y_i, z_i is used once; φ_A is syntactic multilinear over Y∪Z of the same size.

Permanent of the substituted matrix = ∏_{i=1}^m (y_i + z_i) (both 2×2 layouts have permanent y_i+z_i). Determinant = ±∏_{i=1}^m (y_i − z_i) (both layouts have determinant y_i−z_i).

## Lemma B (small formula ⇒ k-weak whp)

There is a universal constant ε > 0 (e.g. ε = 10^{−6}) such that if a syntactic multilinear formula φ over X has |φ| ≤ n^{ε·log n}, then under A, **Pr[φ_A is k-weak] = 1 − o(1)** with **k = n^{1/32}**.

*Proof.* Let α(v) = |X_v|/n². Generate the special-variable pairs W_i = {A^{-1}(y_i), A^{-1}(z_i)} by: pick w_1^i uniform in X; with prob ½ pick w_2^i uniform in the same row, ½ same column; the row/col-distinctness rejection has prob o(1), so analyze the i.i.d. distribution μ* and transfer (statistical distance o(1)). Step three independently labels each pair's y vs z. **Which paths are central depends only on the W_i (steps 1–2); k-unbalancedness also depends on step 3.**

- **Claim 1 (Chernoff):** whp ∀v: α(v) ≥ n^{−1/8} ⇒ 0.5α(v)·2m ≤ |W_v| ≤ 1.5α(v)·2m; α(v) < n^{−1/8} ⇒ |W_v| ≤ 1.5n^{−1/8}·2m. (Each |W_v| is a sum of 2m indicators of mean α(v); deviation ½, failure < 4e^{−n^{1/12}/2}; union over ≤ n^{ε log n} nodes.)
- **Claim 2:** whp ∀v with 1/8 ≥ α(v) ≥ n^{−1/8}: |W_v^0| ≥ (1/16)α(v)m, where W_v^0 = pairs meeting X_v in exactly one element. (≤ 2α(v)n dense rows/cols ⇒ ≥ half of X_v's variables are "good"; for each i, Pr[w_1^i∈X_v ∧ w_2^i∉X_v] ≥ α(v)/8; Chernoff, failure < 2e^{−n^{1/12}/128}.)

Call W̃ good if both claims hold (prob 1−o(1)). Fix good W̃ and a central path γ (leaf→root). Pick v_1 = first node with α ≥ 100n^{−1/8}, v_{i+1} = first with α ≥ 100α(v_i), stop past 1/8. Centrality + Claim 1 ⇒ α grows by < factor 1000 per step ⇒ l = Ω(log n) nodes, with 1/8 ≥ α(v_i) ≥ 100n^{−1/8} and |W^0_{v_{i+1}}| ≥ 2|W_{v_i}| ≥ 2n^{1/8}.

Let E = "γ not k-unbalanced", E_i = "v_i not k-unbalanced". E ⊆ ∩E_i, so Pr[E|W̃] ≤ ∏_i Pr[E_i | ∩_{i'<i}E_{i'}]. For fixed i, let S = W^0_{v_i} \ W_{v_{i-1}}, T = W_{v_i} \ S; then |S| ≥ |W^0_{v_i}| − |W_{v_{i-1}}| ≥ 2|W_{v_{i-1}}| − |W_{v_{i-1}}| = |W_{v_{i-1}}| ≥ n^{1/8}, and W_{v_{i-1}} ⊆ T. With χ_j = 1 iff A(x_j) ∈ Y, d(v_i) = |∑_{j} χ_j − r/2| = |σ + τ − r/2|, σ = ∑_{x_j∈S}χ_j, τ = ∑_{x_j∈T}χ_j. The S-coins are mutually independent fair coins (distinct pairs) and independent of the T-coins ⇒ σ ~ Binomial(|S|,½) even conditioned on T. Binomial anti-concentration: σ hits no value with prob > O(|S|^{−1/2}) = O(n^{−1/16}). So d(v_i) hits no specific value with prob > O(n^{−1/16}), conditioned on any T-event. Since ∩_{i'<i}E_{i'} is a T-event (it depends on W_{v_{i-1}} ⊆ T) and d(v_i) is an integer:

  Pr[E_i | ∩_{i'<i}E_{i'}] = Pr[d(v_i) < k | T-event] ≤ k·O(n^{−1/16}) = O(n^{1/32}·n^{−1/16}) = O(n^{−1/32}).

Hence Pr[E|W̃] ≤ (O(n^{−1/32}))^{Ω(log n)} = n^{−Ω(log n)}. Union over < n^{ε log n} leaves (paths) ⇒ for good W̃, whp all central paths are k-unbalanced ⇒ root k-weak. Since W̃ is good whp, φ_A is k-weak with prob 1−o(1). ∎

## Theorem (the landing)

**Over any field, any multilinear arithmetic formula for the permanent or the determinant of an n×n matrix has size n^{Ω(log n)}.** (Corollary: any multilinear circuit for them has depth Ω(log² n).)

*Proof.* Take the permanent; suppose a multilinear formula φ has |φ| ≤ n^{ε·log n}. By the reduction, assume φ syntactic multilinear. By Lemma B there is an A with φ_A k-weak, k = n^{1/32}. By Lemma A at the root,

  rank(M_{φ_A}) ≤ |φ_A|·2^{m − k/2} = n^{ε·log n}·2^{m − k/2}.

Since n^{ε·log n} = 2^{ε(log n)²} and ε(log n)² < k/2 = n^{1/32}/2 for large n, this is < 2^m. But φ_A computes the permanent of the substituted matrix = ∏_{i=1}^m (y_i + z_i), whose M_f is a permutation matrix of rank exactly 2^m. So rank(M_{φ_A}) is both < 2^m and = 2^m — contradiction. Hence every multilinear formula for the permanent has size n^{Ω(log n)}. For the determinant, φ_A computes ±∏_{i=1}^m (y_i − z_i), again rank 2^m; same contradiction. ∎

Remark: ∏(y_i+z_i) itself has a linear-size multilinear formula. The contradiction is therefore not that the restricted polynomial is hard, but that a small formula for the permanent becomes k-weak under the random cut, and a k-weak formula cannot output a full-rank polynomial — even an easy one. The full-rank-ness of the permanent/determinant restriction (engineered by the 2×2-block design) is incompatible with the rank deficit a small formula is forced into.
