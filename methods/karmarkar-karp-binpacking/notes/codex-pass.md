# §2.6 Independent reviewer pass

Codex companion (gpt-5.5 --effort xhigh, then retry with default model) FAILED:
"You've hit your usage limit ... try again at Jun 7th, 2026 1:55 AM."

Fallback per §2.6: careful manual re-derivation of every load-bearing equation
against the primary (refs/karmarkar-karp-1982-primary.txt). Verified:
- Lemma 1  OPT(I) <= 2*SIZE(I)+1                               (primary l.152)
- Lemma 2  SIZE <= LIN <= OPT <= LIN + (m+1)/2; LIN>=SIZE via 1.x>=(sA)x>=sb
           residual min(m,2SIZE'+1)=SIZE'+min(m-SIZE',SIZE'+1)<=SIZE'+(m+1)/2
                                                               (primary l.75-79,124-127)
- Lemma 3  reinsert small (size<=g/2) <= max(A,(1+g)OPT+1)    (primary l.225,243)
          -> CORRECTED draft (had (1+2g)/threshold g) to match paper's g/2,(1+g)
- Lemma 4  linear grouping OPT(J)<=OPT(I)<=OPT(J)+k            (primary l.218)
- Lemma 5  geometric grouping m(J)<=(2/k)SIZE+ceil(log2 1/a),
           OPT loss k*ceil(log2 1/a)                          (primary l.243-279)
- dual (II) max u.b s.t. uA<=1; separation = knapsack max v.u s.t. v.s<=1
           feasibility iff knapsack opt<=1                     (primary l.560-605)
- price rounding u_bar=(t/n)floor(n u/t); DP F(0)=0,
           F(k)=min_i[F(k-u_bar_i)+s_i], feasible iff F(k)>1 for k>1 (l.530-543)
- ellipsoid iters M=4m^2 ceil(ln(n/t))                         (primary l.598)
- constraint elimination to m constraints, x=B^-1 b           (primary l.618-697)
- Algorithm 2 / Theorem 4: k=2, g=1/SIZE -> OPT + O(log^2 OPT);
           iterations t <= ln SIZE / ln k + 1                 (primary l.395,444-454)
- T(m,n)=O(m^8 log m log^2 n + m^4 n log m log n)              (primary l.742-744)

Also: ran all three code copies (code/, reasoning.md, answer.md). Found and fixed
a capacity-feasibility bug: size-grid weight used round(); replaced with
math.ceil(s*G) so the knapsack oracle never returns a configuration whose true
size exceeds 1. After fix, all bins feasible + pieces conserved across 6 seeds.
