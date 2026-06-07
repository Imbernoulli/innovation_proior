# Synthesis — Low-Autocorrelation Binary Sequences & the Merit Factor

## The problem (in-frame)
- Binary sequence A = (a_0,...,a_{n-1}), a_i ∈ {+1,-1}.
- Aperiodic autocorrelation: C_A(u) = sum_{i=0}^{n-1-u} a_i a_{i+u}, u=0..n-1.  C_A(0)=n.
- Merit factor F(A) = n^2 / (2 sum_{u=1}^{n-1} C_A(u)^2).  (Golay 1972 def.)
- Equivalent: L4 norm of P_A(z)=sum a_i z^i: ||P_A||_4^4 = n^2 + 2 sum C_A(u)^2 = n^2(1 + 1/F). (Littlewood)
- Goal: families with large asymptotic F. Random sequence: E[1/F]=(n-1)/n → 1, so F≈1 typical. Want F large.
- Anti-pattern as METHOD: brute-force/stochastic search (LABS problem; spin-glass; "golf-hole"). Search caps ~ F<6 reliably for large n. Principled target = explicit constructions + character-sum analysis.

## Key constructions & their asymptotic F
- Rudin-Shapiro (Golay-Shapiro) pair, recursive append: F → 3 (Littlewood 1968). Computed directly from recurrence, no periodic structure.
- m-sequences (max-length shift register), length 2^m-1: any rotation → F = 3 asymptotically (Jensen-Høholdt 1989). Mean over rotations 3n^2/((n-1)(n+4)) → 3 (Sarwate 1984).
- Legendre sequence, prime length p: x_i = (i|p) Legendre symbol, x_0:=1. ROTATED by fraction r:
  Theorem (Høholdt-Jensen 1988):
    1/lim F(X_r) = 1/6 + 8(r-1/4)^2   for 0≤r≤1/2
                 = 1/6 + 8(r-3/4)^2   for 1/2≤r≤1.
  i.e. 1/F = 2/3 - 4r + 8r^2 on [0,1/2].  Max F = 6 at r=1/4, 3/4. UNROTATED r=0 (Fekete polynomial): 1/F = 1/6+8(1/16) = 2/3 ⇒ F→3/2 (verified numerically). r=1/4 doubles-and-quadruples relative to r=0.
- Jacobi (product of Legendre), modified Jacobi / twin-prime: same g, F→6 (Borwein-Choi 2001; Jensen-Jensen-Høholdt 1991).
- Appending (rotate then append own initial elements), Borwein-Choi-Jedwab 2004: F>6.34 numerically.
- Negaperiodic / periodic constructions (Parker; Jedwab-Katz-Schmidt 2013): rigorous F_a = 6.342061... = largest root of 29x^3-249x^2+417x-27, at T=1.057827 (mid root of 4x^3-30x+27), R=3/4-T/2. Disproves Høholdt-Jensen conj that 6 is optimal. Skew-symmetric versions achieve it (Cor 2.4).

## The methodological leap (what to reconstruct)
1. Search is hopeless as a method; need structure. What structure makes aperiodic autocorrelations collectively small?
2. Key bridge: PERIODIC autocorrelation R_A(u) = C_A(u) + C_A(n-u). For a cyclic difference set sequence (Paley/quadratic-residue, n≡3 mod 4), R_A(u) = -1 for all u≠0 (constant). This is invariant under rotation. So C_A(u)+C_A(n-u) = -1 for every rotation. Tiny periodic correlations ⇒ HOPE that some rotation makes aperiodic ones collectively small — but periodic flat does NOT imply aperiodic flat (the periodic identity only constrains the PAIR sum). Turyn's actual rationale (per Jedwab).
3. The Legendre symbol gives such a difference set (Paley). So Legendre = canonical candidate.
4. Why rotation matters: the aperiodic correlation C_{X_r}(u) is a windowed character sum over an interval; its size depends on WHERE the window sits relative to the quadratic-character structure. Need to actually estimate sum_u C(u)^2 as function of the window offset r. This is the leap from "flat periodic" hope to a real computation.
5. Compute via Gauss sums / character sums. Two routes:
   (a) Original (HJ 1988): C_{X_r}(u) is a partial sum of Legendre symbols; use the multiplicativity (i|p)(i+u|p) and Gauss-sum / Polya-Vinogradov-type estimates to get an asymptotic for sum_u C(u)^2 as a function of the interval [r p, r p + ...]; produces the parabola 1/6 + 8(r-1/4)^2.
   (b) Modern (JKS 2013): work in Fourier domain. X_p(ζ_k)-1 is a quadratic Gauss sum of magnitude p^{1/2}. Define
       L_A(a,b,c) = (1/n^3) sum_k A(ζ_k)A(ζ_{k+a})A(ζ_{k+b})A(ζ_{k+c}).
       Then 1+1/F(A^{r,t}) is an explicit expression in L_A. For Legendre, |L_{X_p}(a,b,c) - I_p(a,b,c)| ≤ 18 p^{-1/2} where I_p is the "ideal" indicator (one of a,b,c =0 and other two equal). Plug into the general formula (Thm 4.1) ⇒ F → g(R,T); at T=1 recover the parabola, max g(1/4,1)=6.
   The L_A→I_p estimate uses: Weil bound on sum_x (x(x+a)(x+b)(x+c) | p), magnitude ≤ 3 p^{1/2} unless the quartic is a square (two double roots ⇒ p-2, quadruple ⇒ p-1).
6. The g function (JKS general):
   1/g(R,T) = 1 - 4T/3 + 4 sum_{m∈N} max(0, 1 - m/T)^2 + sum_{m∈Z} max(0, 1 - |1 + (2R-m)/T|)^2.
   T=1 (no truncation/appending): collapses to 1/g(R,1) = 1/6 + 8(R-1/4)^2 (mod 1/2 periodicity in R).
7. The "single shift u=n" obstruction in appending: appending own initial t·n elements would keep improving F except the shift u=n maps the appended block onto its copy, contributing (tn)^2 to sum C^2; balancing this against the gains gives the optimum t≈0.0578 and F→6.342. (Borwein-Choi-Jedwab intuition; JKS rigor via negaperiodic/periodic product with (++--) / (++-+) masks.)

## Design decisions → why
- Use Legendre symbol (not arbitrary ±1): need a sequence whose PERIODIC autocorrelation is provably constant ⇒ difference set ⇒ quadratic residues (Paley). Algebraic structure ⇒ Gauss sums ⇒ analyzable.
- x_0 := 1 convention (Legendre symbol (0|p)=0 undefined for sequence): need ±1; choose +1. Asymptotically negligible (one element).
- Rotate by r: periodic flatness invariant under rotation, but aperiodic sum-of-squares is NOT; optimize over the free parameter r. r=1/4 is where the windowed-character-sum energy is minimized (parabola minimum).
- n ≡ 3 mod 4 (Paley): gives genuine cyclic difference set, R(u) = -1. For n ≡ 1 mod 4, partial difference set; handle by slight modification.
- Skew-symmetric sieve (Golay): a_{m+i} = (-1)^i a_{m-i}; forces C(u)=0 for all odd u ⇒ halves the work, plausibly no asymptotic loss (Golay's ergodicity heuristic; proven mean unchanged).
- Merit factor not max|C(u)| (M(A)): better theory, energy interpretation, Turyn's intuition that lim sup F easier than lim inf M.

## Numbers to be exact about
- Golay 1972 merit factor def; Littlewood L4 identity n^2(1+1/F).
- Rudin-Shapiro F = 3/(1-(-1/2)^m) → 3.
- HJ parabola 1/6 + 8(r-1/4)^2; max F=6.
- Weil bound 3 p^{1/2}; quartic-square cases p-2 (two double roots), p-1 (quadruple root).
- |L - I| ≤ 18 p^{-1/2} (Lemma 5.1); Galois |L - J| ≤ (n+1)^{3/2}/n^2.
- F_a = 6.342061..., root of 29x^3-249x^2+417x-27; T=1.057827 (4x^3-30x+27); R=3/4 - T/2.
- Galois: F_b = 3.342065..., root 7x^3-33x^2+33x-3; T=1.115749 (x^3-12x+12).

## Code grounding
No single canonical repo (textbook-level). Implement from definitions in Jedwab survey eqs (1),(2),(15): Legendre symbol via pow(i,(p-1)//2,p); build X; rotate; compute C(u); F. Demonstrate periodic F→3, r=1/4 F→6; verify against parabola.

## Sources read this run
- Jedwab survey 2005 (full text). refs/jedwab_survey_2005.txt
- Jedwab-Katz-Schmidt "Advances" 2013 (Thm 2.1, g, Sec 4 LA machinery, Sec 5 Legendre Lemma 5.1 character sum). refs/jedwab_katz_schmidt_advances_2013.txt
- Høholdt-Jensen-Justesen 1985 (Rudin-Shapiro class, F=3, recurrence). refs/hoholdt_jensen_justesen_1985.txt
- ar5iv 1205.0626 (g formula, F_a). WebFetch.
