Let me start from what I actually have to prove and how stuck I am with it. I take a sequence of reals α₁, α₂, α₃, … , roll the line onto a circle of circumference 1, and ask whether the reduced points {αₙ} cover the circle evenly: for every arc a of length |a|, the count n_a of the first n points landing in a should satisfy n_a/n → |a|. That is the definition, and it is a brutal object to work with directly. It quantifies over *all* arcs simultaneously, and an arc indicator is a step function — rigid, discontinuous, nothing I can differentiate or expand. For the one sequence I understand, nξ with ξ irrational, the only proofs I know are interval-chasing: take two equal arcs, show their counts differ by a bounded amount, conclude each gets its fair share. That works because successive points nξ and (n+1)ξ differ by the constant step ξ; the whole argument lives off that. The moment the points are n²ξ, the increments themselves change with n, and the two-interval comparison has nothing to grip. And Hardy and Littlewood's quadratic proof leans on a Cauchy integral built around degree 2 — I see no way to push it to degree q. So I want to stop proving each family by its own trick and find one condition that *is* even distribution, checkable on any sequence.

Let me loosen the definition first, because indicators are the problem. Counting points in arcs is the same as averaging indicator functions: n_a/n = (1/n) Σ_{h≤n} χ_a(αₕ), and the target |a| = ∫₀¹ χ_a. So "n_a/n → |a| for every arc" reads as "(1/n) Σ f(αₕ) → ∫₀¹ f for every interval indicator f." Can I widen "indicator" to a friendlier class? Suppose I knew

  (1/n) Σ_{h=1}^n f(αₕ) → ∫₀¹ f(x) dx

for every bounded 1-periodic Riemann-integrable f. Then in particular for f = χ_a I'd get the counting statement back, so this is at least as strong. And it's also implied by it: any Riemann-integrable f is, by definition of the Riemann integral, squeezed between two step functions f₁ ≤ f ≤ f₂ with ∫f₂ − ∫f₁ < ε; a step function is a finite sum of interval indicators, so the averaging law holds for f₁ and f₂ by the arc statement, and then

  lim (1/n) Σ f(αₕ) ≥ lim (1/n) Σ f₁(αₕ) = ∫f₁ ≥ ∫f − ε,

and symmetrically ≤ ∫f + ε, so it holds for f. The two formulations are equivalent. Good — that buys me room: I'm no longer chained to indicators, I can test against *any* convenient family of functions, as long as the family is rich enough that controlling it controls all Riemann-integrable f by sandwiching.

So which family? I want functions that (a) behave well under the dynamics on the circle, and (b) are dense enough to approximate everything. The dynamics here is "add a real number mod 1," and the functions that diagonalize translation are the exponentials. Write e(x) = e^{2πix}. Then e(k(x+t)) = e(kx)·e(kt): translating x just multiplies by a constant. These are the period-1 functions whose graph is "rigid under rotation." And the theory of Fourier series tells me they're complete: every continuous periodic function is a uniform limit of finite trigonometric polynomials a₀/2 + Σ(a_k cos2πkx + b_k sin2πkx), i.e. of finite combinations of the e(kx). Their integrals are the cleanest possible: ∫₀¹ e(kx) dx is 1 when k = 0 and 0 when k ≠ 0. So e(x) is the genuine analytic invariant of a residue class mod 1 — it's the right test function.

Now run the averaging law against f = e(mx). For m = 0 it's f ≡ 1 and the law is the trivial (1/n)Σ1 = 1 = ∫1. For m ≠ 0 the target is ∫₀¹ e(mx) dx = 0, so the law says (1/n) Σ_{h=1}^n e(mαₕ) → 0. Suppose I *assume* exactly this for every m ≠ 0:

  Σ_{h=1}^n e(mαₕ) = o(n)   for every integer m ≠ 0.

Does it give me back the full averaging law, and hence even distribution? Take any finite trig polynomial f(x) = a₀/2 + Σ_{k=1}^m (a_k cos + b_k sin) — a finite combination of e(jx) with |j| ≤ m. Averaging is linear, so (1/n) Σ f(αₕ) is a finite combination of (1/n) Σ e(jαₕ): the j = 0 term gives a₀/2 = ∫f, every j ≠ 0 term goes to 0 by assumption. So the averaging law holds for every trig polynomial. Now a continuous periodic f: pick a trig polynomial f_ε with |f − f_ε| < ε everywhere (Weierstrass/Fejér). Then f_ε − ε ≤ f ≤ f_ε + ε, and since the averaging law holds for f_ε (hence for f_ε ± ε), squeezing gives it for f up to 2ε, and ε is arbitrary, so for all continuous f. Finally a Riemann-integrable f: its jumps can be replaced by steep continuous ramps, trapping it between two continuous functions f₁ ≤ f ≤ f₂ with integrals as close as I like — squeeze once more. So the averaging law holds for every Riemann-integrable f, in particular for arc indicators, which is even distribution.

That's the whole game changed. I have a criterion: **the points αₙ are uniformly distributed mod 1 if, for every integer m ≠ 0, Σ_{h=1}^n e(mαₕ) = o(n).** A geometric condition on infinitely many arcs has become an analytic condition on a single kind of sum — one I can attack with algebra, with Cauchy–Schwarz, with closed forms. Let me see what it does on the cases.

Linear first, to make sure it's not vacuous. ξ irrational, αₕ = hξ. Fix m ≠ 0 and set η = mξ. Because ξ is irrational and m ≠ 0, η is not an integer. The sum is literally geometric:

  Σ_{h=1}^n e(hη) = Σ_{h=1}^n e(η)^h = e(η) · (e(η)^n − 1)/(e(η) − 1) = [e((n+1)η) − e(η)] / [e(η) − 1].

The numerator has modulus at most 2 (difference of two unit-modulus numbers), so

  |Σ_{h=1}^n e(hη)| ≤ 2 / |e(η) − 1|.

And |e(η) − 1| = |e^{2πiη} − 1| = 2|sin πη| — square it out: |e^{2πiη} − 1|² = (cos2πη − 1)² + sin²2πη = 2 − 2cos2πη = 4sin²πη. So the bound is 1/|sin πη|, a *finite constant independent of n*. That's not just o(n), it's O(1) — the partial sums never grow. Criterion satisfied for every m ≠ 0, so {hξ} is uniformly distributed. And the proof didn't touch the additive geometry; it just summed a geometric series. (If ξ were rational, η could be an integer for suitable m, e(η) = 1, denominator zero, sum = n, not o(n) — and indeed {hξ} then cycles through finitely many points, not even distributed. The criterion sees exactly the right dividing line.)

Several coordinates next, because the same characters live on the torus. A sequence α(n) = (α₁(n),…,α_p(n)) in the p-torus; "evenly distributed" means each region of volume V is hit with frequency V. The characters of the torus are e(m·α) = e(m₁α₁ + … + m_pα_p) for integer vectors m = (m₁,…,m_p), and exactly the same Fourier completeness and sandwiching argument runs. So the criterion is: α(n) is uniformly distributed iff for every nonzero integer vector m,

  Σ_{h=1}^n e(m₁α₁(h) + … + m_pα_p(h)) = o(n).

Apply it to α(n) = (nξ₁,…,nξ_p). For a nonzero integer vector m, m·α(n) = n·(m₁ξ₁+…+m_pξ_p) = nη with η = Σ m_iξ_i. If there is *no* nontrivial integer linear relation among ξ₁,…,ξ_p (no l₁ξ₁+…+l_pξ_p ∈ ℤ with the l_i integers not all zero), then η ∉ ℤ for every nonzero m, and the same geometric series gives |Σ e(hη)| ≤ 1/|sin πη| = O(1). So the orbit is uniformly distributed on the torus. This is stronger than Kronecker's density theorem — density only says the orbit comes close to every point; this says how *often* it visits each region. Kronecker falls out as the weak shadow: even distribution forces density.

Now the real wall: polynomials. αₕ = φ(h) with φ(z) = α_q z^q + … + α_1 z + α_0, some non-constant coefficient irrational, q ≥ 2. The criterion demands Σ_{h} e(mφ(h)) = o(n) for each m ≠ 0; absorbing m into the coefficients, I just need

  σ_n := Σ_{h=0}^n e(φ(h)) = o(n)

for φ with an irrational non-constant coefficient. Try the geometric-series move: e(φ(h)) = e(φ(h)) but e(φ(h+1))/e(φ(h)) = e(φ(h+1) − φ(h)) is *not* a constant ratio — φ(h+1) − φ(h) is itself a polynomial in h of degree q − 1. No geometric series. The continuous version is fine: ∫₀ᵗ e(φ(t))dt, substitute φ(t) = x so it's ∫ e(x) dx/φ′(t), integrate by parts to get [e(x)/φ′]₀ᵗ + ∫ e(φ)·φ″/φ′² dt, and since ∫|φ″/φ′²| converges the whole thing is O(1). But that uses an antiderivative; the discrete sum has none. I'm stuck in exactly the place the continuous trick lives off of.

What can I do to a sum I can't close? I can square its modulus, and watch what that does to the phase. |σ_n|² = σ_n σ̄_n = (Σ_h e(φ(h)))(Σ_k e(−φ(k))) = Σ_{h,k} e(φ(h) − φ(k)). The exponent is now a *difference of values of φ*. Put h = k + r. Then

  φ(k + r) − φ(k) = r · φ(r, k),

where φ(r, k), as a polynomial in k, has degree q − 1 — the top term α_q (k+r)^q − α_q k^q = α_q (q r k^{q-1} + …), so dividing by the common factor r, the polynomial φ(r,k) has leading term q α_q k^{q-1}, and crucially the constant α_0 has cancelled out of the difference. So

  |σ_n|² = Σ_r Σ_k e(r φ(r, k)),

with k ranging 0 ≤ k ≤ n, 0 ≤ k + r ≤ n. The inner sum over k is over a phase of degree q − 1 in k. Differencing dropped the degree by exactly one. That's the lever — if I can keep doing it, I'll grind the degree down to 1, where the geometric series works.

But I can't just square again: |σ_n|² is already real, and Σ_r Σ_k isn't of the form whose square I want. I need to isolate the inner sum over k and reduce *its* degree, with the outer Σ_r along for the ride. Cauchy–Schwarz does this: writing the inner sum as a_r,

  |σ_n|² = Σ_r a_r = Σ_r 1·a_r,  so  |σ_n|⁴ = |Σ_r a_r|² ≤ (Σ_r 1)(Σ_r |a_r|²) = n₁ Σ_r |a_r|²,

where n₁ is the number of r's in range. Now |a_r|² = |Σ_k e(rφ(r,k))|² = Σ_{k,l} e(rφ(r,k) − rφ(r,l)) = Σ_{k,l} e(r[φ(r,k) − φ(r,l)]), and I difference the *inner* polynomial again: put k = l + s, φ(r, l+s) − φ(r, l) = s·φ(r, s, l), where φ(r,s,l) has degree q − 2 in l with leading term q(q−1) α_q l^{q-2}, and now both α_0 and α_1 have dropped out. So

  |σ_n|⁴ ≤ n₁ Σ_{r,s} Σ_l e(r s φ(r, s, l)),

inner phase degree q − 2. Each round costs me a Cauchy–Schwarz (an extra factor like n₁, n₂, … counting the new outer variable) and doubles the exponent on |σ_n| — 2, 4, 8, … — but lowers the inner degree by one. So after q − 1 rounds the inner polynomial in the running variable is *linear*.

Let me carry the bookkeeping to the end of the q − 1 steps, naming the difference variables r₁, r₂, …, r_{q-1} and letting h be whatever index still runs inside. At the bottom the inner phase in h is

  φ(r₁, …, r_{q-1}, h) = q! · α_q · h + (β₀ + β₁ r₁ + β₂ r₂ + … + β_{q-1} r_{q-1}),

linear in h. The leading coefficient is q! α_q: each differencing step multiplied the leading coefficient by the descending factor q, then q−1, …, then 2, so the product is q·(q−1)···2 = q!. And *because α_q is irrational, q! α_q is irrational*, so the inner sum over h is exactly the linear case I already cracked. Set ξ = q! α_q, R = r₁ r₂ ··· r_{q-1}, and let ϱ = R·(β₀ + β₁r₁ + … + β_{q-1}r_{q-1}) collect the part that doesn't depend on h. With Q = 2^{q-1} the accumulated exponent and N the accumulated product of the Cauchy–Schwarz counting factors, N = (n₁)^{2^{q-2}}(n₂)^{2^{q-3}}···n_{q-3}n_{q-2} (take N = 1 when q = 2), I get

  |σ_n|^Q ≤ N · Σ_{r′} | e(ϱ) · Σ_h e(R ξ h) |,    (★)

where r′ = (r₁,…,r_{q-1}) ranges over the region |r′| = |r₁| + … + |r_{q-1}| ≤ n (the constraints 0 ≤ everything ≤ n pile up into this octahedron), and for each r′ the index h runs over a block of n + 1 − |r′| consecutive integers.

The inner sum Σ_h e(Rξh) is geometric — same closed form as before — bounded by 1/|sin π(Rξ)|. And |e(ϱ)| = 1. So (★) becomes |σ_n|^Q ≤ N Σ_{r′} 1/|sin π(Rξ)|. I want to divide by something like n^q and send n → ∞. Let me size N: each n_i is just the count of one new outer variable, of order n, so replacing every n_i by n turns N into a power of n whose exponent is 1·2^{q-3} + 2·2^{q-4} + … + (q−2)·2^0 = Q − q, so N ~ κ n^{Q − q} with κ depending only on q.

The frequency in the inner sum is Rξ = r₁r₂···r_{q-1}·q!α_q — it depends *multiplicatively* on the outer variables r′. For most r′ that's fine: Rξ mod 1 sits away from 0, |sin π(Rξ)| is bounded below, the term is O(1). But for some r′ the product Rξ lands very close to an integer, |sin π(Rξ)| → 0, and 1/|sin| blows up. If too many r′ are "bad," the bound is worthless. I need to know *how many* r′ make Rξ mod 1 fall near 0.

So the genuinely hard estimate isn't the differencing — that's bookkeeping — it's controlling the count of bad r′, i.e. understanding the distribution of the multilinear quantity r₁r₂···r_{q-1}·ξ as the r_i range over the octahedron. Let me isolate that as a lemma and prove it on its own.

Claim (the lemma): for ξ irrational, summing the character of the *product* over the octahedron is negligible —

  Σ_{r over |r|≤n} e(r₁ r₂ ··· r_q ξ) = o(n_q),

where n_q = #{integer points with |r| ≤ n} ~ (2n)^q/q! (the volume of the octahedron is (2n)^q/q!, and the lattice-point count is asymptotic to it). And the version I actually need: the number of r′ in the (q−1)-dimensional octahedron |r′| ≤ n with R ξ = r₁···r_{q-1} ξ falling within ε of an integer is < 3ε · n_{q-1} for large n.

Prove the lemma by induction on q. For the sum, peel off the last variable r_q and write Σ_r = Σ_{r′} Σ_{r_q}, where r′ = (r₁,…,r_{q-1}) is the projected point in the octahedron |r′| ≤ n and r_q runs over the integers with |r_q| ≤ n − |r′|. The inner sum over r_q is geometric in r_q: with R = r₁···r_{q-1},

  |Σ_{r_q} e(r_q · Rξ)| ≤ 1/|sin π(Rξ)|,

and also, trivially, ≤ 2n + 1 since there are at most 2n + 1 terms. Now split the outer r′ by whether Rξ mod 1 is ε-close to an integer. For the "good" r′ (Rξ at distance ≥ ε from ℤ), |sin π(Rξ)| ≥ |sin πε| ≈ ε, so the inner sum is ≤ 1/|sin πε|, and there are at most n_{q-1} of them. For the "bad" r′, use the crude 2n + 1; but the bad r′ are exactly those for which Rξ mod 1 ∈ (−ε, ε), and *by the induction hypothesis applied to the (q−1)-fold product*, their count is < 3ε · n_{q-1} (this is the same statement one dimension down — the distribution of r₁···r_{q-1}ξ near 0). Combining,

  |Σ_r e(r₁···r_q ξ)| ≤ n_{q-1}{ 3ε(2n + 1) + 1/|sin πε| }.

Divide by n_q. Since n_{q-1}(2n+1)/n_q → q (because n_q ~ (2n)^q/q! and n_{q-1} ~ (2n)^{q-1}/(q-1)!, the ratio is (2n+1)·q/(2n) → q), for large n

  (1/n_q)|Σ_r e(r₁···r_q ξ)| < ε(3q + 1),

and ε is arbitrary, so the sum is o(n_q). That's the lemma; and reading the bad-count step on its own gives the corollary I needed: #{r′ : |r′| ≤ n and Rξ mod 1 ∈ (−ε,ε)} < 3ε·n_{q-1}. The base case q = 1 is just the linear statement: r₁ξ equidistributes, so the proportion of r₁ with r₁ξ ε-near 0 tends to 2ε < 3ε. Induction closes.

Now back to (★) with the count in hand. Split Σ_{r′} the same way: good r′ (Rξ at distance ≥ ε from ℤ) contribute ≤ 1/|sin πε| each, at most n_{q-1} of them; bad r′ (count < 3ε·n_{q-1}) get the crude inner bound |Σ_h e(Rξh)| ≤ n + 1. So

  |σ_n|^Q ≤ N · n_{q-1} { 3ε(n + 1) + 1/|sin πε| }.

Plug N ~ κ n^{Q−q} and n_{q-1} ~ (2n)^{q-1}/(q-1)!. The dominant term is 3ε(n+1): N·n_{q-1}·3ε(n+1) ~ κ n^{Q−q} · (2^{q-1}/(q-1)!) n^{q-1} · 3ε n = 3ε κ (2^{q-1}/(q-1)!) n^Q. The other term, with the fixed 1/|sin πε|, carries one fewer power of n and is lower order. So dividing by n^Q,

  |σ_n / n|^Q ≤ 3ε ( κ · 2^{q-1}/(q-1)! + 1 )

for n large enough. The right side is a fixed multiple of ε, ε is arbitrary, therefore σ_n/n → 0, i.e. σ_n = o(n). The criterion is met, and φ(0), φ(1), φ(2), … is uniformly distributed mod 1 — *whenever the leading coefficient α_q is irrational*. Degree q arbitrary, one mechanism: square-and-difference to drop the degree, Cauchy–Schwarz to iterate, the lemma to control the multiplicative frequencies, the linear geometric series at the bottom.

One case left, and I almost missed it: what if the *leading* coefficient is rational, and the irrationality sits lower down — α_q, …, α_{l+1} all rational but α_l irrational? Then at the bottom of the differencing the inner frequency is q! α_q, which is *rational*, and the whole machine collapses (the geometric sum can blow up because q!α_q R can be an integer). I have to manufacture an irrational leading coefficient. Let G be a common denominator of the rational coefficients α_q, …, α_{l+1}, so that Gᵏαₖ is an integer for those top k. Split the index by residue classes mod G: write the index as Gh + r, r = 0,1,…,G−1, and

  Σ_{index} e(φ(index)) = Σ_{r=0}^{G-1} Σ_{h} e(φ(Gh + r)).

Look at φ(Gh + r) as a polynomial in h. Its coefficient of hᵏ is αₖ Gᵏ (times binomial junk plus lower contributions). For k > l those αₖ Gᵏ are integers — the rational top coefficients, cleared by the denominator G — so they contribute integers to the phase and vanish mod 1: φ(Gh + r) ≡ ψ_r(h) (mod 1) for a polynomial ψ_r of degree l whose leading coefficient is α_l Gˡ, which is *irrational* (α_l irrational, Gˡ a nonzero integer). So each of the G inner sums is over a degree-l polynomial with irrational leading coefficient — exactly the case I just proved — hence each is o(n), and the finite sum of G of them is o(n). Done: φ is uniformly distributed mod 1 whenever *any* non-constant coefficient is irrational.

Let me also note the strengthening that drops out for free: the differencing only ever used that the leading frequency q!α_q (or α_l Gˡ) was bounded away from integers via 1/|sin|, and the count bound 3ε·n_{q-1}, neither of which cared about the values of the other coefficients. So with the highest-index irrational coefficient held fixed, the o(n) bound is *uniform* over the remaining coefficients — the convergence rate depends only on that one irrational number, not on the rest of the polynomial.

And one more corollary by partial summation, which I'll want for the analytic application. Suppose a₀ ≥ a₁ ≥ … are positive and decreasing with divergent sum. Abel summation gives Σ_{h=0}^n a_h e(φ(h)) = a_n σ_n + Σ_{h=0}^{n-1} σ_h (a_h − a_{h+1}). Given ε, pick H with |σ_h| < ε h for h ≥ H (possible since σ_h = o(h)); then |Σ a_h e(φ(h))| ≤ C_ε + ε n a_n + ε Σ_{h} h(a_h − a_{h+1}), and for decreasing a_h the telescoping Σ h(a_h − a_{h+1}) = Σ a_h − n a_n ≤ Σ a_h, so |Σ a_h e(φ(h))| ≤ C_ε + ε Σ a_h = o(Σ a_h). In particular with a_h = 1/h, Σ_{h=1}^n e(φ(h))/h = o(log n) — the kind of weighted exponential sum that controls ζ on the line Re s = 1, which was the original analytic motive for caring about Σ e(n²ξ) at all.

Stepping back to see the chain whole: the rigid arc-counting definition is equivalent, by sandwiching through Riemann integrability, to averaging arbitrary functions; the right functions to average are the characters e(mx), because they diagonalize translation and are Fourier-complete, which turns even distribution into the single checkable condition Σ e(mαₕ) = o(n) for all m ≠ 0; on linear and torus orbits that condition is a bounded geometric series, settling those cases and surpassing Kronecker's density; on polynomials the sum has no closed form, so I square-and-difference to drop the degree, iterate with Cauchy–Schwarz down to a linear inner phase whose frequency q!α_q is irrational, and tame the multiplicatively-coupled frequencies with a lattice-point counting lemma proved by the same induction; a residue-class splitting reduces an irrational lower coefficient to the irrational-leading case; and partial summation extends it to weighted sums. The clean statements and full proofs follow.
