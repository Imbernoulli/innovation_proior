Let me start from the thing that nags me. Bloch's spin-wave calculation says the magnetization of a Heisenberg ferromagnet drops as M(0) − const·T^{3/2} in three dimensions, and that number comes from counting thermally excited magnons. The dispersion is quadratic and gapless, E(k) ≈ Dk² at small k, because the exchange is rotationally symmetric — tilting all the spins rigidly costs nothing, so the uniform mode is a zero-energy mode and the cost only turns on at order k². Fine. Now I put the same spins on a plane instead of in a bulk crystal and redo the count. The density of magnon states in d dimensions, from E ∼ k² and the volume element k^{d-1}dk, goes like N(E) ∼ E^{(d−2)/2}. The reduction of the magnetization is

  ΔM(T) ∼ ∫₀^∞ N(E) / (e^{E/k_B T} − 1) dE.

In d=3 that's ∫ E^{1/2} n_B(E) dE, converges, gives T^{3/2}, all good. In d=2 the density of states is *constant*, N(E) ∼ E⁰, and near the bottom of the band e^{E/k_BT} − 1 ≈ E/k_BT, so

  ΔM(T) ∼ k_B T ∫₀ dE/E,

which diverges logarithmically at the lower limit. In d=1 it's worse, ΔM ∼ ∫ E^{-1/2}·(k_BT/E) dE, power-law divergent. So at any finite temperature the magnetization reduction is infinite. The long-wavelength magnons are infinitely easy to excite when the mode is gapless and the dimension is low, and they eat all the order.

That *looks* like a proof that two-dimensional Heisenberg magnets can't order. But it isn't, and the reason it isn't is exactly the reason I can't trust it: the whole calculation is built on the premise that the spins are aligned. I expanded around the fully-ordered state to define the magnons in the first place. When the "correction" ΔM diverges, all I've actually shown is that my expansion is inconsistent — the perturbation theory is eating its own assumption. A diverging correction to an assumed ordered state is a red flag, not a theorem. Maybe the true state isn't the one I expanded around; maybe the magnons interact and tame the divergence; I have no control. I need something that never assumes order in the first place and still pins the magnetization down.

So let me be careful about what "spontaneous magnetization" even means. The Hamiltonian H = −Σ_{ij} J_{ij} **S**_i·**S**_j commutes with every component of the total spin — it has the full continuous SO(3) symmetry. That means in any finite system the thermal average of, say, the total S^z is exactly zero: the trace of ρ S^z, with ρ ∝ e^{−βH}, vanishes because for every configuration there's a rotated one with the opposite S^z and the same energy. No preferred direction. So ⟨S^z⟩ = 0 identically, in every dimension, ordered or not. "Spontaneous magnetization" can't be ⟨S^z⟩ in the symmetric ensemble. It has to be the *quasi-average*: I add a small field b that breaks the symmetry, take the system to infinite size so it can choose a direction, and only then send b → 0. If the limit

  m = lim_{b→0} lim_{N→∞} m(T, b)

is nonzero, the symmetry is spontaneously broken. The order of limits matters — that's the whole content of "spontaneous." So the object I actually have to control is m(T, b), the magnetization in a small field, and what I want to show is that as I turn the field off, in d ≤ 2, m is squeezed to zero for every T > 0.

That reframes the problem beautifully. I don't need to compute m. I need an *upper bound* on m(T, b) — some function f(b, T) with m ≤ f, where f → 0 as b → 0. If I can produce such an f, the spontaneous magnetization is trapped at zero and I'm done, and I never had to assume anything about the state. The question is where to get a rigorous upper bound on an order parameter, valid in any state.

Here's where I remember the tool that does exactly this without presupposing order — the inequality Bogoliubov wrote down. It relates two essentially arbitrary operators and the Hamiltonian, and it's an exact statement about thermal equilibrium, true regardless of whether there's symmetry breaking. Let me reconstruct it, because I want to be sure of every factor, and because the proof tells me what freedom I have in choosing the operators.

Take the energy eigenbasis H|n⟩ = E_n|n⟩, with thermal weights w_n = e^{−βE_n}/Z, Z = tr e^{−βH}. I'll define an inner product on operators by

  B(X, Y) = Σ_{n,m, E_n ≠ E_m} w_n ⟨n|X⁺|m⟩⟨m|Y|n⟩ / (E_n − E_m),

restricting the sum to E_n ≠ E_m. Why this thing? Because (w_n − w_m)/(E_n − E_m) > 0 whenever E_n ≠ E_m — both numerator and denominator flip sign together since w decreases with E — so this is a positive-semidefinite Hermitian form. Let me check it's positive on the diagonal: B(X, X) = Σ_{E_n≠E_m} w_n |⟨m|X|n⟩|²/(E_n − E_m). Symmetrizing the (n,m)↔(m,n) terms, the coefficient of |⟨m|X|n⟩|² becomes (w_n − w_m)/(E_n − E_m) ≥ 0. Good, it's a genuine semi-inner-product, so Schwarz holds:

  |B(X, Y)|² ≤ B(X, X) · B(Y, Y).

Now I get to choose. I have two operators in mind, call them A and C, and I'm going to choose the second slot of the inner product cleverly: set Y = [C⁺, H]. Then

  B(X, [C⁺,H]) = Σ_{E_n≠E_m} w_n ⟨n|X⁺|m⟩⟨m|[C⁺,H]|n⟩/(E_n − E_m).

The commutator with H is the trick: ⟨m|[C⁺,H]|n⟩ = ⟨m|C⁺H − HC⁺|n⟩ = (E_n − E_m)⟨m|C⁺|n⟩. The (E_n − E_m) cancels the denominator exactly, and now the restriction E_n ≠ E_m can be dropped because the summand vanishes when E_n = E_m anyway. So

  B(X, [C⁺,H]) = Σ_{n,m} w_n ⟨n|X⁺|m⟩⟨m|C⁺|n⟩ = ⟨X⁺ C⁺⟩ … wait, let me be careful with which weight sits where. Reinstating both halves of the original symmetric sum I get the difference ⟨C⁺X⁺⟩ − ⟨X⁺C⁺⟩ = ⟨[C⁺, X⁺]⟩. So with X = A,

  B(A, [C⁺,H]) = ⟨[C⁺, A⁺]⟩ = ⟨[A, C]⟩⁺ in magnitude, i.e. |B(A,[C⁺,H])| = |⟨[A,C]⟩|

up to taking the adjoint inside the bracket. Good — the *left* side of Schwarz, with this choice, is the commutator of A and C. That's the slot where I'll plant the order parameter.

Now the two norms. First B([C⁺,H], [C⁺,H]). Put X = Y = [C⁺,H] and run the same cancellation of (E_n − E_m): this collapses to ⟨[C⁺,[H,C]]⟩ — a double commutator. Let me get its sign right. Using ⟨m|[C⁺,H]|n⟩ = (E_n−E_m)⟨m|C⁺|n⟩ twice, B([C⁺,H],[C⁺,H]) = Σ w_n ⟨n|[H,C]|m⟩⟨m|C⁺|n⟩·(E_n−E_m)/(E_n−E_m)… it telescopes to ⟨[[C,H], C⁺]⟩, which I can write as ⟨[C⁺,[H,C]]⟩. It's a norm, so it's ≥ 0; good, a positive real number.

Second, B(A, A) = Σ_{E_n≠E_m} w_n |⟨m|A|n⟩|²/(E_n − E_m), and I want to bound it *above* by something simple. The dangerous factor is (w_n − w_m)/(E_n − E_m) (after symmetrizing), which I must cap. Write it as

  (w_n − w_m)/(E_n − E_m) = [(w_n + w_m)/(E_n − E_m)] · (w_n − w_m)/(w_n + w_m).

The second factor is (e^{−βE_n} − e^{−βE_m})/(e^{−βE_n} + e^{−βE_m}) = tanh(β(E_m − E_n)/2). And |tanh(x)| < |x|, so |tanh(β(E_m−E_n)/2)| < β|E_m − E_n|/2. Therefore

  (w_n − w_m)/(E_n − E_m) < (w_n + w_m) · β/2.

That kills the energy denominator entirely. Feeding it back,

  B(A,A) < (β/2) Σ_{n,m} (w_n + w_m) |⟨m|A|n⟩|² = (β/2) ⟨A A⁺ + A⁺ A⟩ = (β/2) ⟨{A, A⁺}⟩.

The anticommutator. Beautiful — the energies have washed out and I'm left with a thermal expectation of {A,A⁺}, which is the kind of thing a sum rule can bound.

Assemble Schwarz, |B(A,[C⁺,H])|² ≤ B(A,A)·B([C⁺,H],[C⁺,H]):

  |⟨[A, C]⟩|²  ≤  (β/2) ⟨{A, A⁺}⟩ · ⟨[[C, H], C⁺]⟩.

There it is — the Bogoliubov inequality, no assumption about order anywhere in sight:

  ½ β ⟨{A, A⁺}⟩ · ⟨[[C, H], C⁺]⟩  ≥  |⟨[A, C]⟩|².

Everything now rides on choosing A and C. I want the left commutator ⟨[A,C]⟩ to *be* the order parameter, the anticommutator ⟨{A,A⁺}⟩ to be boundable above by the length of the spins, and the double commutator ⟨[[C,H],C⁺]⟩ to be boundable above by something that carries the field b and the wavevector k. If I can do all three, I get

  m² ≤ (something with b and k) / (something), summed or integrated over k,

and I read off whether m → 0.

Let me set up the operators in momentum space, because the divergence I'm chasing lives at small k and momentum is where the spin algebra is cleanest. Define the Fourier components S^±(**k**) = Σ_i e^{i**k**·**R**_i} S_i^±, and let me allow an ordering wavevector **K**: **K**=0 will mean ferromagnetic order, and **K** at a zone boundary will mean staggered (antiferromagnetic) order. The order parameter I'm after is the (staggered) magnetization

  m = (1/N) |⟨Σ_i e^{−i**K**·**R**_i} S_i^z⟩| = (1/N) |⟨S^z(−**K**)⟩|.

And to even have a nonzero m, I add the symmetry-breaking field: H → H − b Σ_i e^{−i**K**·**R**_i} S_i^z. For K=0 that's a uniform field along z; for K≠0 it's a staggered field, exactly the field conjugate to the order I'm probing. Good — one parameter b, one phase **K**, and the same machinery covers ferro and antiferro.

Now, what should A and C be? I need [A,C] to produce S^z(−**K**). Look at the spin algebra: [S^+(**k**₁), S^-(**k**₂)] = 2ℏ S^z(**k**₁+**k**₂). So if I take one operator to be an S^+ at wavevector k and the other to be an S^- at wavevector −k−**K**, their commutator gives S^z(−**K**), which is the order parameter. Let me set

  C = S^+(**k**),   A = S^-(−**k** − **K**).

Then [A, C] = [S^-(−k−K), S^+(k)] = −[S^+(k), S^-(−k−K)] = −2ℏ S^z(−**K**), and

  ⟨[A, C]⟩ = −2ℏ ⟨S^z(−**K**)⟩ = −2ℏ N m (up to the sign/phase, which I'll carry as magnitude).

So |⟨[A,C]⟩|² = 4ℏ² N² m². The order parameter is now sitting on the right-hand side of the inequality, squared, exactly where I wanted it.

Next, the anticommutator ⟨{A, A⁺}⟩. With A = S^-(−k−K), A⁺ = S^+(−k−K), so {A,A⁺} = {S^-(q), S^+(q)} with q = −k−K. Here's where I want a bound independent of k. Summing over all **k** is the same as summing over all q, and

  Σ_q ⟨{S^-(q), S^+(q)}⟩ = Σ_q Σ_{ij} e^{iq·(R_i − R_j)} ⟨S_i^- S_j^+ + S_j^+ S_i^-⟩.

Do the q-sum first: Σ_q e^{iq·(R_i−R_j)} = N δ_{ij}. So the cross terms die and I'm left with the on-site piece,

  Σ_q ⟨{S^-(q), S^+(q)}⟩ = N Σ_i ⟨S_i^- S_i^+ + S_i^+ S_i^-⟩ = N Σ_i ⟨2(S_i^x)² + 2(S_i^y)²⟩,

since S^∓S^± = (S^x)² + (S^y)² ∓ ℏS^z and the ℏS^z pieces cancel in the symmetric combination. But (S_i^x)² + (S_i^y)² = **S**_i² − (S_i^z)² ≤ ℏ² S(S+1), the total spin length squared. So each site contributes at most 2ℏ²S(S+1), and

  Σ_**k** ⟨{A, A⁺}⟩ ≤ 2 ℏ² S(S+1) N².

That's the spin-length sum rule. It's an *upper* bound, exactly the direction I need, and it's k-independent. This is the whole reason S^± were the right operators: their anticommutator is capped by the finite length of a quantum spin. A particle-number system wouldn't hand me this for free; it's special to the spin algebra.

Now the hard one, the double commutator ⟨[[C, H], C⁺]⟩ with C = S^+(**k**) and the full H including the field. Split H = H_ex + H_b, the exchange part and the field part, and compute [[S^+(k), H], S^-(−k)] for each. Let me work in real space and Fourier at the end. The inner commutator [S_m^+, H_ex]: from H_ex = −Σ_{ij} J_{ij} **S**_i·**S**_j = −Σ J_{ij}[½(S_i^+S_j^- + S_i^-S_j^+) + S_i^z S_j^z], and using [S_m^+, S_j^-] = 2ℏδ_{mj}S_m^z, [S_m^+, S_j^z] = −ℏδ_{mj}S_m^+, I get terms that, when I take the *second* commutator with S_p^- and then the thermal average, organize into a structure proportional to (1 − cos(**k**·(**R**_m − **R**_p))). Let me see why that combination appears: the exchange double commutator brings down two phase factors e^{i**k**·**R**_m} and e^{−i**k**·**R**_p} from the two S^±(**k**) operators, and the J_{mp}-weighted sum of products of spin operators is real and symmetric under m↔p, so only the symmetric, real part 1 − cos(**k**·(**R**_m − **R**_p)) of the phase survives after averaging. Concretely the exchange piece comes out as

  ⟨[[S^+(k),H_ex],S^-(−k)]⟩ = Σ_{mp} J_{mp} (1 − cos(**k**·(**R**_m − **R**_p))) · ⟨2 S_p^z S_m^z + S_m^+ S_p^-⟩ + (… real positive structure …).

I don't need its exact value — I only need an *upper bound*. Two moves. First, the spin-operator expectations here are each bounded in magnitude by the spin length: |⟨2S_p^z S_m^z + S_m^+S_p^-⟩| ≤ const · ℏ² S(S+1). Second, and this is the move that injects the physics, bound the cosine:

  1 − cos(**k**·(**R**_m − **R**_p)) ≤ ½ (**k**·(**R**_m − **R**_p))² ≤ ½ k² (R_m − R_p)².

This is where the k² is born — the same k² that made the magnon gapless and quadratic, now appearing as a rigorous *upper* bound on the energy cost of the double commutator, with no spin-wave assumption at all. Pulling it together, the exchange contribution to the double commutator is bounded by

  ½ k² · Σ_{mp} |J_{mp}| (R_m − R_p)² · const·ℏ²S(S+1) ≡ ℏ² S(S+1) Q̃ k²,   Q̃ ≡ const · Σ_j |J_{ij}|(R_i − R_j)².

And right here I see the *short-range* condition surface, not as an assumption I imposed but as the condition for this bound to be finite: Q̃ is the second moment of the exchange, Σ_j |J_{ij}|(R_i−R_j)². If the couplings are short-ranged it converges and Q̃ is a finite number, and the bound really is ∝ k². If instead J fell off slowly — say a dipolar 1/r³ tail in two dimensions — then Σ |J| R² would diverge, Q̃ would be infinite, and this whole bound would be vacuous. Long-range interactions change the small-k dispersion away from k² (a 1/r³ tail gives E ∼ k, hence N(E) ∼ E and a convergent magnon integral), and order can survive. So "short-range" is precisely the statement that Q̃ < ∞, i.e. that the double commutator really vanishes like k² at small k. I'll carry it as a hypothesis with its teeth showing.

Now the field part, [[S^+(k), H_b], S^-(−k)] with H_b = −b Σ_m e^{−i**K**·**R**_m} S_m^z. The inner commutator [S_m^+, S_m^z] = −ℏ S_m^+, then commuting with S^-(−k) gives a 2ℏ S^z, so this term reduces to something proportional to ⟨S^z⟩ weighted by the field, i.e. proportional to the magnetization itself:

  ⟨[[S^+(k), H_b], S^-(−k)]⟩ = 2ℏ² b Σ_m e^{−i**K**·**R**_m}⟨S_m^z⟩ = 2ℏ² N b m   (up to sign/phase).

So the field contributes a piece proportional to b·m — independent of k — while the exchange contributes the k²-piece. Bounding the magnitude and combining,

  ⟨[[C, H], C⁺]⟩ ≤ 2 N ℏ² ( |b m| + S(S+1) Q̃ k² ).

Stare at this. The denominator-to-be of my bound is small in two ways: it's small when the field b is small (the |b m| term), and it's small when k is small (the k² term). And both smallnesses are about to fight against an integral over k that piles up modes near k = 0. That is the entire mechanism. Let me also double-check the structure is right for the antiferromagnet: nothing in the double commutator referenced **K** except through the field phase e^{−i**K**·**R**_m}, which got absorbed into the definition of m. So the bound is **K**-independent — it will treat ferro and staggered order identically, which is exactly what I want, since the question "can it order" shouldn't care which kind of order.

Now drop the three pieces into Bogoliubov. For each **k**,

  ½ β ⟨{A,A⁺}⟩ · 2Nℏ²(|bm| + S(S+1)Q̃k²)  ≥  4ℏ²N² m².

I want to use the *summed* form so I can use my k-independent sum rule on the anticommutator. So divide both sides by the double commutator and sum over **k**:

  Σ_**k** |⟨[A,C]⟩|² / ⟨[[C,H],C⁺]⟩  ≤  (β/2) Σ_**k** ⟨{A,A⁺}⟩ ≤ (β/2)·2ℏ²S(S+1)N² = β ℏ² S(S+1) N².

On the left, |⟨[A,C]⟩|² = 4ℏ²N²m² is k-independent and pulls out:

  4ℏ²N² m² Σ_**k** 1/[2Nℏ²(|bm| + S(S+1)Q̃k²)]  ≤  β ℏ² S(S+1) N²,

  i.e.   (2N m²) Σ_**k** 1/(|bm| + S(S+1)Q̃k²)  ≤  β S(S+1) N².

Convert the sum to an integral. With V_d the volume per spin, the N modes of the Brillouin zone fill it as Σ_**k** → N V_d/(2π)^d ∫ d^d k; the extra N cancels one power of N on each side, dropping the inequality from N² to a per-spin statement. The angular part gives the surface Ω_d of the unit d-sphere and the radial measure k^{d-1}dk. Restricting the integration domain to a sphere of radius k̄ inscribed in the Brillouin zone only *drops positive terms from the left*, which only *strengthens* the inequality — legitimate, and it gives a clean integral. So

  S(S+1)  >  [2 V_d Ω_d m² T /(2π)^d] ∫₀^{k̄} dk · k^{d-1} / ( |b m| + S(S+1) J̄ k² ),

where I've written J̄ for the exchange-second-moment constant and used T = 1/(k_B β) (setting k_B=1). Now the dimension lives in exactly one place: the volume element k^{d-1} dk. Everything else is dimension-blind. Let me just do the integral in each d and watch the b → 0 limit.

Take d = 1 first. The integral is ∫₀^{k̄} dk/(bm + Jk²) where I write J = S(S+1)J̄. That's (1/√(bmJ)) arctan(k̄ √(J/(bm))). As b → 0 the argument of arctan → ∞ so arctan → π/2, and the whole thing → (π/2)/√(bmJ), which *diverges* like 1/√(bm). Plug back, keeping J = S(S+1)J̄ inside the root so I don't lose track of the spin factor: the inequality reads

  S(S+1) > const · m² T · (π/2)/√(bm · S(S+1) J̄) = const · T m^{3/2} / √(b · S(S+1) J̄).

Move the √(S(S+1)) over: (S(S+1))^{3/2} √(J̄) > const · T m^{3/2}/√b, i.e. m^{3/2} < const · (S(S+1))^{3/2} √(J̄) · √b / T, so taking the 2/3 power

  m < c₁ · b^{1/3} / T^{2/3},   c₁ = S(S+1) J̄^{1/3}.

As b → 0, m → 0. No spontaneous order in one dimension. Power-law.

Now d = 2, where the radial measure carries an extra k: ∫₀^{k̄} k dk/(bm + Jk²) = (1/2J) ln(1 + J k̄²/(bm)). This is the crucial one. As b → 0 the argument of the logarithm blows up, so the integral *diverges logarithmically* — exactly the same logarithm Bloch's heuristic stumbled into, but now it's on the rigorous side of an honest inequality. Plug back, again keeping J = S(S+1)J̄ explicit (it sits in the 1/2J prefactor):

  S(S+1) > const · m² T · (1/(S(S+1)J̄)) · ln(1 + J k̄²/(bm)).

Solve for m²: (S(S+1))² > const · m² T · ln(1 + Jk̄²/(bm)) / J̄, so m² < const · (S(S+1))² J̄ / [ T ln(1 + Jk̄²/(bm)) ]. As b → 0 the denominator → ∞, so m² → 0, hence

  m < c₂ · 1 / ( √T · √|ln b| ),   c₂ = √(2π) S(S+1) J̄^{1/2}.

It goes to zero, but only as 1/√(ln(1/b)) — *infinitely slowly*. That slowness is itself the signature of two dimensions sitting right at the edge: the order is destroyed, but barely, by a logarithm. No spontaneous magnetization in two dimensions, for any T > 0.

And d = 3, as a control: ∫₀^{k̄} k² dk/(bm + Jk²). At small k the integrand is k²/(bm + Jk²), which is perfectly finite even at b = 0 — the k² in the numerator beats the k² in the denominator, the integrand tends to 1/J, and ∫ k² dk/(bm + Jk²) → k̄/J − (something finite) as b → 0. The integral stays bounded. So the right-hand side does *not* blow up, the inequality places no vanishing ceiling on m, and the argument simply fails to exclude order. Three-dimensional ferromagnetism is left untouched — exactly as it must be, since real bulk magnets order. The whole effect is the competition between the k^{d-1} in the numerator and the k² in the denominator near k = 0: ∫ k^{d-1}/k² dk = ∫ k^{d-3} dk diverges at the origin precisely when d − 3 ≤ −1, i.e. d ≤ 2. That single power-counting line is the theorem.

Let me make sure I haven't smuggled in an assumption. Did I anywhere assume order? No — the Bogoliubov inequality holds in the exact thermal state of the finite system in a field, ordered or not. Did the result depend on which kind of order? No — **K** dropped out of the double-commutator bound and only re-entered through the definition of m, so the conclusion forbids ferromagnetic (K=0) and antiferromagnetic/staggered (K≠0) order alike. Did I need spin waves? No — the only place a "k²" appeared was the rigorous bound 1 − cos(k·R) ≤ ½k²R², not a dispersion relation. Did I need the spins to be classical or large? No — S enters only through S(S+1), the result holds for every spin magnitude including S = ½. The two hypotheses that did real work are the two I flagged: **isotropic** (so the field is the only thing breaking the symmetry and there's no gap — an anisotropy would add a constant Δ to the denominator, bm + Δ + Jk², which stays finite as b→0 and rescues the order) and **short-range** (so Q̃ = Σ|J|R² is finite and the denominator really is ∝ k² at small k — a long-range tail changes the small-k power and can rescue the order). Both assumptions are now visibly load-bearing, not decorative.

Let me also sanity-check the small puzzle from the start. Bloch's divergent ΔM and my rigorous vanishing m are the *same* logarithm in d=2, but with opposite epistemic status: Bloch assumed alignment and got a divergence that signaled inconsistency; I assumed nothing, kept the field as a regulator, and got a finite inequality whose b → 0 limit is a clean zero. The logarithm was never a sign that something blew up — it was the integral Σ_k 1/k² diverging, the infrared catastrophe of a gapless mode in low dimensions, and the right way to use it is as the denominator of an upper bound, not as a correction to an assumed state.

So I can state it cleanly. For the isotropic Heisenberg model with finite-range exchange and quantum spin S, at any temperature T > 0, in spatial dimension d = 1 the magnetization in an ordering field b satisfies m(T,b) < c₁ b^{1/3}/T^{2/3}, and in d = 2 it satisfies m(T,b) < c₂ /(√T √|ln b|), with c₁ = S(S+1) J̄^{1/3} and c₂ = √(2π) S(S+1) J̄^{1/2} and J̄ the exchange second moment. In both cases m → 0 as b → 0 for every finite T, and the bound is independent of the ordering wavevector. Therefore the one- and two-dimensional isotropic Heisenberg model has neither spontaneous ferromagnetic nor spontaneous antiferromagnetic order at any nonzero temperature. In three dimensions the corresponding integral converges and the argument gives no such bound, leaving ordinary order intact.

The causal chain in one breath: the order parameter can't be computed but it *can* be bounded by Bogoliubov's exact inequality; choosing the spin raising/lowering Fourier components puts the magnetization into the commutator slot, the spin-length sum rule caps the anticommutator, and the energy-cost double commutator is bounded by |bm| + S(S+1)Q̃k² where the k² comes from 1−cos ≤ ½k²R² (gapless because isotropic) and the finiteness of Q̃ comes from short range; dividing and summing over **k** turns the inequality into S(S+1) > const·m²T ∫ k^{d-1}dk/(bm + Jk²), whose small-k behavior is ∫ k^{d-3}dk — convergent for d=3, divergent for d ≤ 2 — so in one and two dimensions the integral forces m → 0 as the field is removed, and continuous-symmetry breaking is impossible.

To make the heart of it visible, the divergence is just a number I can watch: integrate k^{d-1}/(bm + k²) over a small sphere and shrink the field.

```python
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy renamed trapz -> trapezoid

def infrared_integral(d, bm, kbar=1.0, J=1.0, n=200000):
    """The integral I_d(bm) = ∫_0^kbar k^{d-1}/(bm + J k^2) dk that controls the bound.
    The (staggered) magnetization obeys  S(S+1) > const * m^2 * T * I_d(bm),
    so a DIVERGING I_d as bm->0 forces m->0."""
    k = np.linspace(0.0, kbar, n)
    return _trapz(k**(d - 1) / (bm + J * k**2), k)

# watch the b->0 limit: divergence in d=1,2 (order forbidden), convergence in d=3 (order allowed)
for d in (1, 2, 3):
    vals = [infrared_integral(d, bm) for bm in (1e-1, 1e-3, 1e-5, 1e-7)]
    print(f"d={d}:  I(bm) for bm=1e-1..1e-7 ->", [f"{v:.3f}" for v in vals])
# d=1: I ~ 1/sqrt(bm)   -> diverges  => m < c1 * b^{1/3}/T^{2/3} -> 0
# d=2: I ~ (1/2) ln(1/bm) -> diverges => m < c2 /(sqrt(T) sqrt(|ln b|)) -> 0
# d=3: I -> finite        -> bounded  => no vanishing ceiling on m; order survives
```
