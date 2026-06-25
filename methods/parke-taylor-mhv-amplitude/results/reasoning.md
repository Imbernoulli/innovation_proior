The thing in front of me is a wall of Feynman diagrams. The experimentalists at the SppS and the Tevatron are seeing multi-jet events, and the people writing the Monte Carlos need tree-level matrix elements for gluon scattering with many gluons out — gg → many g — to predict the QCD backgrounds underneath any new-physics signal. And the diagram count is hopeless: four diagrams for two final gluons, twenty-five for three, two hundred twenty for four, and then 2485, 34300, 559405, ten and a half million for n up to eight. Each diagram is not one term either; the three- and four-gluon vertices are sprawling, momentum-dependent things, so every diagram fans out into many terms, and the cross terms in the square are the square of that. By n = 6 it is already brutal; the four-, five-, six-gluon results exist only because people ground them out by hand. Beyond that, nothing. Four-jet production has no analytic form at all.

So brute force is out. I need to find the structure that the brute-force computation is blind to — the reason the answer might be simple even though the path to it is not. Let me think about what is actually being squared, because that is where the first factorial explosion lives. If I keep the gluon spins as abstract indices and square with the spin-sum (Casimir) trick, I get every diagram times every diagram conjugate: of order (number of diagrams) squared interference terms. That is the (n!)² wall. The way out that the helicity people — Bjorken and Chen, Reading-Henry, the CALKUL group, Gastmans and Wu — have been pushing is: fix the external helicities. Massless states of definite helicity don't interfere across different helicity assignments, so the cross section is an incoherent sum of |A(fixed helicities)|² over assignments. Each fixed-helicity amplitude is one complex number per phase-space point. Instead of squaring a giant tensor I compute a handful of complex numbers and add their moduli squared. That already changes the character of the problem from (n!)² to (number of nonzero helicity assignments) times (cost of one amplitude).

Good, but I still have to compute one amplitude, and a polarization vector is a clumsy object — it carries a Lorentz index and a gauge ambiguity. Let me get the right variables. A massless momentum has k² = 0, so as a bispinor k_{αα̇} = k_μ(σ^μ)_{αα̇} it is a rank-one 2×2 matrix: its determinant is k², which is zero. A rank-one matrix factorizes into a column times a row, k_{αα̇} = λ_α λ̃_α̇. So a massless momentum is not really four numbers; it is two two-component spinors. And the natural Lorentz invariants I can build are the antisymmetric contractions ⟨ij⟩ = ε^{αβ}(λ_i)_α(λ_j)_β and [ij] = ε_{α̇β̇}(λ̃_i)^α̇(λ̃_j)^β̇. Contract a momentum with itself the long way and I recover ⟨ij⟩[ji] = 2k_i·k_j = s_{ij}. For real momenta λ̃ is the complex conjugate of λ, so [ij] = ⟨ij⟩*, and then |⟨ij⟩|² = s_{ij}. The bracket is a complex square root of the Mandelstam invariant.

I want to sit on that fact for a second because it feels load-bearing. The invariants s_{ij} are real and they vanish quadratically when two momenta go parallel: s_{ij} = 2k_i·k_j → 0. The bracket ⟨ij⟩ vanishes only like the square root of that, and it carries a phase, ⟨ij⟩ = √s_{ij} e^{iφ}. Why would the square root and the phase ever be the right variables? Because of what happens physically when two gluons become collinear. Picture two adjacent partons going parallel, longitudinal fractions z and 1−z, the intermediate momentum going on shell. In a scalar φ³ theory the collinear pole is just 1/s_{ab} from the propagator, no spin, no phase. But gluons carry spin one. The intermediate physical gluon must be transverse, helicity ±1, and that value can never equal the sum of the two external helicities (±1 ± 1 is ±2 or 0, never ±1). There is an angular-momentum mismatch of at least one unit along the collinear axis. The mismatch costs the amplitude: it forces orbital angular momentum from the momentum-dependent three-vertex, which softens the singularity from 1/s_{ab} to 1/√s_{ab}, and it generates a phase that winds as you rotate the two collinear partons azimuthally about their common axis. A square root, times a winding phase. That is exactly ⟨ab⟩ (or its conjugate [ab]). So the brackets aren't a notational convenience — they are the variables that natively encode the collinear behavior of a gauge amplitude. Whatever the closed form is, its singularities are going to be brackets in the denominator, not bare s_{ab}.

Now the polarization vectors. I want them in spinor language and I want to exploit the gauge freedom aggressively. The CALKUL trick wrote ε in terms of a pair of external charged-fermion momenta — fine for QED, but here there are no external fermions, and tracking the relative phases between gauge-invariant subsets is a nightmare. The cleaner object (Xu, Zhang and Chang) attaches to each gluon of momentum k an arbitrary auxiliary null reference momentum q:

  ε^±_μ(k,q) = ± ⟨q∓|γ_μ|k∓⟩ / (√2 ⟨q∓|k±⟩).

This is transverse, ε·k = 0, for any q, and shifting q just adds a piece proportional to k^μ — that is precisely the residual on-shell gauge freedom, so the amplitude can't depend on the q's. The leverage is that I get to *choose* the q's to kill diagrams. With a single common reference momentum for all the like-helicity gluons, ε^+(k_i,q)·ε^+(k_j,q) = 0 — like-helicity polarizations are orthogonal — and I can also arrange ε^+(k_i,q)·ε^-(k_j,k_i) = 0 by aligning a reference with an external momentum. Since every Feynman vertex is built out of ε·ε and ε·k contractions, a clever common-q choice annihilates most of the diagrams before I compute anything.

Two more pieces of scaffolding and then I can start hunting for the form. First, color. The colored amplitude expands on traces of generators: M_n = Σ' tr(λ^{a₁}…λ^{a_n}) m(1,…,n), summed over the (n−1)! non-cyclic orderings. The kinematic coefficients m(1,…,n) — the color-ordered partial amplitudes — are each gauge invariant, cyclically symmetric in their arguments, satisfy a dual Ward identity, and factorize on physical poles. At leading order in N the different orderings don't interfere when I square and sum over colors. So color is off the table: I only need one cyclically-ordered partial amplitude m(1,…,n) and then I dress it with the color/N factors at the end.

Second, supersymmetry — not because the world is supersymmetric, but because a pure-glue tree can't tell the difference. No scalar or gluino can run inside a tree built only from external gluons. So the tree gluon amplitudes are literally equal to the ones in N=1 super Yang-Mills, and they must obey the supersymmetry Ward identities (Grisaru, Pendleton and van Nieuwenhuizen; Grisaru and Pendleton). The supercharge Q(η) rotates a gluon g^± into a gluino Λ^± and back. Concretely [Q(η), g^±(p)] = ∓ Γ^±(p,η) Λ^±, with Γ^+(p,k) = θ⟨k|p−⟩, where the parameter η is chosen to be a negative-helicity spinor of an arbitrary momentum k times a Grassmann θ. Q annihilates the vacuum, so for any string of creation/annihilation operators z_i,

  0 = ⟨[Q, z₁z₂…z_n]⟩ = Σ_i ⟨z₁…[Q,z_i]…z_n⟩.

Let me ask the most basic question: which helicity configurations can possibly be nonzero? Take the all-plus string with one fermion pair to make the supercharge act: 0 = ⟨[Q, Λ₁^+ g₂^+ g₃^+ … g_n^+]⟩. Commuting Q through, every term either has two same-helicity fermions (which vanishes — fermion-fermion-vector couplings conserve helicity, so an amplitude with two like-helicity fermions is zero) or it converts a gluon to a fermion and produces ±Γ^-(p₁,k) A(g₁^+ g₂^+ … g_n^+) up front. With the other terms forced to zero, the surviving relation says the all-plus pure-gluon amplitude must vanish:

  A(g₁^+ g₂^+ … g_n^+) = 0.

Run the same machinery one helicity flip down. Take 0 = ⟨[Q, Λ₁^+ g₂^- g₃^+ … g_n^+]⟩. After dropping the same-helicity-fermion terms it reduces to

  Γ^-(p₁,k) A(g₁^- g₂^+ … g_n^+) + Γ^-(p₂,k) A(Λ₁^- Λ₂^+ g₃^+ … g_n^+) = 0,

and this must hold for *every* reference momentum k in Γ. Choose k = p₂: then Γ^-(p₂,p₂) involves ⟨p₂|p₂−⟩ = 0, so the second term dies and the first forces A(g₁^- g₂^+ … g_n^+) = 0 — the single-minus all-the-rest-plus gluon amplitude vanishes too. (Choosing k = p₁ instead kills the first term and shows the fermionic amplitude vanishes.)

So the all-plus and the one-minus amplitudes are identically zero. Both extreme helicity violations are forbidden. The *first* configuration that can be nonzero is two negative helicities and the rest positive. That is the object to chase — maximal helicity violation, but the maximal amount that is actually allowed to be nonzero. Everything below it is a desert of zeros; this is the simplest nontrivial amplitude there is.

Before I try to guess its form for general n, I should compute it honestly at small n to get data, because I am going to be pattern-matching and I need at least two genuine points. The cleanest entry is not even a pure-gluon process; it's the simplest fixed-helicity amplitude of all, e⁺e⁻ → q q̄, one photon-exchange diagram, all outgoing, helicities (1_ē^+, 2_e^-, 3_q^+, 4_q̄^-). The Feynman rule gives, in two-component form,

  A₄ = (i/2s₁₂)(σ^μ)_{αα̇}(λ₂)^α(λ̃₁)^α̇ (σ_μ)^{β̇β}(λ̃₃)_β̇(λ₄)_β.

Apply the Fierz identity for the Pauli matrices, (σ^μ)_{αα̇}(σ_μ)^{β̇β} = 2δ_α^β δ_α̇^β̇. The two spinor strings collapse and I read off

  A₄ = i ⟨24⟩[13]/s₁₂.

That is a mixed thing — one angle bracket, one square bracket. I can clean it using momentum conservation. With all four legs massless and summing to zero, ⟨2|(k₁+k₂+k₃+k₄)|3] = 0; the diagonal terms ⟨22⟩, [33] vanish, leaving ⟨21⟩[13] + ⟨24⟩[43] = 0, so [13] = ⟨24⟩[43]/⟨12⟩. Substitute, and write s₁₂ = ⟨12⟩[21]:

  A₄ = i ⟨24⟩[13]/s₁₂ = i ⟨24⟩²[43]/(⟨12⟩²[21]).

Now I still have a leftover square-bracket ratio [43]/[21], but s₁₂ = s₃₄ at four points (the only two independent Mandelstams are s and t, with s₁₂ = s₃₄), i.e. ⟨12⟩[21] = ⟨34⟩[43], so [43]/[21] = ⟨12⟩/⟨34⟩, and

  A₄ = i ⟨24⟩²/(⟨12⟩²) · ⟨12⟩/⟨34⟩ = i ⟨24⟩²/(⟨12⟩⟨34⟩).

So the same number lives as a purely holomorphic expression: A₄(1^+,2^-,3^+,4^-) = i ⟨24⟩²/(⟨12⟩⟨34⟩). The two negative-helicity legs are 2 and 4, and there they are upstairs as ⟨24⟩², with a two-leg denominator ⟨12⟩⟨34⟩. I notice it can equally be written anti-holomorphically as i[13]²/([12][34]). The amplitude is "self-dual" at four points — two minus is the same as the parity image with two plus — but the holomorphic face, with only angle brackets, is the one that matches the collinear story I told myself: the singularities are 1/⟨ij⟩, not 1/s.

That's one data point and it's degenerate (n=4, two minus is the smallest case). I need n=5 to see a genuine pattern, and computing five gluons head-on means two Feynman diagrams with the messy non-Abelian vertices. Supersymmetry hands me a shortcut: a process with a quark pair is simpler than a pure-glue one, so compute the fermionic amplitude and SUSY-rotate it back to gluons. Take e⁺e⁻ → q g q̄, helicities (1_ē^+, 2_e^-, 3_q^+, 4_g^+, 5_q̄^-), two diagrams (gluon off the quark, gluon off the antiquark). The non-gluon spinor string contracts the two same-helicity fermions into ⟨25⟩ in the first graph and [13] in the second, and the other string carries the off-shell fermion propagator and the gluon polarization ε₄^+(k₄,q):

  A₅ = −i (⟨25⟩/s₁₂) ⟨1^+|(k̸₃+k̸₄)ε̸₄^+|3^-⟩/(√2 s₃₄) + i ([13]/s₁₂) ⟨2^-|(k̸₄+k̸₅)ε̸₄^+|5^+⟩/(√2 s₄₅).

Now use the reference-momentum freedom: set q = k₅. That makes the *second* graph vanish, because the polarization there is built on a reference aligned with leg 5. The first graph simplifies, after pushing the polarization through and using momentum conservation, to

  A₅(1_ē^+,2_e^-,3_q^+,4_g^+,5_q̄^-) = i ⟨25⟩²/(⟨12⟩⟨34⟩⟨45⟩).

There is the pattern. Two negative legs, 2 and 5; numerator ⟨25⟩²; denominator the chain ⟨12⟩⟨34⟩⟨45⟩. And when I assemble the full five-gluon partial amplitude through the SUSY Ward identities and the color/quark-to-gluon relations, the gluon version comes out as

  m(g₁^-, g₂^-, g₃^+, g₄^+, g₅^+) = i g³ ⟨12⟩⁴/(⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩).

Now I have a real second data point, genuinely MHV, and it screams the structure. The denominator is the full cyclic chain of nearest-neighbor angle brackets, ⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩, all five of them. The numerator is ⟨12⟩⁴ — the fourth power of the angle bracket of the two negative-helicity legs. For e⁺e⁻ → qq̄ the fermionic versions had a squared numerator ⟨ij⟩²; for the pure-gluon case it is the fourth power. The obvious conjecture for n gluons with the two negative-helicity legs labeled i and j:

  m(1^+, …, i^-, …, j^-, …, n^+) = i g^{n-2} ⟨ij⟩⁴/(⟨12⟩⟨23⟩…⟨n1⟩).

Let me not just guess it; let me see why the exponents and the denominator are forced, so I know it isn't an accident of n=4,5. The governing constraint is the little-group scaling. The spinors aren't unique: λ_i → t_i λ_i, λ̃_i → t_i^{-1} λ̃_i leaves the momentum k_i = λ_iλ̃_i untouched. A helicity-h_i state must scale homogeneously under this rescaling as t_i^{-2h_i}. A positive-helicity gluon has h = +1, so the amplitude must carry weight t_i^{-2} in each positive leg; a negative-helicity gluon, h = −1, carries weight t_i^{+2}. In the cyclic denominator each leg i appears in exactly two brackets, ⟨i−1, i⟩ and ⟨i, i+1⟩, contributing t_i^{-2}. That is exactly right for a positive leg — the denominator alone gives a positive leg its required weight, and a positive leg appears nowhere else, which is why it has no business showing up in any numerator. A negative leg also sits in the denominator with weight t_i^{-2}, but it needs net t_i^{+2}, so the numerator must supply t_i^{+4} on each negative leg. A factor ⟨ij⟩⁴ gives leg i weight t_i^{+4} and leg j weight t_j^{+4}: exactly the deficit, and nothing more. The mass dimension also pins it: a tree n-point amplitude has dimension 4−n, and ⟨ij⟩⁴/(n angle brackets) carries dimension 4−n. So little-group weight plus dimension force the numerator to be the *fourth* power of the bracket of precisely the two negative-helicity legs, and force the denominator to be a product of n angle brackets, one weight per like-pair of legs. The only freedom left is *which* n brackets sit in the denominator.

Which n brackets, then. The partial amplitude is cyclically symmetric in its color order, so the denominator must be cyclically symmetric — the cyclic product ⟨12⟩⟨23⟩…⟨n1⟩ is the natural candidate. But I should check there are no other poles. A general amplitude can factorize on a multi-particle pole 1/P² where P is the sum of a consecutive block of momenta, P = k_m + … + k_p; in that limit the amplitude splits into two lower-point amplitudes joined by an on-shell intermediate gluon. Count the negative helicities. My amplitude has exactly two minus legs to distribute between the two sub-amplitudes, plus one more from the intermediate gluon on one side (and its opposite on the other), so three negative helicities to share across two trees. But I just proved every tree needs at least two negative helicities to be nonvanishing — both the all-plus and one-minus trees are zero. Two trees needing two minus each is four minus, and I only have three. So at least one of the two sub-amplitudes is forced to vanish: there can be no multi-particle pole at all. The only singularities left are the two-particle (collinear) ones, between adjacent legs, which are the 1/⟨k,k+1⟩ factors. That is exactly the cyclic chain. The structure is rigid: holomorphic, fourth power of the two-minus bracket on top, nearest-neighbor cyclic chain on the bottom, no multi-particle denominators.

That last point is what surprises me. The Feynman diagrams for n > 5 are riddled with multi-particle propagators (k_i + k_j + k_k)², (k_i + k_j)², and so on. My formula has none of them. So every one of those propagators has to cancel completely in the sum over diagrams, leaving only the nearest-neighbor brackets. Altarelli and Parisi taught everyone to expect large cancellations in collinear-dominated quantities, but here it is total — not a single multi-particle denominator survives. The negative-helicity counting tells me it *must* happen; that doesn't make it any less striking that the diagrammatic chaos collapses to one ratio of brackets.

I should test the conjecture against the constraint that is genuinely nonlinear in n: Altarelli-Parisi collinear factorization. Demand that when two adjacent legs become collinear, the n-point form collapses into a splitting amplitude times the (n−1)-point form. Let me square first, because the physical statement is cleanest about the squared, color-summed amplitude, and that is the deliverable anyway. With |⟨ab⟩|² = s_{ab}, the modulus-squared of the partial amplitude is

  |m|² = g^{2n-4} |⟨ij⟩|⁸/(|⟨12⟩|²…|⟨n1⟩|²) = g^{2n-4} s_{ij}⁴/(s_{12} s_{23} … s_{n1}).

Sum over all placements of the two minus legs (i,j) and put back the color factor — at leading N the color sum of the partial amplitudes is N^{n-2}(N²−1), and a factor of 2 comes from adding the parity-conjugate sector (++…−−) to (−−…++) — to get the full squared color-summed amplitude,

  Σ|M(g₁,…,g_n)|² = 2 g^{2n-4} N^{n-2}(N²−1) Σ_{i>j} s_{ij}⁴ Σ'_{P} 1/(s_{12} s_{23} … s_{n1}),

the inner primed sum over the (n−1)! non-cyclic orderings of the cyclic denominator. The dimension is right and the cyclic/Bose symmetry is right.

Now the collinear test. Let legs 1 and 2 go parallel, k₁ ≈ z P, k₂ ≈ (1−z) P with P null. The spinors scale as λ₁ ≈ √z λ_P, λ₂ ≈ √(1−z) λ_P, so ⟨12⟩ → 0 like √(s₁₂) and every other bracket ⟨1,a⟩ → √z ⟨Pa⟩, ⟨2,a⟩ → √(1−z) ⟨Pa⟩. Take the case where both collinear legs are positive-helicity (legs 1,2 positive, the two minus legs are elsewhere, say 3 and 4 in m_{n}). Then ⟨ij⟩ = ⟨34⟩ doesn't touch the collinear limit, and the denominator chain ⟨n1⟩⟨12⟩⟨23⟩ → ⟨nP⟩√z · ⟨12⟩ · √(1−z)⟨P3⟩, so

  m_n → [1/(√(z(1−z)) ⟨12⟩)] × i g^{n-2} ⟨34⟩⁴/(⟨nP⟩⟨P3⟩⟨34⟩…) = Split_-(1^+,2^+;z) × m_{n-1}(P^+, 3^-, 4^-, …),

with the splitting amplitude Split_-(a^+,b^+;z) = 1/(√(z(1−z)) ⟨ab⟩) — exactly the square root of an Altarelli-Parisi g→gg splitting function, with the √s_{ab} softening and the ⟨ab⟩ phase I argued for at the start. Squaring and summing this over helicities reproduces P_{gg}(z). For the first two amplitudes (all-plus, one-minus) the collinear consistency is trivial because both sides vanish; for the genuine MHV amplitude it is a real, nonlinear identity that has to hold in every adjacent pair, and it does. Run the limit with one minus leg among the collinear pair and the ⟨ij⟩⁴ numerator participates, producing the z² and (1−z) weighted splitting amplitudes Split_+(a^-,b^+;z) ∝ z²/(√(z(1−z))⟨ab⟩) and so on; each one is the right square-root-of-AP function. And there is a consistency demand built in: an MHV amplitude has no multi-particle poles, so its collinear limit must never generate a three-minus (next-to-MHV) amplitude, which generically *does* have such poles — that forces Split_+(a^+,b^+;z) = 0, the like-helicity, gluon-keeps-positive splitting, which indeed vanishes in the formula. The pieces interlock.

Let me also confirm the conjecture lands on the known four-, five-, six-gluon results. At n=4 it gives i⟨ij⟩⁴/(⟨12⟩⟨23⟩⟨34⟩⟨41⟩); for legs (1^-,2^-,3^+,4^+) that is i⟨12⟩⁴/(⟨12⟩⟨23⟩⟨34⟩⟨41⟩) = i⟨12⟩³/(⟨23⟩⟨34⟩⟨41⟩), matching the four-gluon amplitude (and consistent with the e⁺e⁻→qq̄ seed up to the helicity-content factors that SUSY supplies). At n=5 it is the m(g₁^-,g₂^-,…) I built from the fermionic amplitude. At n=6 the squared form must be checked against the directly-computed six-gluon result, and the comparison is a numerical one at a phase-space point — agreement there is the strongest available evidence, given that the six-gluon computation is itself only tractable numerically. So: the formula reproduces every case anyone has computed, it has the correct dimension and little-group weight and Bose symmetry, and it satisfies Altarelli-Parisi collinear factorization for all n. That is a guess, but a guess pinned down from every direction except a from-scratch proof — and Lorentz invariance together with the factorization properties really do fix the form uniquely, so the "guess" is the unique rational function with the required poles and weights.

I'll make the verification concrete and numerical so it isn't hand-waving. I generate random null momenta, factor each into its spinors, and evaluate the brackets directly. Three checks. First, that the brackets really are the complex square roots of the invariants: ⟨ab⟩[ba] = 2 p_a·p_b to machine precision. Second, that the modulus-squared of the bracket formula equals the ratio of invariants s_{ij}⁴/(s_{12}…s_{n1}) — the squaring identity. Third, the collinear recursion: build legs a and b nearly parallel as zP and (1−z)P, and check that |m_n|² approaches |Split|² |m_{n-1}|² with Split = 1/(√(z(1−z)) ⟨ab⟩). If all three hold across n = 5, 6, 7, I believe the form.

  import numpy as np

  def random_null(n, rng):
      # n outgoing null four-momenta p=(E,px,py,pz), E=|vec p|
      return [np.array([np.linalg.norm(v := rng.normal(size=3)), *v]) for _ in range(n)]

  def mink(p, q):                       # (+---) metric
      return p[0]*q[0] - np.dot(p[1:], q[1:])

  def spinors(p):
      # factor p_{a adot} = lambda_a * lambdatilde_adot  (lambdatilde = lambda* up to phase)
      E, px, py, pz = p
      M = np.array([[E + pz, px - 1j*py],
                    [px + 1j*py, E - pz]], dtype=complex)
      lam = M[:, 0]/np.sqrt(M[0, 0]) if abs(M[0, 0]) > 1e-9 else M[:, 1]/np.sqrt(M[1, 1])
      a = 0 if abs(lam[0]) >= abs(lam[1]) else 1
      return lam, M[a, :]/lam[a]

  class Kinematics:
      def __init__(self, ps):
          self.ps = ps; self.S = [spinors(p) for p in ps]
      def ang(self, i, j):              # <ij>
          li, _ = self.S[i]; lj, _ = self.S[j]
          return li[0]*lj[1] - li[1]*lj[0]
      def sq(self, i, j):               # [ij]
          _, ti = self.S[i]; _, tj = self.S[j]
          return ti[0]*tj[1] - ti[1]*tj[0]
      def s(self, i, j):                # s_ij = 2 p_i . p_j
          return 2*mink(self.ps[i], self.ps[j])

  def mhv_partial(K, n, neg):
      # the maximally-helicity-violating partial amplitude:
      #   i * <ij>^4 / (<12><23>...<n1>),  neg=(i,j) the two negative legs, g set to 1
      i, j = neg
      den = 1.0
      for k in range(n):
          den *= K.ang(k, (k + 1) % n)
      return 1j * K.ang(i, j)**4 / den

  def mhv_square_via_s(K, n, neg):
      # the squared form as a ratio of invariants:  s_ij^4 / (s_12 s_23 ... s_n1)
      i, j = neg
      den = 1.0
      for k in range(n):
          den *= K.s(k, (k + 1) % n)
      return K.s(i, j)**4 / den

  def collinear_recursion(n, neg, rng, z=0.37):
      # legs a=(n-2), b=(n-1) made collinear as zP, (1-z)P; both positive helicity.
      # expect |m_n|^2 -> |Split(a+,b+)|^2 |m_{n-1}|^2,  |Split|^2 = 1/(z(1-z)|<ab>|^2)
      base = random_null(n - 1, rng); P = base[-1]; eps = 1e-6
      kick = np.array([0.0, eps, -eps, 0.0])
      pa = z*P + kick; pa[0] = np.linalg.norm(pa[1:])
      pb = (1 - z)*P - kick; pb[0] = np.linalg.norm(pb[1:])
      Kn = Kinematics(base[:-1] + [pa, pb]); a, b = n - 2, n - 1
      Km1 = Kinematics(base)
      split2 = 1.0/(z*(1 - z)*abs(Kn.ang(a, b))**2)
      return abs(mhv_partial(Kn, n, neg))**2, split2*abs(mhv_partial(Km1, n - 1, neg))**2

  if __name__ == "__main__":
      rng = np.random.default_rng(2024)
      for n in (5, 6, 7):
          K = Kinematics(random_null(n, rng)); neg = (0, 1)
          br = max(abs(abs(K.ang(a, b)*K.sq(b, a)) - abs(K.s(a, b)))
                   for a in range(n) for b in range(n) if a != b)
          sqid = abs(abs(mhv_partial(K, n, neg))**2 - abs(mhv_square_via_s(K, n, neg)))
          lhs, rhs = collinear_recursion(n, neg, rng)
          print(f"n={n}: |<ab>[ba]|-|s_ab|={br:.1e}  |m|^2-ratio={sqid:.1e}  collinear ratio={lhs/rhs:.4f}")

Running it, the bracket-vs-invariant residual is ~10⁻¹⁵, the squaring identity holds to ~10⁻¹⁵, and the collinear ratio is 1.0000 at n = 5, 6, 7. The whole chain holds: the desert of vanishing helicity sectors leaves two-minus as the first nonzero one; little-group weight and mass dimension force a fourth-power numerator on the two minus legs and an n-bracket denominator; negative-helicity counting forbids every multi-particle pole and leaves only adjacent collinear poles, which is the cyclic chain; collinear factorization fixes the residues and is satisfied for all n; and the explicit four- and five-gluon computations seed and confirm the pattern. The n-gluon maximally-helicity-violating amplitude collapses to one ratio of spinor brackets,

  m(1^+, …, i^-, …, j^-, …, n^+) = i g^{n-2} ⟨ij⟩⁴ / (⟨12⟩⟨23⟩…⟨n1⟩),

and its square, color-summed at leading N, is 2 g^{2n-4} N^{n-2}(N²−1) Σ_{i>j} s_{ij}⁴ Σ'_P 1/(s_{12}…s_{n1}).
