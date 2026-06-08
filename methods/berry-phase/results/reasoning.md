I keep coming back to the question Ron Fox asked me at the end of my talk, and I cannot let it go. I had been showing how Wilkinson and I hunt for degeneracies in the spectra of triangle billiards — two parameters, two angles of the triangle, and we sweep a closed loop in that two-parameter shape space and watch the real wavefunction. If a true degeneracy sits inside the loop, the eigenfunction comes back with its sign flipped; if there is only a near-miss, an avoided crossing, it comes back unchanged. That sign flip is our detector. It is the old Herzberg and Longuet-Higgins result from 1963, the same minus one a spinor picks up under a rotation by 2π. Clean, discrete, reliable. And then Fox asks the innocent thing: put a magnetic field in the billiard, charge the particle, break time-reversal symmetry. Now the Hamiltonian is genuinely complex, not just real-symmetric. By Von Neumann and Wigner, a real-symmetric degeneracy needs only two parameters to occur, but a complex Hermitian one needs three — so the field destroys the degeneracy our loop used to enclose. The codimension jumps from two to three. So what happens to the sign change as I continue the wavefunction around the very same loop that used to surround the coincidence?

I did not have an answer in the room and I still feel the gap. Let me think about what the sign change even *is*. It is a phase. Minus one is e^{iπ}. The eigenstate is single-valued as a function of *position* — it has to be, it is a physical wavefunction — but there is nothing that forces it to be single-valued under continuation of the *parameters* in the Hamiltonian. When the Hamiltonian was real, the only phases available to a real eigenvector were +1 and −1, so the continuation could only give back a sign. Make the Hamiltonian complex and that constraint is gone. The eigenstate lives in the complex projective space now; continuing it around a loop can bring it back rotated by *any* phase e^{iγ}, not merely flipped. So Fox's question answers itself in outline: the sign change does not disappear, it *thaws* into a continuous geometric phase. The minus one was a frozen special case of something larger, visible only because time-reversal symmetry had locked the phase to ±1.

That is the picture. Now I have to make it real — I have to actually compute this γ and show it is there, that it is not something I can define away. Because the immediate objection, the one every adiabatic-theorem treatment would throw at me, is that such a phase is pure gauge and can be removed by rephasing the eigenstates. I need to see exactly where that objection holds and exactly where it breaks.

Let me set it up properly from the adiabatic theorem itself; spin can wait. I have a Hamiltonian that depends on parameters R = (X, Y, …), and I move R slowly around a closed circuit C, with R(T) = R(0), so H returns to its start. The state obeys

  H(R(t)) |ψ(t)⟩ = iℏ ∂_t |ψ(t)⟩.

At every instant there is a natural basis: the instantaneous eigenstates,

  H(R) |n(R)⟩ = E_n(R) |n(R)⟩,

and here is the first thing to be careful about — this eigenvalue equation fixes |n(R)⟩ only up to a phase at each R. There is no relation between the phase I happen to choose at one R and the phase at another. I am free to pick any smooth, single-valued choice of |n(R)⟩ over a parameter region containing C. Hold that freedom in mind; it is going to be the whole battleground.

The adiabatic theorem tells me that if I start in |n(R(0))⟩ and move slowly enough — slow compared with the Bohr frequencies set by the gaps, so that |⟨m|ṅ⟩| = |⟨m|Ḣ|n⟩/(E_n − E_m)| stays small compared with |E_m − E_n|/ℏ — then the system rides the instantaneous eigenstate, staying in |n(R(t))⟩ at time t. So |ψ(t)⟩ is, up to a phase, just |n(R(t))⟩. The familiar part of that phase is the dynamical one, exp[−(i/ℏ)∫E_n dt′]. But I refuse to assume that is all of it. Let me write the most general adiabatic state with an *extra* phase γ_n(t) sitting in front and let the Schrödinger equation tell me what γ_n must be:

  |ψ(t)⟩ = exp[−(i/ℏ)∫₀ᵗ E_n(R(t′)) dt′] · exp[iγ_n(t)] · |n(R(t))⟩.

Now substitute into iℏ ∂_t|ψ⟩ = H|ψ⟩ and watch what survives. The time derivative hits three things: the dynamical exponential, the γ exponential, and the eigenstate |n(R(t))⟩ itself, which moves because R(t) moves. Differentiating the dynamical exponential brings down −(i/ℏ)E_n, and times iℏ that is E_n |ψ⟩ — which is exactly H|ψ⟩, since |ψ⟩ rides the eigenstate. So that term on the left cancels the entire right-hand side. What is left over, after the dynamical phase has paid off the energy, is

  iℏ ( i γ̇_n |n⟩ + |ṅ⟩ ) e^{(phases)} = 0,

where |ṅ⟩ = (∇_R n)·Ṙ is the rate of change of the eigenstate as the parameters move. Divide out the phases, project onto ⟨n|, and use ⟨n|n⟩ = 1:

  i γ̇_n + ⟨n|ṅ⟩ = 0  ⇒  γ̇_n(t) = i ⟨n(R)| ∇_R n(R)⟩ · Ṙ.

There it is. The dynamical phase took care of the energy; this is a *separate*, residual rate of phase accumulation, and it is driven not by E_n but purely by how the eigenstate tilts in Hilbert space as the parameters move. Integrate it around the full circuit and the time drops out completely — Ṙ dt is just dR along the path:

  γ_n(C) = i ∮_C ⟨n(R)| ∇_R n(R)⟩ · dR.

Stare at this. The elapsed time T has vanished. Whether I crawl around C over an hour or a year, as long as I am slow enough for the adiabatic theorem, γ_n(C) is the same number — it depends only on the *path* C traced in parameter space, on its geometry, not on how fast or how the speed varies along it. That already separates it cleanly from the dynamical phase, which is an integral of E_n dt and grows without bound as T → ∞. One is a clock; this other thing is a shape.

Two quick sanity checks before I get excited. Is γ_n real? It had better be, or it is not a phase. ⟨n|n⟩ = 1 everywhere, so ∇⟨n|n⟩ = ⟨∇n|n⟩ + ⟨n|∇n⟩ = 0, which means ⟨n|∇n⟩ = −⟨∇n|n⟩ = −⟨n|∇n⟩*, so ⟨n|∇n⟩ is *purely imaginary*. Multiply by the i out front and γ_n is real. Good. Define A_n(R) ≡ i⟨n|∇_R n⟩, a real vector field on parameter space, and then

  γ_n(C) = ∮_C A_n · dR.

This A_n is begging to be read as a vector potential and γ_n as the line integral of it — the circulation of an abstract gauge field living in parameter space. Which brings me straight to the objection I have to defeat.

The objection: that phase freedom in |n(R)⟩. Suppose I rephase the eigenstate, |n(R)⟩ → e^{iμ(R)}|n(R)⟩ for some smooth real μ(R). Then ⟨n|∇n⟩ → ⟨n|∇n⟩ + i∇μ, so A_n → A_n − ∇μ. This is exactly a gauge transformation; A_n is not unique. The textbook treatment of the adiabatic theorem seizes on this: choose μ so that ∇μ cancels A_n along the path, and the integrand vanishes, and the whole extra phase is gone — declared a non-physical gauge artifact, dropped, never spoken of again. And for a moment I worry that Fox's question has no content, that the generalized "sign change" is just a phase convention.

But wait — look at *where* that cancellation has to happen. I need a single, smooth, single-valued μ(R) that kills A_n at every point of the circuit *simultaneously*. For an open path that is fine: I can always pick μ along the path to soak up the phase, because the endpoints are different points and I am free to assign μ at each. But C is *closed*. Under my rephasing the loop integral changes by

  γ_n(C) → γ_n(C) − ∮_C ∇μ · dR.

And ∮ ∇μ · dR around a closed loop, for a single-valued μ, is *zero* — it is the integral of an exact differential around a closed curve. If I let μ change by 2π times an integer while e^{iμ} remains single-valued, only γ_n itself shifts by that integer multiple of 2π, so exp(iγ_n) is still unchanged. The very rephasing that annihilates the phase on an open arc is powerless on a closed circuit, because to kill it on the loop I would need μ to be multivalued by a non-2π amount, and then e^{iμ} is not a legitimate single-valued phase choice. The phase I can sweep away pointwise refuses to be swept away globally. *That* is the resolution of Fox's question, and it is sharper than I expected: the geometric phase is precisely the part of the eigenstate's continuation that survives every legal rephasing, exactly because the circuit closes. The dynamical phase was the obvious phase; this is the one hiding inside the gauge freedom that everyone assumed was empty.

And now the analogy with Aharonov and Bohm is not loose — it is the same structure. There, a charged particle whose support never touches the field still accumulates (q/ℏ)∮A·dR = qΦ/ℏ around a flux line: a vector potential that is locally pure gauge, removable on any simply-connected patch, yet gives a real, observable phase around a loop that encircles the flux. Here A_n = i⟨n|∇n⟩ is an abstract vector potential in parameter space, locally removable, globally not — and γ_n(C) is its flux. I should be able to make this literal later, but the parallel already tells me the phase is physical: observable, by interference, exactly as the Aharonov–Bohm phase is.

So the phase is real and gauge-invariant for a closed loop. The trouble is computing it. To evaluate ⟨n|∇n⟩ I need an explicit, locally single-valued choice of |n(R)⟩, and differentiating eigenstates is awkward — and worse, if there is a degeneracy lurking, I may not even be able to choose |n(R)⟩ single-valued globally; I would be dragging branch cuts, Dirac strings, around with me. The line-integral form is conceptually clean but operationally nasty. I want a form that does not depend on the phase choice at all.

Stokes. The loop integral of A_n over C equals the surface integral of its curl over any surface S spanning C. Take parameter space three-dimensional for now — that is the case that matters for the spin and degeneracy problems, three real parameters by Von Neumann–Wigner — and

  γ_n(C) = ∮_C A_n · dR = ∬_S (∇ × A_n) · dS = i ∬_S ⟨∇n| × |∇n⟩ · dS.

Now I insert a complete set of states inside ⟨∇n| × |∇n⟩, ∑_m |m⟩⟨m|. The m = n term: that is ⟨∇n|n⟩ × ⟨n|∇n⟩, a vector crossed with its own conjugate-paired partner; since ⟨n|∇n⟩ is purely imaginary, ⟨∇n|n⟩ = −⟨n|∇n⟩ is the same imaginary vector up to sign, and A × A = 0 — the diagonal term drops out. So only m ≠ n contribute:

  γ_n(C) = −Im ∬_S dS · ∑_{m≠n} ⟨∇n|m⟩ × ⟨m|∇n⟩.

Now I want the off-diagonal ⟨m|∇n⟩ without ever differentiating the eigenstate's phase. Differentiate the eigenvalue equation H|n⟩ = E_n|n⟩ with respect to the parameters: (∇H)|n⟩ + H|∇n⟩ = (∇E_n)|n⟩ + E_n|∇n⟩. Project on ⟨m| with m ≠ n. The ⟨m|H = E_m⟨m|, and ⟨m|n⟩ = 0, so

  ⟨m|∇H|n⟩ + E_m⟨m|∇n⟩ = E_n⟨m|∇n⟩  ⇒  ⟨m|∇n⟩ = ⟨m|∇H|n⟩ / (E_n − E_m), m ≠ n.

Beautiful — this is built entirely from matrix elements of ∇H, which I can compute in *any* basis whatsoever, with no constraint on the phases of |m⟩ or |n⟩, because every place a phase of |n⟩ could enter, the conjugate appears too and it cancels. Substitute:

  γ_n(C) = −∬_S V_n(R) · dS,

with the field

  V_n(R) = Im ∑_{m≠n} ⟨n|∇H|m⟩ × ⟨m|∇H|n⟩ / (E_n − E_m)².

This is the object I was after. It is a real vector field in parameter space. With my convention A_n = i⟨n|∇n⟩, Stokes says ∇ × A_n = −V_n, so the phase is the negative flux of this field. The sign is bookkeeping; the important thing is that V_n makes no reference to ∇|n⟩ or to any phase convention at all. Phase relations between eigenstates at different R have become completely immaterial; any solutions of the eigenvalue equation give the same V_n. That is genuinely surprising, because A_n manifestly *does* depend on the phase of |n⟩, yet the field whose flux it produces does not. The phase ambiguity lives entirely in the curl-free part, and Stokes throws that part away around a closed loop. This is the same statement as the gauge-invariance argument, now made manifest in a computable formula.

I should check that this surface integral does not depend on *which* surface S I choose to span C, or the whole construction is ill-defined. Two surfaces with the same boundary differ by a closed surface, and the integral over a closed surface is the flux of V_n out of the enclosed volume, which is ∭ ∇·V_n. Away from degeneracies I can always cover the volume by patches where |n(R)⟩ is smooth. On each patch V_n = −∇×A_n, so ∇·V_n = −∇·(∇×A_n) = 0 identically; on overlaps the potentials differ by a gradient, so the curl and therefore V_n agree. Thus the closed-surface flux vanishes wherever the level is isolated, and γ_n(C) is independent of the spanning surface. Everywhere, that is, *except* where E_n − E_m → 0: at a degeneracy the denominator blows up and V_n has a singularity. The divergence-free field has its singular sources exactly at the degeneracies. Fox's degeneracy is not destroyed after all — it has become a monopole.

That denominator, (E_n − E_m)², tells me the whole geometry is organized by degeneracies. If C passes near a point R* where level n nearly meets a single other level, the term for that one neighbor dominates everything else, and I can compute γ_n essentially from the two-level problem alone. So let me work the cleanest possible case: two levels, near their degeneracy, and see what the geometric phase actually is.

Two states, call them + and − with E_+ > E_−. Near the degeneracy I can shift energies so the crossing is at E = 0 and put the degeneracy at R* = 0. Any 2×2 Hermitian Hamiltonian, traceless, is a real linear combination of the three Pauli matrices, so after a linear change of the three parameters I can write it in the standard form

  H(R) = ½ ( Z, X − iY ; X + iY, −Z ) = ½ σ·R,

with R = (X, Y, Z). The eigenvalues are E_± = ±½|R| = ±½R — they meet only at R = 0, an isolated point, three parameters needed to reach it, exactly Von Neumann–Wigner's codimension three. And the reason this form is a gift: ∇H = ½σ, a constant, so the matrix elements I need are just Pauli matrix elements.

Compute V_+(R). I will use the isotropy: rotate axes so the Z-axis points along R at the point of evaluation. With the Pauli relations σ_x|±⟩ = |∓⟩, σ_y|±⟩ = ±i|∓⟩, σ_z|±⟩ = ±|±⟩, and ∇H = ½σ, the field is

  V_+ = Im [⟨+|½σ|−⟩ × ⟨−|½σ|+⟩] / (E_+ − E_−)².

With (E_+ − E_−)² = R² and the ¼ from the two factors of ½ in ∇H, this is V_+ = (1/4R²) Im[⟨+|σ|−⟩ × ⟨−|σ|+⟩]. Take the Z-component: it is built from ⟨+|σ_x|−⟩⟨−|σ_y|+⟩ − ⟨+|σ_y|−⟩⟨−|σ_x|+⟩. Now ⟨+|σ_x|−⟩ = 1, σ_y|+⟩ = i|−⟩ so ⟨−|σ_y|+⟩ = i, σ_y|−⟩ = −i|+⟩ so ⟨+|σ_y|−⟩ = −i, and ⟨−|σ_x|+⟩ = 1. The bracket is (1)(i) − (−i)(1) = 2i, so Im[(1/4R²)·2i] gives V_z = 1/(2R²) along the local Z, which points along R. The X and Y components involve σ_z in an off-diagonal slot and vanish. Reverting to unrotated axes, the field points radially:

  V_+(R) = R / (2R³).

V_+ itself points outward with half the unit radial flux; because γ is minus the flux of V_+, the phase behaves as if a monopole of strength −½ sits at the degeneracy. (And by E_− = −E_+ in the off-diagonal-symmetric structure, V_−(R) = −V_+(R), so γ_−(C) = −γ_+(C).) Then

  γ_+(C) = −∬_S V_+ · dS = −½ ∬_S (R/R³)·dS = −½ Ω(C),

because ∬ (R/R³)·dS over a surface spanning C is exactly the solid angle Ω(C) that the circuit subtends at the degeneracy — the radial inverse-square flux through C is the apparent angular size of C as seen from R = 0. So the geometric phase factor is

  exp{iγ_±(C)} = exp{∓ ½ i Ω(C)}.

Half the solid angle. Let me make sure I believe the factor of one half and that it does the right thing in the limits. The surface-independence is automatic here: deform S to the other side of the degeneracy and Ω changes by 4π (the full sphere), so γ_+ changes by −½·4π = −2π, and exp(iγ) is unchanged — consistent, as it must be, with the monopole strength being a half so that a full sphere gives a 2π ambiguity, never less. Now confine C to a plane of real Hamiltonians, Y = 0, so the Hamiltonian stays real-symmetric and the levels cross conically — the "diabolical point." A loop in that plane around the degeneracy subtends Ω = ±2π (it sees a hemisphere's worth, the great-circle limit), so γ_+ = ∓π and exp(iγ) = −1; a loop not enclosing it gives Ω = 0 and +1. That is exactly the Herzberg–Longuet-Higgins sign change, recovered as the real-Hamiltonian special case of the continuous phase. Fox's question is fully answered: switch on the field, leave the real plane, and the −1 unfreezes into e^{−iΩ/2}, a continuous function of how the loop is tilted and how much solid angle it captures. The sign was the phase all along, seen edge-on.

Now spin, the case I can actually propose to measure. A particle of spin s in a magnetic field, H = κℏ B·S, with the components of B as the parameters R. The eigenstates |n, s⟩ are the spin projections n along B, n running from −s to s in integer steps, with energies E_n = κℏ B n — linear in field magnitude, and crucially *independent of the direction* of B. So if I compare sweeps of equal duration at fixed |B|, the dynamical phase ∫E_n dt is the same; whatever interference remains after ordinary path phases are nulled will be geometric. There is a (2s+1)-fold degeneracy at B = 0, the origin of parameter space, and that is the monopole.

Compute V_n(B). From H = κℏB·S, ∇_B H = κℏ S, and E_m − E_n = κℏ B (m − n), so the κℏ's cancel between numerator and denominator:

  V_n(B) = Im ∑_{m≠n} ⟨n|S|m⟩ × ⟨m|S|n⟩ / [B²(m − n)²].

The vector spin operator only connects n to n ± 1 (the raising and lowering operators), so only m = n ± 1 survive, each with (m − n)² = 1. Rotate axes so Z points along B, use S_z|n⟩ = n|n⟩ and the ladder relations (S_x ± iS_y)|n⟩ = √[s(s+1) − n(n±1)] |n±1⟩. Let c_+² = s(s+1) − n(n+1) and c_-² = s(s+1) − n(n−1). The up-rung m = n+1 gives Im[⟨n|S_x|m⟩⟨m|S_y|n⟩ − ⟨n|S_y|m⟩⟨m|S_x|n⟩] = −c_+²/2. The down-rung m = n−1 gives +c_-²/2. Their sum is

  ½(c_-² − c_+²) = ½([s(s+1) − n(n−1)] − [s(s+1) − n(n+1)]) = n,

so the Z-component comes out to n/B². The transverse components vanish by the same parity that killed them in the spin-½ case. Reverting to unrotated axes,

  V_n(B) = n B / B³.

V_n has radial flux 4πn; because γ is minus that flux through the swept loop, the phase behaves as if a monopole of strength −n sits at the origin of field space. Therefore

  γ_n(C) = −∬_S V_n · dS = −n Ω(C),  exp{iγ_n(C)} = exp{−i n Ω(C)},

where Ω(C) is the solid angle the swept field-direction loop subtends at B = 0. Look at what this says: γ_n depends only on the eigenvalue n, the spin component along the field — *not* on s. The strength 2s+1 of the degeneracy at the origin is invisible; a spin-½ and a spin-9/2 in the same n = ½ state pick up the same phase. And spin-½ with n = +½: γ = −½Ω. Half the solid angle, exactly the two-level monopole result of before, as it must be since s = ½ *is* that two-level problem.

The special cases line up. For a half-integer n (fermions), a full turn of the field direction in a plane gives Ω = 2π, so γ_n = −2πn = an odd multiple of −π for n = ½, and exp(iγ) = −1: the spinor sign change under 2π rotation, the Aharonov–Susskind result, now exposed as the same geometric phase as the degeneracy sign change — same mathematical origin, finally. For integer n (bosons) a full turn gives Ω = 2π and exp(−i·2π·n) = +1; no sign change from a full turn, but I can still make one — take n = 1 and sweep B around a cone of semiangle 60°, Ω = 2π(1 − cos60°) = π, and γ = −π, exp(iγ) = −1. So the phenomenon is genuinely *not* the property of fermions that the spinor story suggested; bosons carry it too, you just need the right circuit.

This is measurable, and the measurement is the point of the spin example. Split a polarized monoenergetic beam in spin state n along the field. On one arm hold B fixed; on the other, hold |B| fixed but slowly sweep its direction around a cone of semiangle θ, slow compared to the dynamical precession frequency κB, so the spin stays in the eigenstate. Recombine. The dynamical phase is identical on the two arms because E_n is blind to field direction, and any leftover propagation phase I null out by adjusting one path length at Ω = 0. What is left in the fringes is purely geometric: γ_n = −nΩ = −2πn(1 − cosθ), and the count rate goes as

  I(θ) = cos²( n π (1 − cosθ) ).

A direct geometric-phase interferometer, reading the solid angle off the fringe contrast. And the objection that the dynamical phase, growing like T, would drown the geometric one fails here because the two arms can be arranged to have the same energy history, so that clock phase cancels in the comparison. The only slowness requirement is the ordinary adiabatic one, ℏ|⟨m|Ḣ|n⟩|/|E_m − E_n|² ≪ 1, which can be met while the remaining path length and timing offsets are nulled at Ω = 0.

One more case, because it ties the whole thing back to where the loop-phase idea historically came from. The Aharonov–Bohm effect. Take a particle of charge q confined to a small box, and let R be the box's *position*; the parameter space is ordinary space. With no flux, H = H(p, r − R) and the eigenstates are ψ_n(r − R), with energies independent of R — rigid translation. Switch on a line of magnetic flux Φ that the box never penetrates; the field is zero everywhere the particle goes, but the vector potential A(R) is not, with ∮A·dR = Φ around any loop threaded by the flux line. The exact eigenstates are now the field-free ones dressed by a Dirac phase factor:

  ⟨r|n(R)⟩ = exp[ (iq/ℏ) ∫^r A(r')·dr' ] ψ_n(r − R),

single-valued in r and locally in R. Compute the connection: ⟨n(R)|∇_R n(R)⟩. The gradient in R acts on both the Dirac factor and the argument of ψ_n; the ψ_n-gradient piece integrates to zero by normalization (it is ∇_R of a constant norm), and what is left is the gradient of the Dirac phase, which pulls down −(iq/ℏ)A(R). So

  ⟨n(R)|∇_R n(R)⟩ = −i q A(R)/ℏ,  A_n = i⟨n|∇n⟩ = q A(R)/ℏ.

The abstract parameter-space vector potential has become the *literal* electromagnetic vector potential. Carry the box around the flux line:

  γ_n(C) = ∮_C A_n · dR = (q/ℏ) ∮_C A·dR = q Φ/ℏ.

Independent of n, and independent of C as long as it winds once around the flux. There was no need for the transport even to be adiabatic here, because E_n does not depend on R at all — this is an exact statement, not an adiabatic limit. So the Aharonov–Bohm phase is a geometric phase of exactly the type I have been deriving, and I have obtained it using only single-valued wavefunctions, sidestepping the usual multivalued-wavefunction objection of the elementary "two paths around the flux" picture. The thing that started the whole line of thought — an observable phase from a loop with no local field — is a special case of the general formula, the one case where the parameter-space gauge field is a real one.

Let me hold the whole chain in view. The adiabatic theorem says a slowly cycled system returns to its eigenstate; the obvious phase is dynamical, an integral of energy over time. I refused to assume that exhausts it, carried an undetermined extra phase through Schrödinger's equation, and found a residual rate γ̇ = i⟨n|∇n⟩·Ṙ whose loop integral i∮⟨n|∇n⟩·dR has lost all reference to time — a pure function of the circuit's geometry. It looked removable, a gauge artifact, until I noticed that the rephasing which kills it on an open path cannot kill it on a closed one, because ∮∇μ·dR = 0 for any single-valued phase — so the loop phase is gauge-invariant and physical, the exact analogue of the Aharonov–Bohm holonomy. To compute it without fighting the eigenstate's phase I pushed it through Stokes into a surface integral, γ_n = −∫∫V_n·dS with V_n = Im ∑_{m≠n} ⟨n|∇H|m⟩×⟨m|∇H|n⟩/(E_n−E_m)², which depends on no phase convention at all, is divergence-free, and so is surface-independent — except at degeneracies, where its (E_n−E_m)² denominator makes the phase behave as if a monopole sits there. For the generic two-level degeneracy ½σ·R the effective strength is −½ and the phase is −½ the solid angle subtended at the crossing; the real-symmetric restriction of that is precisely the old −1 sign change. For a spin in a rotated field the phase is −nΩ, half the solid angle for spin-½, depending only on the spin projection and not on the spin, and it predicts an interference experiment in which fermion *and* boson eigenstates alike carry the geometric phase. And the Aharonov–Bohm effect is the special case where the parameter-space connection is a genuine vector potential. The sign change Fox asked about was never destroyed by the magnetic field — it was the tip of a continuous geometric phase that the field's breaking of time-reversal symmetry finally let me see.

The final formulas are the artifact. The geometric phase of an adiabatically cycled eigenstate is

  γ_n(C) = i ∮_C ⟨n(R)|∇_R n(R)⟩·dR = −∬_S V_n(R)·dS,
  V_n(R) = Im ∑_{m≠n} ⟨n|∇_R H|m⟩ × ⟨m|∇_R H|n⟩ / (E_n − E_m)²,

with γ_n(C) = −½Ω(C) for the upper state of the generic two-level degeneracy and γ_n(C) = −nΩ(C) for spin s in a rotated field. If I want to *see* the −½Ω with my own eyes rather than trust the algebra, the smallest honest check is to transport the spin-½ n = +½ eigenstate numerically around a discretized cap of fixed |B| and read the loop phase off the product of consecutive overlaps — which is insensitive to the arbitrary phase the diagonalizer hands back at each site — then compare to minus half the solid angle the cap subtends:

```python
import numpy as np

sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

def H(B):                          # spin-1/2 in field B: H = (1/2) sigma . B
    return 0.5*(B[0]*sx + B[1]*sy + B[2]*sz)

def upper_eigvec(B):               # |+(B)>: the n = +1/2 state (E = |B|/2)
    w, V = np.linalg.eigh(H(B))
    return V[:, 1]                 # eigh returns ascending eigenvalues

def loop_cone(theta, N=400):       # circle of polar angle theta on the unit sphere
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    return [np.array([np.sin(theta)*np.cos(p),
                      np.sin(theta)*np.sin(p),
                      np.cos(theta)]) for p in phi]

def berry_phase(loop):             # discrete holonomy, gauge-invariant by the product:
    states = [upper_eigvec(B) for B in loop]    # eigh phases are arbitrary, but the
    prod = 1.0 + 0j                             # product of overlaps round the loop is not
    M = len(states)
    for k in range(M):
        prod *= np.vdot(states[k], states[(k+1) % M])   # <n_k|n_{k+1}>, %M closes it
    return -np.angle(prod)          # gamma = -arg( prod <n_k|n_{k+1}> )

theta = np.pi/3
Omega = 2*np.pi*(1 - np.cos(theta))             # solid angle of the cap edge
# the n = +1/2 state should give gamma = -(1/2) Omega  (mod 2*pi)
print(berry_phase(loop_cone(theta)), -0.5*Omega)
```

That is the result: a slowly cycled quantum state, beyond ticking its dynamical clock, records the geometry of its journey through parameter space as a gauge-invariant phase set by the flux of the curvature field through the circuit — minus half the solid angle for the spin-½ upper state, the old sign change as its frozen real-Hamiltonian limit, and the Aharonov–Bohm effect as the case where the connection is a real vector potential.
