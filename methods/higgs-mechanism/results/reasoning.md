I start with the obstruction, because it is too strong to ignore. If I hide a continuous global symmetry in the vacuum, the angular motion along the valley is not optional. It is a flat direction. In the complex scalar model I can see it with no machinery: the potential depends only on `|phi|`, the minima form a circle, and after I choose one point the radial displacement climbs the wall while the tangential displacement slides along the valley. The tangential field has no restoring force, so it is massless. Goldstone, Salam, and Weinberg make this more than a picture: in a manifestly Lorentz-invariant relativistic field theory with a conserved current and a broken continuous global symmetry, the current creates a massless spinless state. So if I try to use spontaneous symmetry breaking in particle physics, I immediately inherit a particle I often do not want.

The other obstruction is just as hard. A vector field can be massive if I write a Proca term, but that term is not locally gauge invariant. In electrodynamics I learn to take gauge redundancy seriously, because the longitudinal and time-like parts of `A_mu` are not physical photon polarizations. The massless vector has two transverse polarizations. A massive vector has three. If I write the mass explicitly, I get the third polarization at the price of throwing away the local symmetry. That feels like a bad bargain. The local symmetry is not decoration; it is the structure that controls the theory.

So the problem has two locked doors. The global broken symmetry gives me an unwanted massless scalar. The gauge theory gives me a massless vector unless I break gauge invariance explicitly. I need to see whether those two problems are actually one problem. Anderson's superconductivity analogy keeps pulling at me here. In a neutral system the phase mode is real and gapless. In a charged superconductor the phase mode is not seen as an independent massless excitation; the electromagnetic response is massive, short range. But I cannot just import a condensed-matter medium into a relativistic field theory. I need the relativistic model to say the same thing by itself.

Let me take the smallest possible model and refuse to add a vector mass by hand. I use a complex scalar with a local U(1) symmetry. With the sign convention `D_mu = partial_mu - i e A_mu`, the local transformation is

`phi -> exp(i alpha(x)) phi`,  `A_mu -> A_mu + (1/e) partial_mu alpha`.

The Lagrangian is

`L = (D_mu phi)^* D^mu phi - V(phi^* phi) - (1/4)F_{mu nu}F^{mu nu}`,

where `F_{mu nu} = partial_mu A_nu - partial_nu A_mu`, and I choose

`V = lambda(phi^*phi - v^2/2)^2`,  with `lambda > 0`.

This is exactly gauge invariant. There is no `m^2 A_mu A^mu` term. The scalar potential asks for `|phi| = v/sqrt(2)`, but I should be careful about what that means now. In the global theory different phases on the circle are distinct vacua. In the local theory the phase can be changed independently at every point by a gauge transformation. So the circle is not a circle of physically different vacua in the same way. The modulus is physical; the phase is suspicious.

I write the scalar in polar variables,

`phi(x) = (rho(x)/sqrt(2)) exp(i theta(x))`.

Under the local U(1), `theta -> theta + alpha`, while `rho` is unchanged. The covariant derivative becomes

`D_mu phi = exp(i theta)/sqrt(2) [partial_mu rho + i rho(partial_mu theta - e A_mu)]`.

The kinetic term is therefore

`(D_mu phi)^*D^mu phi = (1/2)(partial_mu rho)(partial^mu rho) + (1/2)rho^2(partial_mu theta - e A_mu)(partial^mu theta - e A^mu)`.

There it is. The phase never appears alone. It appears only in the gauge-invariant combination

`B_mu = A_mu - (1/e) partial_mu theta`.

Check the transformation: `A_mu` gains `(1/e)partial_mu alpha`, `theta` gains `alpha`, and the two changes cancel in `B_mu`. Also `F_{mu nu}(A) = F_{mu nu}(B)`, because the curl of a gradient vanishes. So the Lagrangian is really

`L = (1/2)(partial rho)^2 + (1/2)e^2 rho^2 B_mu B^mu - V(rho) - (1/4)B_{mu nu}B^{mu nu}`.

I have not fixed the physics by choosing a gauge; I have rewritten it in variables where the gauge phase is absent. The would-be angular field is not showing up as a separate physical scalar. It has been absorbed into the definition of the vector variable. If I set `theta = 0`, that is the unitary gauge version of the same statement, but the gauge-invariant variable `B_mu` makes the point harder to miss.

Now expand around the low-energy modulus. Let

`rho = v + h`.

The potential gives

`V = (lambda/4)((v+h)^2 - v^2)^2 = lambda v^2 h^2 + lambda v h^3 + (lambda/4)h^4`.

The quadratic scalar part of the Lagrangian is

`(1/2)(partial h)^2 - lambda v^2 h^2`,

so in the usual normalization `-(1/2)m_h^2 h^2 = -lambda v^2 h^2`, and

`m_h^2 = 2 lambda v^2`.

If I write the potential instead as `-mu^2 phi^*phi + lambda(phi^*phi)^2` plus a constant, then `v^2 = mu^2/lambda`, and this is `m_h^2 = 2 mu^2`. This is the radial scalar, the curvature of the Mexican hat in the physical modulus direction.

The vector term is

`(1/2)e^2 rho^2 B_mu B^mu = (1/2)e^2 v^2 B_mu B^mu + e^2 v h B_mu B^mu + (1/2)e^2 h^2 B_mu B^mu`.

The first piece is exactly the mass term for a vector field, but it has come from the gauge-invariant scalar kinetic energy, not from an explicit gauge-breaking addition. The vector mass is

`m_B = e v`.

The next pieces are interactions between the scalar excitation and the massive vector. They are not optional decoration; they are the leftover expansion of the same gauge-invariant term that generated the mass.

Now the degree count has to work, or the whole story is fake. Before choosing the nonzero scalar modulus, I have a massless vector with two physical transverse polarizations and a complex scalar with two real degrees of freedom. That is four. After the rewrite and expansion, I have a massive vector `B_mu` with three physical polarizations and a real scalar `h` with one degree of freedom. That is still four. The missing Goldstone boson has not vanished from arithmetic. Its degree of freedom supplies the longitudinal polarization of the vector. This is the point I need: the Goldstone mode is not destroyed as a variable count, but it is no longer a physical massless scalar particle.

This also tells me why the result is not a mere recombination of the global Mexican hat and a Proca field. If I had started with a Proca mass, gauge invariance would be broken explicitly and the longitudinal polarization would be installed by force. If I had started with only the global Mexican hat, the angular excitation would be a real massless particle. Here the local redundancy changes what counts as a physical angular displacement. Motions around the valley are gauge-equivalent descriptions once the symmetry is local. The vector field is already required by local invariance, and the scalar phase is exactly the variable that can become its longitudinal polarization.

The Goldstone theorem tension is now less mysterious. In a covariant gauge I may still see a zero-mass pole associated with the phase variable, but I have to ask whether it is a physical particle. Gauge theories contain redundant variables precisely so that relativistic covariance and local constraints can coexist. In a physical gauge, the assumptions used to turn the broken current into a conserved charge that creates a physical massless state can fail; the surface terms and gauge constraints are not innocent. In gauge-invariant variables the lesson is cleaner: the phase `theta` is pure gauge in the Abelian model where `phi` is nonzero, and the physical variables are `rho` and `B_mu`. The massless scalar is absent from the physical spectrum.

I should also make sure the limiting cases behave. If I take `e -> 0`, the vector decouples, `B_mu` is no longer a useful absorption mechanism, and the scalar phase returns as the global Goldstone mode. The radial mode remains massive. That is exactly the global symmetry-breaking limit. If I keep `e` nonzero but force `v -> 0`, the vector mass disappears, and I am back to the massless gauge field. So the mass is tied to both ingredients: local coupling and nonzero scalar modulus.

For a non-Abelian group the same accounting should be written without relying on the circle picture. Let the scalar multiplet have vacuum value `v`, and let the covariant derivative be `D_mu phi = partial_mu phi - i g A_mu^a T^a phi`. Expanding the scalar kinetic term around `phi = v + ...`, the terms quadratic in gauge fields are

`g^2 A_mu^a A^{mu b} (T^a v)^\dagger (T^b v)`,

up to the conventional factor determined by real or complex normalization. Thus the gauge-boson mass matrix is

`(M_V^2)_{ab} = g^2 (T^a v)^\dagger (T^b v)`.

If a generator leaves the vacuum unchanged, `T^a v = 0`, its gauge boson remains massless. If a generator moves the vacuum along a broken direction, `T^a v != 0`, the corresponding gauge field acquires mass and the scalar degree of freedom in that direction supplies the longitudinal polarization. The broken global generators would have produced Goldstone bosons; once gauged, their directions label gauge redundancies that become longitudinal vector modes.

So the final mechanism is not "break gauge invariance." It is: keep the local gauge symmetry exact, choose a scalar potential whose physical modulus has a nonzero minimum, rewrite the scalar phase together with the gauge field into a gauge-invariant vector variable, and expand. The vector mass appears as `e^2 v^2 B_mu B^mu/2` from `|D phi|^2`; the radial scalar remains as a massive spin-0 excitation; the would-be Goldstone field becomes the longitudinal polarization of the massive vector. The theory pays for vector mass with a scalar phase degree of freedom already present, not with explicit violation of local symmetry.

