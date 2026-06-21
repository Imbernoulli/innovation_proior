I keep seeing the same Fourier object under different names. On the circle the modes are integers. On the integers the modes are points of the circle. On the real line the modes are real frequencies again. In a finite abelian group the modes are rows of a character table. The formulas look different, but in every case the test function has the same form: it respects the group law and lands in complex numbers of modulus one. So the primitive object should be a continuous homomorphism `chi:G->T`.

The target cannot be arbitrary complex numbers if I want a stable harmonic theory. Multiplication in `T` records phases, conjugation gives inverses, and Haar integration against unit-modulus functions is bounded in the right way. If I let characters land in all of `C*`, I would import growth that belongs to representation theory or complex analysis, not to the unitary Fourier theory. The circle is the one-dimensional unitary group, and abelian groups should be built from one-dimensional unitary representations. That is the right size of probe.

Now I can form the probes themselves:

`G^ = Hom_cont(G,T)`.

Pointwise multiplication makes this a group. Since `G` is abelian, there is no conjugacy complication; every irreducible unitary representation is expected to be one-dimensional in this setting. But the group of probes is not just an algebraic group. I need to know when two probes are close. A character is a function on `G`, and local compactness tells me which subsets are controlled at finite scale: compact subsets. So the topology I can put on the character group is uniform convergence on compact sets, the compact-open topology. If `K subset G` is compact and `U subset T` is a neighborhood of `1`, then a basic neighborhood of the trivial character asks `chi(K) subset U`. This is exactly the topology that sees characters as continuous functions without pretending that all of `G` can be controlled at once.

Let me test this topology before believing it. If `G` is discrete, compact subsets are finite. Uniform convergence on finite subsets is just pointwise convergence on finite tests. The dual sits inside `T^G`, and a closed subgroup of a product of compact groups should be compact. So a discrete group has compact dual. If `G` is compact, then compact-open convergence is uniform convergence on all of `G`. A nontrivial character cannot be arbitrarily close to `1` everywhere: its image is a compact subgroup of `T`; if it is not trivial, it has values separated from `1`. That makes the trivial character isolated, and translating shows every character is isolated. So a compact group has discrete dual. This is already the reversal I want Fourier theory to explain: series on compact groups have discrete frequencies, while transforms on discrete groups live on compact frequency spaces.

The next question is whether the probes detect points. Given `x in G`, there is a natural functional on the probes:

`e_G(x)(chi)=chi(x)`.

This lands in `T`, and it is a homomorphism in `chi` because `e_G(x)(chi psi)=chi(x)psi(x)`. It is continuous on `G^` because evaluation at a fixed point is continuous in the compact-open topology: `{x}` is compact. So every point of `G` gives a character of `G^`. This produces the evaluation map `e_G:G->G^^`.

If this map is to recover `G`, first it must be injective. Suppose `x != 0`. I need a character with `chi(x) != 1`. For finite abelian groups this is elementary: split off a cyclic quotient where the image of `x` is nonzero, then send that cyclic group to roots of unity. For `Z`, send `1` to a point of `T` that does not kill the chosen integer. For `R`, exponentials separate points. The general locally compact abelian case has to contain the same separation principle: closed subgroups and quotients should supply enough characters so that nonzero elements survive in some circle-valued quotient. Once that is true, `e_G(x)=1` forces every character to kill `x`, hence `x=0`.

Continuity of `e_G` is also built into the topology. If characters in a compact subset of `G^` are to be controlled near `e_G(x)`, the compact-open topology on the second dual asks for uniform control on compact families of characters. This is where local compactness is doing real work: compact subsets of the dual are equicontinuous in the needed sense, so moving `x` slightly in `G` moves `chi(x)` slightly for all `chi` in such a compact family. Without this topology on `G^`, evaluation might be algebraically visible but topologically wrong.

Surjectivity is the hard part. Take a character `Phi:G^->T`. I want it to be evaluation at a point of `G`. In the easy examples this says something familiar. A character on the integer-indexed dual of the circle is determined by where it sends the generator `1 in Z`, so it is evaluation at a point of `T`. A continuous character on the compact dual of a discrete group is forced, through finite-coordinate dependence and approximation, to come from an element of the original discrete group. For `R`, a continuous character of the frequency line has the form `xi |-> exp(2 pi i x xi)`, so again it is evaluation at `x`.

The general statement should be proved by reducing to these visible cases through the structure of locally compact abelian groups: compact neighborhoods, compact subgroups, discrete quotients, and Euclidean pieces where characters are explicit. The algebraic heart is that characters are not a loose family of tests; they are all compatible tests. A character of `G^` must respect multiplication of probes, and continuity prevents it from depending on an infinite uncontrolled pattern of probes. That forces it to be evaluation at a single point. So every `Phi in G^^` has the form `Phi(chi)=chi(x)`.

Once `e_G` is bijective, I still need it to be a homeomorphism. A continuous bijection between locally compact groups is not automatically a homeomorphism in total generality, so I need the group structure and the compact-open construction. The same compact/discrete tests suggest the right behavior. Compact subgroups go to compact subgroups, discrete quotients go to discrete quotients, and dualizing twice reverses the reversal. The neighborhoods of the identity in `G` are recovered by asking which points are uniformly small on compact sets of characters. Thus the topology on `G` is exactly the topology induced from `G^^`. The evaluation map is open and continuous, hence a topological isomorphism.

So the theorem lands cleanly: for every locally compact abelian group `G`, the natural map

`e_G:G->G^^, x |-> (chi |-> chi(x))`

is an isomorphism of topological groups.

This changes what I think Fourier analysis is doing. The Fourier transform is not primarily the integral formula

`f^(chi)=int_G f(x) conjugate(chi(x)) dx`.

That formula is the analytic shadow of the duality. The variable `chi` ranges over a group in its own right, `G^`. Translation in `G` becomes multiplication by characters on `G^`; convolution on `G` becomes pointwise multiplication on `G^`; compactness of one side becomes discreteness of the other. For `T`, the dual is `Z`, so Fourier analysis becomes a series. For `Z`, the dual is `T`, so it becomes a function on a compact frequency circle. For `R`, the group is self-dual, so the classical transform hides the duality because the two sides look the same. The insight is that the transform formula is not the source of the theory. The source is that a locally compact abelian group is recoverable from its continuous circle-valued characters.
