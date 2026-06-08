Let me start from where it actually hurts. I can compute the hydrogen spectrum from the
Bohr–Sommerfeld rules, I can get the fine structure, I can get the Stark effect — and then I
try the hydrogen atom in crossed electric and magnetic fields and the whole machine seizes up.
Helium: wrong. Many electrons: nothing. The anomalous Zeeman effect: I'm reduced to half-
integer quantum numbers I can't justify. And every time I trace the failure back, it ends in
the same place — I wrote down an *orbit* for the electron, ∮p dq = nh on that orbit, energies
from the Hamiltonian evaluated on that orbit, and the orbit is exactly the thing that has no
business being there.

Why do I say it has no business being there? Not out of philosophical squeamishness. Suppose I
actually tried to watch the electron go around. I'd need light of wavelength short enough to
resolve an atomic orbit, which means very high frequency, which means each quantum carries a
huge momentum. The very first quantum I scatter off the electron to see it — by the Compton
effect — knocks it clean out of the orbit. So I get, at most, one point, once, and then the
orbit I was trying to observe no longer exists. The position of the electron as a function of
time, its orbital period: these are not quantities I am merely failing to measure for want of a
better apparatus. There is no apparatus. They are not in the world in the operational sense.

And here's the thing that should have bothered me from the start. The Bohr frequency
condition, ν = (W(n) − W(m))/h, which is valid in full generality and which I trust
completely — that condition is itself a flat negation of classical kinematics. Classically a
bound charge radiates at its orbital frequency and the overtones of it. Bohr says: no, it
radiates at an *energy difference* over h, with no necessary relation to any orbital period at
all. So I am in the absurd position of building a theory on classical orbits whose single most
reliable empirical input already declares that the kinematics those orbits live in is wrong. I
cannot keep treating the orbit as real and the frequency condition as a strange exception. The
frequency condition is the truth; the orbit is the fiction.

So what *is* real — what does an experiment actually hand me? Spectral lines. For each line, a
frequency and an intensity (and a polarization). The frequency of a line is ν(n, n−α) =
(1/h)[W(n) − W(n−α)] — tagged by *two* states, the one I start in and the one I land in. The
intensity is proportional to the square of an amplitude, also tagged by the two states of the
jump. That's the whole list of observables. Frequencies between pairs of states, and
amplitudes between pairs of states. Everything else — where the electron "is" between jumps,
how fast it "goes around" — is scaffolding I invented and can never check.

Let me try to take that completely seriously and see what a mechanics looks like if I refuse,
on principle, to write down the position x(t) at all, and allow myself only quantities carried
by transitions. I don't yet know if it can be done. But I have one piece of evidence that it
*can*, and it's worth dwelling on because it's the whole reason I think this is possible rather
than just desirable.

When Kramers built the dispersion formula, and when we extended it together, something
remarkable happened to the bookkeeping. You start from a classical formula for how an atom
scatters light — and classically that formula is full of orbital amplitudes and orbital
frequencies, all attached to a particular orbit. Then you apply the correspondence-principle
transcription: replace the classical amplitudes by transition probabilities, the orbital
frequencies by transition frequencies, and — this is the move that does the work — replace the
derivative with respect to the action, d/dJ, by a difference over neighboring states,
(1/h)Δ/Δn. And when the dust settles, the orbit has vanished from the formula. What's left
depends *only* on the transitions between states — the ν(n, n−α) and the amplitudes attached to
those jumps. A quantity that classically lived on one orbit has turned, in the quantum formula,
into an array of numbers spread across all the transitions between orbits. Nobody put the orbit
back in because the formula never needed it. That is the proof of concept: an observable-only,
transition-indexed description is not a fantasy; we already have one, for dispersion, and it
works.

So the plan crystallizes. I will *keep the classical laws* — the equation of motion, the phase
integral — because I have no reason to think the dynamical relations are wrong; what's wrong is
the kinematics, what the symbols *mean*. I will reinterpret every classical quantity. Wherever
the classical theory has a quantity carried by an orbit and labeled by one state n and one
harmonic α, I will put a quantity carried by a transition and labeled by the *pair* of states
(n, n−α), oscillating at the observable frequency ω(n, n−α). The orbit is gone from the
foundation; the laws stay; only the nature of the things the laws relate has changed. If this
is going to be a mechanics and not just a slogan, the entire content now sits in one question:
when I multiply two of these transition quantities together, what do I get?

I have to ask that question because dynamics is full of products. The radiated field in the
next approximation has terms like v̇v; the energy has x²; any force law f(x) beyond the linear
one has x², x³, and so on. I cannot do mechanics without knowing how to square and multiply my
new objects. So let me build the multiplication rule from the bottom, using the one bridge I
trust — the classical Fourier representation — as scaffolding, and then reinterpret.

Classically, take the coordinate of a periodic motion in state n and write it as a Fourier
series:

  x(t) = Σ_α A_α(n) e^{i α ω(n) t},

sum over integer harmonics α, with ω(n) the orbital frequency. For α > 0, the α-th term is
the piece the correspondence principle pairs with the transition n → n−α; the negative
harmonic is the conjugate piece with the reversed frequency. I have to keep that orientation
straight, because in the new notation the two indices name the actual pair of states, not just
an algebraic sign on α. Now take a second quantity y(t) = Σ_β B_β(n) e^{i β ω(n) t}. What is
the product x(t) y(t)? Multiply term by term:

  A_α e^{i α ω t} · B_{β−α} e^{i (β−α) ω t} = A_α B_{β−α} e^{i β ω t}.

To get the coefficient of the frequency β ω in the product, I collect every way α and β−α can
add to β:

  C_β(n) = Σ_α A_α(n) B_{β−α}(n).

That's the classical product rule — a convolution of the two coefficient sequences. Notice it
is completely symmetric: swapping x and y just renames the dummy α to β−α and gives the same
C_β. Classically x y = y x, as it must.

Now reinterpret. For a positive-frequency leg, the factor is no longer A_α(n)e^{iαω(n)t} on
an orbit; it is X(n, n−α)e^{iω(n,n−α)t} on the transition n → n−α. The opposite-frequency
piece is carried by the reversed leg, X(n−α, n)e^{iω(n−α,n)t}, not by a formal
one-state Fourier label. So the product of a term of x and a term of y is

  X(n, n−α) e^{i ω(n,n−α) t} · Y(?, ?) e^{i ω(?,?) t}.

And here I have to stop and think, because the question marks are the whole game. In the
classical case the two exponentials just added their phases: αω + (β−α)ω = βω, automatically,
because everything is a multiple of the single orbital frequency ω. But my new frequencies are
*not* multiples of one fundamental. They obey the Ritz combination principle instead: the
frequency of a line from n to n−β equals the sum of the frequencies of two lines that go from n
to some intermediate state and from there to n−β. Let me write that out. Is

  ω(n, n−α) + ω(n−α, n−β) = ω(n, n−β)?

Substitute ω(a,b) = (1/ℏ)[W(a) − W(b)]:

  (1/ℏ)[W(n) − W(n−α)] + (1/ℏ)[W(n−α) − W(n−β)] = (1/ℏ)[W(n) − W(n−β)].

The W(n−α) cancels — yes. It's an identity, automatically, *provided the second factor begins
at the state where the first factor ended.* That's the constraint the question marks were
hiding. If I want the product term to sit at an allowed frequency ω(n, n−β) — and I must, because
only allowed frequencies appear in radiation, that's the Ritz principle and it's observable
law — then I cannot let any term of x multiply any term of y. I can only multiply a transition
n → n−α by a transition that *starts at n−α*, i.e. n−α → n−β. The intermediate index has to
match. Any other pairing lands on a frequency ω(n,n−α) + ω(n−α', n−β) with α' ≠ α; in general
that is not of the form (1/ℏ)[W(n) − W(n−β)], so it is not a line frequency I can use in a
general mechanics.

So the product, collected at frequency ω(n, n−β), is forced to be

  C(n, n−β) = Σ_α X(n, n−α) Y(n−α, n−β).

There's no freedom here. The Ritz principle made this almost inevitable — the moment I insist
every term live at an observable frequency, the chaining-through-the-intermediate-state
structure is the only thing that works. Let me look at what I've written. The first index of
the first factor is n; the second index of the first factor is the first index of the second
factor, n−α; the second index of the second factor is the final index n−β; and I sum over the
shared intermediate index. First-index-of-this matches second-index-of-that, summed over the
junction.

Now compare x y with y x. For y x I'd write C'(n, n−β) = Σ_α Y(n, n−α) X(n−α, n−β). These are
*not* the same sum. In x y the array X is evaluated on the leg (n, n−α) and Y on (n−α, n−β); in
y x it's Y on the first leg and X on the second. There is no rearrangement of the dummy index
that turns one into the other, because the index that's summed over sits in the *middle* of an
ordered chain, not symmetrically as in the classical convolution. So in general

  x y ≠ y x.

Let me make sure I believe this and it isn't an artifact of sloppiness. Classically I had
C_β = Σ_α A_α B_{β−α}, and swapping gave Σ_α B_α A_{β−α} = Σ_α A_{β−α} B_α — same thing,
because both indices on each factor were just additive labels α, β−α with nothing distinguishing
"where it starts" from "where it ends." The reinterpretation broke that symmetry: each factor
now carries an *ordered* pair (start, end), and the product glues end-to-start. Gluing is not
symmetric. So the non-commutativity is not a mistake I can polish away — it is forced by exactly
the same Ritz structure that forced the chaining rule in the first place. The two faces of the
new kinematics are one face.

This genuinely unsettles me, and I want to be honest about that rather than hide it,
because it's the strangest thing in the whole construction. Whereas classically x(t)y(t) is
always equal to y(t)x(t), in the quantum theory it need not be. It is a real difficulty. In one
special case it goes away — when y is x itself, or a power of x, like x · x², the two factors
are the same array and the asymmetry has nothing to bite on, so x² and x³ are unambiguous, and
that's why I can build f(x) = Σ powers of x and the radiation terms like x² without trouble.
But the moment two *different* quantities meet — position and velocity, say — the order matters.
For something like v v̇ I'll have to decide what to write; the symmetric combination (v v̇ + v̇ v)/2
is the natural thing if I want it to be the time-derivative of v²/2. The difficulty is real and
I'll have to live inside it. I keep coming back to it because I have never had to think this way
about ordinary quantities before; multiplication has always just commuted.

Let me stare at the structure of C(n, n−β) = Σ_α X(n, n−α) Y(n−α, n−β) once more. If I lay out
X as a table of numbers with rows labeled by the first index and columns by the second — entry
X(n, n−α) in row n, column n−α — then this sum, fix the row n and the final column n−β, run over
the intermediate, is row n of X dotted into column n−β of Y. That is useful: it gives me a
precise bookkeeping rule I can repeat without ever reconstructing an orbit. The product is
ordered because the two legs are ordered, and the middle index is the state through which the
transition chain passes. The non-commutativity I stumbled on is therefore not an accidental
blemish on the rule; it is the cost of making products live only at Ritz-allowed frequencies.
The whole observable-array idea hangs together.

Good. I have a kinematics. Now the dynamics: given the forces, how do I actually pin down the
X(n, n−α) and the frequencies ω(n, n−α)? Classically there are two steps. First, integrate the
equation of motion

  ẍ + f(x) = 0.

Second, fix the constants of the periodic motion with the phase integral

  ∮ p dq = ∮ m ẋ dx = J = n h.

I'll carry both over by reinterpretation. The equation of motion I adopt directly: ẍ + f(x) = 0
now means the same relation among the transition-array reinterpretations of ẍ and f(x), with all
products read by my chaining rule. Substituting a Fourier-type expansion and matching terms will
give recursion relations for the amplitudes, just as it does classically — only now the products
inside f(x) chain through intermediate states.

The phase integral needs more care, and this is where I think the real subtlety of the quantum
condition lives. Let me first see what ∮ p dq looks like in the classical Fourier
representation, because I'll reinterpret *that* form. With x = Σ_α a_α(n) e^{i α ω_n t},

  m ẋ = m Σ_α a_α(n) · i α ω_n e^{i α ω_n t},

and the phase integral over one period is ∮ m ẋ² dt. When I multiply ẋ by itself and integrate
over a period, every cross term e^{i(α+α')ω t} with α + α' ≠ 0 averages to zero; only the terms
with α' = −α survive, each contributing a full period. Working it out,

  ∮ m ẋ² dt = 2π m Σ_α a_α(n) a_{−α}(n) α² ω_n.

Since x is real, a_{−α} = ā_α, so a_α a_{−α} = |a_α|², and

  ∮ m ẋ² dt = 2π m Σ_α |a_α(n)|² α² ω_n = n h.   (★)

Now, do I just reinterpret (★) directly and set it equal to nh? I could, but something is
wrong, and it's the thing that's been generating half-integer quantum numbers and general
embarrassment. The condition ∮p dq = nh fixes J as a multiple of h, but from the standpoint of
the correspondence principle the action J is only ever determined *up to an additive constant* —
the same reason the correspondence principle pins frequencies only as differences. So writing
J = nh is over-committing: I'm asserting the absolute value of something only defined up to a
constant, and the leftover freedom shows up empirically as the integer-vs-half-integer mess. The
observable content of (★) is not its absolute value but how it *changes* from one state to the
next, because the action enters physics only through dJ/dn = h. So differentiate. Treat (★) as a
function of n and differentiate with respect to n; since J = nh, dJ/dn = h, giving

  h = 2π m d/dn Σ_α α² ω_n |a_α(n)|²,

which I'll write, pulling one factor of α out front to match the structure I'm about to
transcribe,

  h = 2π m Σ_α α · d/dn ( α ω_n |a_α(n)|² ).   (15)

This is now a relation involving the action only through its rate of change — exactly the
observable part. And the right-hand side has the form I recognize: a sum over harmonics of an
α-times-derivative-with-respect-to-n of an amplitude-frequency product. That is *precisely* the
shape Kramers and I had in dispersion theory, the thing the correspondence transcription acts on.
So I apply the same transcription that worked there. In the two-index form used for dispersion,
the α-times derivative becomes the difference between the line above level n and the line below
it,

  α ∂Φ(n,α)/∂n  ↔  Φ(n+α, n) − Φ(n, n−α).

In words: the classical α-times-derivative of a quantity becomes the *difference* between the
positive-frequency line that connects n+α to n and the downward line from n to n−α. The first
term has to be ordered as (n+α, n), because its positive frequency is
(1/ℏ)[W(n+α) − W(n)]; writing it as (n, n+α) would flip the sign of the frequency term. Apply it
to (15), with Φ the amplitude-frequency product |a|² ω. The sum over all α (positive and negative)
folds: the −α terms duplicate the +α terms with the two legs swapped, so the two-sided 2π sum
becomes a one-sided sum over α ≥ 1 with the factor doubled to 4π, and the α = 0 term carries no
transition and drops. The result is

  h = 4π m Σ_{α=1}^{∞} [ |a(n+α, n)|² ω(n+α, n) − |a(n, n−α)|² ω(n, n−α) ].   (16)

There's the quantum condition — and it is built entirely from observable quantities: transition
amplitudes and transition frequencies, the positive-frequency lines above n minus the downward
lines below n. No orbit, no absolute action, no arbitrary additive constant left dangling in a way
that hurts.

Let me check what just happened, because it's almost too good. I recognize the right-hand side.
A sum over all transitions from a state, of (amplitude-squared times frequency), upward minus
downward — that is the Thomas–Reiche–Kuhn sum rule, the high-frequency limit of the Kramers
dispersion formula. I set out to transcribe the phase integral and I landed on a sum rule we
already knew independently from dispersion theory. That is the point: the same structure that
governs how the atom scatters light governs how its action is quantized. I write down this
condition, I translate it according to the scheme of dispersion theory, and I get the
Thomas–Kuhn sum rule — and that is apparently how it is. The new quantum condition isn't an
ad-hoc imposition; it's the same observable relation showing up in a second place.

One loose end. Differentiating (★) cost me one constant of integration — equation (16)
determines the amplitudes only up to an additive constant in n, the same kind of freedom (★) had
in its absolute level. I need one more physical fact to nail it. There it is: there must be a
lowest state, a normal state n₀, below which there is nothing, so no radiation can be emitted
going down from it. That means every downward amplitude out of the normal state vanishes:

  a(n₀, n₀ − α) = 0   for all α > 0.

That single boundary condition fixes the leftover constant. And notice what it buys: it forces
the quantum numbers to come out *integer*, with the normal state at n₀, rather than leaving the
integer/half-integer question open. The half-integer disease was a symptom of imposing
∮p dq = nh and pretending the absolute action was meaningful; once I quantize only the observable
difference and fix the constant by "the ground state doesn't radiate downward," the disease is
gone. The question of half-integer versus integer quantization simply doesn't arise in a theory
that uses only relations between observable quantities.

So I claim a complete method: equations (11) the equation of motion and (16) the quantum
condition, read with the chaining multiplication, together determine the frequencies, the
energies, *and* the transition amplitudes. No inspired guesswork left, no orbit anywhere. Let me
prove it works on the hardest simple thing I can — an anharmonic oscillator, where the nonlinear
term forces the multiplication rule to actually do something. Take

  ẍ + ω₀² x + λ x² = 0.   (17)

Classically I solve this by Fourier expansion with the coefficients as power series in λ, and
the structure is that the α-th harmonic first appears at order λ^{α−1} — the nonlinearity pumps
amplitude up the harmonic ladder one rung per power of λ:

  x = λ a₀ + a₁ cos ωt + λ a₂ cos 2ωt + λ² a₃ cos 3ωt + … + λ^{τ−1} a_τ cos τωt.

(There's a constant term λ a₀ because x² has a nonzero average — the oscillation rectifies a DC
shift.) Substitute into (17), use cos²θ = (1 + cos 2θ)/2 to turn the x² products into sums of
cosines, and collect the coefficient of each cos kωt to zero. Order by order:

  ω₀² a₀ + a₁²/2 = 0,                 (the DC term: x² contributes a₁²/2)
  −ω² + ω₀² = 0,                      (the fundamental: so ω = ω₀ to lowest order)
  (−4ω² + ω₀²) a₂ + a₁²/2 = 0,        (second harmonic: −(2ω)² a₂ from ẍ, plus a₁²/2 from x²)
  (−9ω² + ω₀²) a₃ + a₁ a₂ = 0,        (third harmonic: −(3ω)² a₃, plus a₁ a₂ from x²)
  …                                                                                    (18)

Now the quantum version. The ansatz reinterprets a real cosine coefficient with the same
orientation care as before. For τ > 0, the term a(n,n−τ) cos ω(n,n−τ)t supplies two complex
array entries of size a/2, one on n → n−τ and one on the reversed leg n−τ → n; the diagonal
constant term a(n,n) is not split. Products in x² are then read by the chaining rule, so the
coefficient on n → n−k is a sum over intermediate states. The DC equation is the first place the
factors matter. The classical "a₁²/2" came from the fundamental beating against itself. Here the
zero-frequency loop can go n → n−1 → n or n → n+1 → n, and each nonzero leg is half of a cosine
coefficient, so each two-leg loop contributes a factor 1/4:

  ω₀² a(n, n) + ¼ [ a²(n+1, n) + a²(n, n−1) ] = 0.

The fundamental: −ω²(n, n−1) + ω₀² = 0. The second harmonic, on the leg n → n−2, with its x²
term built by chaining the two single-step legs n → n−1 → n−2:

  (−ω²(n, n−2) + ω₀²) a(n, n−2) + ½ a(n, n−1) a(n−1, n−2) = 0,

and the third harmonic similarly,

  (−ω²(n, n−3) + ω₀²) a(n, n−3) + ½ [ a(n, n−1) a(n−1, n−3) + a(n, n−2) a(n−2, n−3) ] = 0,
  …                                                                                    (19)

Every classical square or product has become a sum over an intermediate state — the
multiplication rule, doing exactly the work it was built for. The quantum condition (16),
specialized here, reads

  h = π m Σ_{τ=0}^{∞} [ |a(n+τ, n)|² ω(n+τ, n) − |a(n, n−τ)|² ω(n, n−τ) ]

Here the prefactor is πm rather than 4πm because I have switched to real cosine amplitudes: each
nonzero cosine amplitude splits into two complex exponential halves, so |X|² = |a|²/4. Take the
lowest approximation. The fundamental gives ω(n, n−1) = ω₀. The condition, to leading order,
keeps only the single-step legs n+1 → n and n → n−1, and reduces to a difference equation whose
solution is

  a²(n, n−1) = (n + const) · h / (π m ω₀).   (20)

Fix the constant with the normal-state condition a(n₀, n₀−1) = 0. Number the states so n₀ = 0;
then const = 0 and

  a²(n, n−1) = n h / (π m ω₀).   (★★)

The amplitude grows like √n — exactly the right classical limit for an oscillator, and it
emerged from the boundary condition, not from a guess. From the recursions (19) the higher legs
follow: classically (18) gives a_τ ∝ n^{τ/2}, and the quantum recursion gives

  a(n, n−τ) = κ(τ) √( n! / (n−τ)! ),   (21)

with κ(τ) the same n-independent factor as in the classical case. For large n, n!/(n−τ)! → n^τ,
so a(n, n−τ) → κ(τ) n^{τ/2} — the quantum amplitudes go over asymptotically into the classical
ones, as the correspondence principle demands. The construction passes its own consistency test.

Now the energy — the real prize, because this is where I find out whether the new mechanics is
saying something genuinely different from the old. Try the classical energy expression,
reinterpreted:

  W = m ẋ²/2 + m ω₀² x²/2 + (m λ/3) x³.

I claim this is constant — time-independent — under the new kinematics in the order I'm working.
That's a demand, not a freebie: a "constant of the motion" in the array language means the array
representing W is *diagonal*, with no off-diagonal time-dependent pieces W(n, n−α) e^{iω(n,n−α)t}
for α ≠ 0. So I have to check that the off-diagonal elements vanish. Compute W to lowest order in
λ. The cubic term is higher order, so to start it's just the harmonic part, m ẋ²/2 + m ω₀² x²/2,
built from the n → n±1 legs and the amplitude (★★). The diagonal element:

  W(n, n) = ½ m [ẋ²]_{nn} + ½ m ω₀² [x²]_{nn}.

Let me make the bookkeeping explicit so I trust the ½. Each real cosine amplitude a(n,n−1)
splits into e^{±iωt} halves of size ½a, so the array element X(n,n−1) = ½a(n,n−1) and
|X(n,n−1)|² = ¼ a²(n,n−1). The diagonal of x² is the chained sum over intermediate states,
[x²]_{nn} = Σ_k X(n,k)X(k,n) = |X(n,n−1)|² + |X(n,n+1)|² = ¼[a²(n,n−1) + a²(n+1,n)]. For ẋ the
amplitudes pick up a factor iω(n,k), so [ẋ²]_{nn} = Σ_k ω(n,k)² |X(n,k)|² = ω₀²·¼[a²(n,n−1) +
a²(n+1,n)], using ω(n,n±1) = ±ω₀. Adding,

  W(n, n) = ½ m ω₀²·¼[a²(n,n−1)+a²(n+1,n)] + ½ m ω₀²·¼[a²(n,n−1)+a²(n+1,n)]
          = ¼ m ω₀² [ a²(n,n−1) + a²(n+1,n) ].

With (★★), a²(n,n−1) + a²(n+1,n) = nh/(πmω₀) + (n+1)h/(πmω₀) = (2n+1)h/(πmω₀), so

  W(n, n) = ¼ m ω₀² · (2n+1) h/(π m ω₀) = (2n+1) h ω₀/(4π) = (n + ½) h ω₀ / 2π.   (23)

Stop and look at that ½. Classically the same calculation gives W = n h ω₀ / 2π — the energy is
n quanta. But the quantum kinematics, with the difference (n + (n+1)) coming from *two* adjacent
transitions rather than one orbit, hands me (n + ½). There is a half-quantum of energy in the
lowest state that the classical picture cannot produce; it cannot be written as "n quanta" even
for the harmonic oscillator. I did not put it in. It fell out of the fact that the energy of
state n is built from the transitions *both* up and down from n, and the average of n and n+1 is
n + ½. The zero-point energy is a consequence of the observable-array kinematics.

I still owe the off-diagonal check. The off-diagonal elements W(n, n−1), W(n, n−2), … must all
vanish for W to be a true constant. Take W(n, n−2), the first off-diagonal element that can
survive already at λ⁰. The potential term contains
⅛ mω₀² a^{(0)}(n,n−1)a^{(0)}(n−1,n−2), and the kinetic term contains the same chained
amplitudes with the opposite sign, because the two velocity factors bring in the corresponding
transition frequencies. They cancel, so W(n, n−2) vanishes at that order. The order-λ checks are
more revealing. For W(n, n−1), four terms from ½mω₀²x², two from ½mẋ², and three from
(mλ/3)x³ give the common factor mλβ³n√n times −5/24, +1/12, and +1/8; the coefficients sum to
zero. For W(n, n−3), the three groups give the common factor mλβ³√(n(n−1)(n−2)) times +1/24,
−1/12, and +1/24; again the sum is zero. Terms farther from the diagonal begin at higher powers
of λ, so to this order these are the only cases I have to check. I have to admit I cannot prove
in general, to all orders and all off-diagonal elements, that every periodic term vanishes — but
it held for every term I actually computed, and the vanishing of the off-diagonal energy is the
crucial test, so I take its success as strong support.

The quadratic force has already exposed the product rule, but I still want a case where the
nonlinearity changes the energy itself at higher order: ẍ + ω₀² x + λ x³ = 0. Here only the odd
harmonics appear (the cubic doesn't rectify a DC term), x = a₁ cos ωt + λ a₃ cos 3ωt + …. If I
grind the recursions and the quantum condition to order λ², I get a shifted frequency

  ω(n, n−1) = ω₀ + λ · (3 n h)/(8 π ω₀² m) − λ² · (3 h²)/(256 ω₀⁵ m² π²)(17 n² + 7) + …,

amplitudes a(n,n−1), a(n,n−3) as the corresponding power series, and the energy, defined as the
constant term in m ẋ²/2 + m ω₀² x²/2 + (m λ/4) x⁴,

  W = (n + ½) h ω₀/2π + λ · 3(n² + n + ½) h²/(8 · 4π² ω₀² m)
        − λ² · h³/(512 π³ ω₀⁵ m²) · (17 n³ + (51/2) n² + (59/2) n + 21/2).

When I instead treat the quartic as a perturbation and run the Kramers–Born dispersion-style
procedure, I get *precisely* this same energy. Two independent roads — my equation-of-motion-
plus-quantum-condition method, and the dispersion perturbation method — meeting on the same
expression. That coincidence is a remarkable vote of confidence in the basic equations. And as a
final internal check, the frequencies and energies I computed satisfy

  ω(n, n−1) / 2π = (1/h) [ W(n) − W(n−1) ]

— the Bohr frequency condition — not because I imposed it, but as a consequence of the
construction. The energy differences I built from transitions reproduce the transition
frequencies I started from. The thing closes on itself.

Let me also touch the rotator, because it tests the kinematics on a system that is barely
"mechanical" at all. An electron at fixed distance a, rotating uniformly — the equation of
motion says only that. The quantum condition (16) gives h = 2π m [a² ω(n+1, n) − a² ω(n, n−1)],
a difference equation whose solution with the normal-state condition is ω(n, n−1) = h n/(2π m a²);
the energy W = ½ m v² works out, via the array form of v², to W = (h²/8π² m a²)(n² + n + ½). And
when I extend to a rotator that can also precess slowly about an external axis, the array
relations (the kinematic chaining rule applied to x² + y² + z² = a²) reproduce the
Goudsmit–Kronig–Hönl intensity formulas and the Ornstein–Burger sum rules for multiplets — which
were established empirically and independently. The kinematics predicts the line intensities
correctly. That's external support for the multiplication rule itself, from data it wasn't
built to fit.

So here is the whole chain, start to end. The orbit is unobservable — no experiment reaches the
electron's position or period, and the one input I fully trust, the Bohr frequency condition,
already declares the orbital kinematics void. The observables are transition frequencies and
amplitudes, each tagged by a *pair* of states. Keep the classical laws but reinterpret every
quantity as an array over transitions. Demand that products live at allowed (Ritz) frequencies,
and the multiplication rule is forced — chain through the intermediate state, C(n, n−β) =
Σ_α X(n, n−α) Y(n−α, n−β) — and the very same chaining makes it non-commutative, x y ≠ y x, the
behavior of square arrays of numbers under composition. Reinterpret the phase integral in its
differentiated form (because only dJ/dn = h is observable), apply the dispersion transcription
α∂Φ(n,α)/∂n → Φ(n+α,n) − Φ(n,n−α), and out comes the quantum condition h = 4π m Σ_{α≥1}
[|a(n+α,n)|² ω(n+α,n) − |a(n,n−α)|² ω(n,n−α)] — which is the Thomas–Kuhn sum rule, fixed
absolutely by the normal-state condition a(n₀, n₀−α) = 0 that also forces integer quanta.
Equation of motion plus quantum condition, read with the new multiplication, determine
everything, with no guesswork; and on the anharmonic oscillator they yield W = (n + ½)hω₀/2π with
its zero-point half-quantum, off-diagonal energies that vanish, agreement with the Kramers–Born
result, and the Bohr frequency condition as an output. The same transition data now supply
radiation, products, quantization, energy, and frequency differences. The orbit is gone.

```python
def array_product(X, Y, states):
    """Chained product forced by the Ritz combination principle.

    C(n, n-beta) = sum over intermediate m of X(n, m) * Y(m, n-beta).
    X, Y: dict keyed by (i, j) giving the transition amplitude on leg i -> j.
    This is the row-by-column rule for ordered two-index arrays."""
    C = {}
    for i in states:
        for k in states:
            C[(i, k)] = sum(X.get((i, m), 0) * Y.get((m, k), 0) for m in states)
    return C


def quantum_condition_residual(a, omega, n, m, alpha_max):
    """Observable quantum condition; this value should equal h.

    h = 4 pi m * sum_{alpha>=1} [ |a(n+alpha, n)|^2 * omega(n+alpha, n)
                                  - |a(n, n-alpha)|^2 * omega(n, n-alpha) ]."""
    s = 0.0
    for alpha in range(1, alpha_max + 1):
        up   = abs(a.get((n + alpha, n), 0))**2 * omega.get((n + alpha, n), 0)
        down = abs(a.get((n, n - alpha), 0))**2 * omega.get((n, n - alpha), 0)
        s += up - down
    return 4 * 3.141592653589793 * m * s     # = h


def harmonic_oscillator_lowest_order(n, h, m, omega0):
    """Lowest-order oscillator result from the condition and normal-state boundary."""
    a2_down = n * h / (3.141592653589793 * m * omega0)
    a2_up = (n + 1) * h / (3.141592653589793 * m * omega0)
    energy = 0.25 * m * omega0**2 * (a2_down + a2_up)
    return a2_down, energy
```
