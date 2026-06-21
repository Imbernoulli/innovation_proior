Liquid helium below the λ-point flows through narrow capillaries and films with no measurable viscosity, and I want that frictionless flow to come out of the molecules themselves — a many-boson Hamiltonian whose low-lying excited states, when I work them out, refuse to dissipate the flow below some finite speed. The tempting story is that helium is a Bose gas, it condenses, and the condensate is the frictionless fluid: London identified the λ-transition with Bose–Einstein condensation into a macroscopically occupied lowest state, and Tisza built a two-fluid hydrodynamics on it. That organizes the phenomenology, but it never answers the one objection that decides which theories are even allowed. Take the degenerate *ideal* Bose gas, every particle in the zero-momentum state, and ask whether it carries current without friction. It cannot. Nothing stops a condensate molecule from being kicked into an excited state of momentum $f$ by a collision; that costs energy $f^2/2m$, which I can make as small as I like by choosing $|f|$ small, so for *any* relative velocity there is an arbitrarily-low-energy single-particle excitation into which a moving object dumps momentum. There is no energetic floor. The friction-or-not question therefore lives entirely in the *shape of the excitation spectrum*, and the bare $f^2/2m$ has exactly the wrong shape.

Landau answered this by quantizing the hydrodynamics and *asserting* the spectrum — phonons $\varepsilon = c\,p$ at small momentum, rotons $\varepsilon = \Delta + p^2/2\mu$ at larger momentum — and then deriving a criterion: a body at velocity $V$ can create one excitation of momentum $p$, energy $\varepsilon(p)$, only if energy and momentum both balance, i.e. only when $V \ge \varepsilon(p)/p$, so the flow is protected below $V_c = \min_p \varepsilon(p)/p$. For a linear branch that minimum is $c>0$; for the ideal gas the same criterion gives $\min_p (p^2/2m)/p = 0$ — no superfluidity. Same criterion, opposite verdict, and the only difference is the spectrum. So the whole problem reduces to one thing: produce a many-boson system whose *low-momentum* excitation energy goes like $|f|$ rather than $f^2/2m$. The ideal gas cannot — its spectrum is rigidly $f^2/2m$ — so the linear branch, if it exists, has to be *made by the interaction*. That fixes the starting point: a weakly repulsive Bose gas, weak so I can carry the interaction as a small parameter rather than face a hopeless $N$-body problem.

What I propose is the theory of the weakly interacting Bose gas built on c-numbering the condensate and a canonical transformation that diagonalizes the result — Bogoliubov theory. Write the Hamiltonian for $N$ identical bosons in volume $V$ with a pair potential, $H = \sum_i T(p_i) + \sum_{i<j}\Phi(|q_i-q_j|)$ with $T(p)=p^2/2m$, and second-quantize it in the plane-wave basis $\phi_f(q)=V^{-1/2}e^{i(f\cdot q)/\hbar}$ so that $a_f, a_f^+$ create and destroy a molecule of momentum $f$ with $[a_f,a_{f'}^+]=\delta_{f,f'}$. The kinetic term is diagonal; the potential, carried through its Fourier amplitude $v(f)=\int\Phi(|q|)e^{-i(f\cdot q)/\hbar}\,dq$, is a quartic operator coupling four momenta with momentum conservation — and that quartic term is the obstruction, since it mixes momenta and cannot be solved as it stands.

The lever is the condensate. Near zero temperature the overwhelming majority of molecules sit in the $f=0$ state, so $N_0 = a_0^+a_0$ is of order $N$. The commutator $a_0a_0^+ - a_0^+a_0 = 1$ is a relative correction of order $1/N_0$ against operators that themselves pull down $\sqrt{N_0}\sim\sqrt N$, so I treat $a_0, a_0^+$ as ordinary c-numbers, both equal to $\sqrt{N_0}$, neglecting their non-commutativity. This is what *lowers the degree* of the interaction: ordinary perturbation theory in the operators keeps the quartic term quartic and never closes, whereas every factor of $a_0$ replaced by $\sqrt{N_0}$ turns interaction legs into numbers, and keeping only the terms with the most condensate legs collapses the quartic interaction to one *quadratic* in the excited operators. Splitting the field $\Psi = a_0/\sqrt V + \vartheta$ with $\vartheta=(1/\sqrt V)\sum_{f\ne0}a_f e^{i(f\cdot q)/\hbar}$ the depleted part, and carrying the equations of motion while dropping second-and-higher powers of $\vartheta$, the condensate amplitude just rotates with phase $E_0=(N_0/V)v(0)$; de-phasing the excited amplitudes by the identical phase (call the de-phased operators $b_f$) cancels that local term, and Fourier-transforming leaves one equation per momentum:

$$ i\hbar\,\partial_t b_f = \{\,T(f) + (N_0/V)\,v(f)\,\}\,b_f + (N_0/V)\,v(f)\,b_{-f}^+ . $$

The right-hand side contains $b_{-f}^+$ — a *creation* operator for the opposite momentum. The interaction, fed by the condensate, takes two molecules out of $f=0$ and puts them into $+f$ and $-f$, or the reverse; these **anomalous pair terms** are absent from the ideal gas and are precisely what will bend the spectrum. So I do not remove them by hand — they are the mechanism — and instead close the system with the companion equation $-i\hbar\,\partial_t b_{-f}^+ = (N_0/V)v(f)\,b_f + \{T(f)+(N_0/V)v(f)\}\,b_{-f}^+$. This is a $2\times2$ linear system mixing $b_f$ with $b_{-f}^+$, with matrix $M=\big[\begin{smallmatrix} T+\alpha & \alpha\\ -\alpha & -(T+\alpha)\end{smallmatrix}\big]$, $\alpha\equiv(N_0/V)v(f)$, the sign asymmetry coming from $b_{-f}^+$ being a creation operator. Its eigenvalues satisfy $E^2 = (T+\alpha)^2-\alpha^2 = T^2+2\alpha T$, giving the **Bogoliubov dispersion relation**

$$ E(f) = \sqrt{\,T(f)^2 + 2\,T(f)\,(N_0/V)\,v(f)\,}\;=\;\sqrt{\,|f|^2\,v(f)/(m v) + |f|^4/(4m^2)\,},\qquad v=V/N . $$

This single expression carries everything. For the *ideal* gas $\alpha=0$ and $E=T=f^2/2m$, the doomed spectrum. With the interaction on, the cross term $2T\alpha$ sits *under the square root*, and at small $f$, where $T\to0$ like $f^2$ while $\alpha\to(N_0/V)v(0)$ is constant, the cross term dominates: $E(f)\approx\sqrt{(f^2/m)(N_0/V)v(0)} = \sqrt{(N_0/V)v(0)/m}\;|f|$. The square root of a constant times $f^2$ is a constant times $|f|$ — the interaction has converted the quadratic dispersion into a *linear* one. That is a phonon, and it came out of the algebra. Its slope is a velocity $c=\sqrt{(N_0/V)v(0)/m}$, and I can check it really is sound: with $N_0\approx N$ the ground-state energy is $E=(N^2/2V)v(0)$, so $P=-\partial E/\partial V = (N^2/2V^2)v(0)$ and $\partial P/\partial\rho = v(0)/(mv)$, whence $c^2 = v(0)/(mv) = \partial P/\partial\rho$ — exactly the hydrodynamic sound speed. At the other end, where $v(f)\to0$ for any smooth potential, $E(f)\to\sqrt{T^2}=f^2/2m$, the bare molecule. One continuous curve from sound to free particle, with no separate roton branch forced on me in this dilute model.

The square root is only real if its radicand is non-negative, and that is not automatic. At small $f$ the sign is governed by $v(0)$: if $v(0)=\int\Phi(|q|)\,dq>0$ (net repulsion) the radicand is positive for all $f$ and $E(f)$ is a genuine excitation energy; if $v(0)<0$ the small-$f$ energy is imaginary, the $2\times2$ system has a runaway solution $e^{+|E|t/\hbar}$, and the condensate tears itself apart. The condition $v(0)>0$ for a stable spectrum is *identical* to the thermodynamic stability condition $\partial P/\partial\rho>0$ — the same statement that a fluid with negative compressibility collapses. The mathematics refusing a real spectrum and the physics refusing to hold the gas together coincide, which is what makes me trust the calculation; I restrict to $v(0)>0$ from here on.

I have the spectrum, but I have not yet exhibited the *operators* that are the genuinely independent excitations. The $2\times2$ mixing tells me the right normal mode is a combination of a destruction operator at $+f$ and a creation operator at $-f$, so I introduce the **canonical (Bogoliubov) transformation** $\xi_f = (b_f - L_f b_{-f}^+)/\sqrt{1-|L_f|^2}$ with a single number $L_f$ to fix. The mixing of a creation into an annihilation operator is not optional: the offending pair term changes particle number by two, so any number-conserving rotation among the $a_f$ leaves it untouched; only a transform that itself mixes $b$ and $b^+$ can rotate it away. Two demands pin $L_f$, and they are compatible. First, $\xi_f$ must stay an honest boson, $[\xi_f,\xi_{f'}^+]=\delta_{f,f'}$; computing it gives $(1-L_f^2)/(1-|L_f|^2)=1$, so the normalization $\sqrt{1-|L_f|^2}$ is *exactly* what keeps the transform canonical for any real $|L_f|<1$ — that is why it is there. Second, $L_f$ must be the value that kills the anomalous term, $L_f = (V/(N_0 v(f)))\{E(f)-T(f)-(N_0/V)v(f)\}$, after which the equations decouple into pure harmonic motion $i\hbar\,\partial_t\xi_f = E(f)\,\xi_f$ at the same $E(f)$. The weights this implies, $|L_f|^2 = [(N_0/V)v(f)/(E+T+(N_0/V)v(f))]^2$ and $1-|L_f|^2 = 2E/(E+T+(N_0/V)v(f))$, tell me what a quasiparticle *is*: at large $f$, $L_f\to0$ and $\xi_f\approx b_f$, a bare molecule; at small $f$, $E\to0$ and $L_f\to1$, a near-equal superposition of creating a $+f$ molecule and destroying a $-f$ one — a collective particle–hole object that cannot be identified with any single molecule. Equivalently $b_f=u_f\xi_f+v_f\xi_{-f}^+$ with $u_f^2-v_f^2=1$, $u_f^2=(T+\alpha+E)/2E$, $v_f^2=(T+\alpha-E)/2E$, both diverging like $1/|f|$ as $f\to0$ (strong particle–hole mixing in the phonon) and $u_f\to1, v_f\to0$ at large $f$.

Assembled in the new operators the Hamiltonian becomes, after using $\sum_{f\ne0}b_f^+b_f = N-N_0$ to collect the condensate self-energy and exchange term into $\tfrac12(N^2/V)\Phi_0$,

$$ H = H_0 + \sum_{f\ne0} E(f)\,n_f,\qquad n_f=\xi_f^+\xi_f, $$
$$ H_0 = \tfrac12(N^2/V)\Phi_0 + \tfrac12\sum_{f\ne0}\big[\,E(f)-T(f)-(N_0/V)v(f)\,\big],\qquad \Phi_0=v(0). $$

The total energy is a ground-state constant plus a sum of independent quanta of energy $E(f)$ and Bose occupation $n_f$: the weakly excited non-ideal Bose gas *is* a perfect gas of non-interacting elementary excitations. (Their mutual interaction would only appear in the cubic-and-higher $\vartheta$ terms I dropped, which are higher order in the depletion.) The $H_0$ beyond the classical $\tfrac12(N^2/V)\Phi_0$ is the zero-point energy of the quasiparticle vacuum. That vacuum is not the bare-molecule vacuum: even at $T=0$ a finite fraction of molecules sits at nonzero momentum, $(N-N_0)/N = (1/N)\,V/(2\pi\hbar)^3\int\{[E(f)+T(f)+(N_0/V)v(f)]/(2E(f))-1\}\,df>0$, which in the dilute contact limit is $(8/3\sqrt\pi)\sqrt{n a^3}$. This depletion is the self-consistency knob — the whole expansion assumed it $\ll1$, true precisely when $n a^3\ll1$.

Now close the loop on friction. I have a free Bose gas of quasiparticles of energy $E(f)$; let it drift at velocity $u$ relative to the condensate, so the equilibrium occupation is the boosted Bose function $\bar n_f = [\exp((E(f)-f\cdot u)/\Theta)-1]^{-1}$, the $-f\cdot u$ being the Galilean shift of a momentum-$f$ excitation. Occupations must be non-negative, forcing $E(f)>f\cdot u$ for every mode, whose tightest case (alignment) is $E(f)>|f||u|$, i.e.

$$ |u| < V_c = \min_{f\ne0}\frac{E(f)}{|f|}. $$

Below this drift no excitations are created spontaneously, so there is a stationary relative motion of condensate and quasiparticle gas with no friction — superfluidity, now Landau's criterion with *my derived* spectrum. The ratio $E(f)/|f|$ is continuous and positive, tends to $c>0$ as $f\to0$ (the phonon slope) and grows like $|f|/2m$ at large $f$, so its minimum is *strictly positive*: a genuine finite critical velocity, derived end to end from the molecular Hamiltonian. Had $E(f)$ stayed $f^2/2m$, that ratio would tend to $0$ and the minimum would be zero — exactly the ideal-gas failure I opened with. The interaction made the small-$f$ branch linear, and the linear branch is what makes $\min E/|f|$ positive. The phonon *is* the superfluidity.

```python
import numpy as np

def kinetic(k):
    "T(f) = f^2 / 2m, the bare molecule kinetic energy."
    return 0.5 * k * k

def excitation_energy(k, g, n0):
    "E(f) = sqrt( 2 T(f) (N0/V) v(f) + T(f)^2 ): the 2x2 eigenvalue."
    T = kinetic(k)
    return np.sqrt(T * (T + 2.0 * g * n0))          # = sqrt(T^2 + 2 T g n0)

def transform_weights(k, g, n0):
    "u^2 - v^2 = 1 (canonical). u->1,v->0 at large k; both ~1/k (phonon) at small k."
    T = kinetic(k); E = excitation_energy(k, g, n0)
    u2 = (T + g * n0 + E) / (2.0 * E)
    v2 = (T + g * n0 - E) / (2.0 * E)
    return u2, v2

def sound_speed(g, n0):
    "c = sqrt(g n0) = sqrt(dP/drho): the linear small-k slope of E(f)."
    return np.sqrt(g * n0)

def healing_length(g, n0):
    "xi = 1/sqrt(2 m g n0): phonon <-> free-particle crossover scale."
    return 1.0 / np.sqrt(2.0 * g * n0)

def critical_velocity(g, n0, kmax=50.0, npts=200000):
    "Landau: V_c = min_f E(f)/|f| > 0 because the small-f branch is linear."
    k = np.linspace(kmax / npts, kmax, npts)
    return np.min(excitation_energy(k, g, n0) / k)

def quantum_depletion(g, n0):
    "(N - N0)/N = (8/3 sqrt(pi)) sqrt(n a^3), a = g/4pi: ground-state depletion."
    a = g / (4.0 * np.pi)
    return (8.0 / (3.0 * np.sqrt(np.pi))) * np.sqrt(n0 * a ** 3)

if __name__ == "__main__":
    g, n0 = 1.0, 0.01                      # weak coupling, dilute (n0 a^3 << 1)
    print("c  =", sound_speed(g, n0))
    print("xi =", healing_length(g, n0))
    print("Vc =", critical_velocity(g, n0), " (equals c)")
    print("depletion =", quantum_depletion(g, n0))
    for k in (0.01, 0.1, 1.0, 10.0):
        E = excitation_energy(k, g, n0); u2, v2 = transform_weights(k, g, n0)
        print(f"k={k:6.2f} E={E:9.4f} ck={sound_speed(g,n0)*k:9.4f} "
              f"k^2/2={0.5*k*k:9.4f} u^2={u2:7.3f} v^2={v2:7.3f}")
```
