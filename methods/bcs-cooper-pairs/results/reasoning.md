Let me get the obstacle straight before I touch any Hamiltonian, because the obstacle is the whole game here. The thing that separates a superconductor from a normal metal — the condensation energy, the energy I gain by going superconducting — is something like $N(0)(kT_c)^2$, which for a real metal is of order $10^{-8}$ electron volts per electron. And the total energy of the interacting electron–phonon system, even just in the normal state, I cannot compute to better than about an electron volt per electron. So the quantity I care about sits eight orders of magnitude below my error bar on the quantity I know how to compute. Any approach that writes down the full energy and waits for superconductivity to fall out as a small residue is dead on arrival. Whatever I do has to isolate the *one* piece of the electron correlations that is different between the normal and the superconducting state, and treat that piece — and only that piece — exactly. Everything else has to be the same in both phases so it cancels in the difference. That is the discipline.

So what do I actually know about that one piece? Landau's Fermi-liquid picture tells me the sharp Fermi surface survives the interactions: the low-lying excitations of the real normal metal are quasiparticles in one-to-one correspondence with a free Fermi gas, with a renormalized mass and a decay rate that dies at the surface. Good — that means I don't have to solve the metal. I can take the normal state as a filled sea of quasiparticles and ask only about their *residual* interaction. And the isotope effect, $T_c \propto M^{-1/2}$, tells me what that residual interaction is: the ions move, so the phonons are in it. Fröhlich predicted this, and the canonical transformation that removes the linear electron–phonon coupling — the one Bardeen and Pines pushed through with the Coulomb screening included — leaves behind a true electron–electron interaction from virtual phonon exchange. Its matrix element for scattering a pair near the Fermi surface goes like

$$ \frac{2\hbar\omega_\kappa |M_\kappa|^2}{(\epsilon_\mathbf{k}-\epsilon_{\mathbf{k}+\kappa})^2 - (\hbar\omega_\kappa)^2}, $$

and the sign of that is the thing that grabs me. When the electronic energy difference $|\epsilon_\mathbf{k}-\epsilon_{\mathbf{k}+\kappa}|$ is *less* than the phonon energy $\hbar\omega_\kappa$, the denominator is negative and the whole thing is *attractive*. Electrons that normally repel each other can, in a thin shell at the Fermi surface, pull on each other — provided this phonon attraction beats the screened Coulomb repulsion $\sim 4\pi e^2/(\kappa^2+\kappa_s^2)$ there. So in a shell of width $\hbar\omega$ around the Fermi surface I have a net attractive interaction between quasiparticles.

Now, the natural next move — and it's the move everyone has been making, including me in my earlier attempts — is to compute the electron's self-energy in this phonon field and look for an energy gap to open at the Fermi surface. I tried it variationally; Fröhlich tried it perturbatively. It fails, and I now think I understand *why* it has to fail. Almost all of that self-energy is already present in the normal state; it barely changes at the transition. Schafroth even showed you can't get the Meissner effect out of the Fröhlich Hamiltonian in any order of perturbation theory, and the careful diagrammatic treatment, correct to order $(m/M)^{1/2}$, gives no gap, no instability, nothing. The self-energy is the big common piece that cancels. So I should stop computing self-energies. The effect I want is in the *interaction between* the electrons, and — given that perturbation theory in the coupling keeps coming up empty — it is probably something perturbation theory *cannot* see.

Let me also pin down which electrons can possibly be involved, because that narrows everything. Pippard's coherence length is $\xi_0 \sim 10^{-4}$ cm. Run it through the uncertainty principle: a wavefunction coherent over $\Delta x \sim \xi_0$ is built from momenta spread by $\Delta p \sim \hbar/\xi_0$, and $\hbar/\xi_0$ is about $10^{-4} p_F$. So only states within $\sim 10^{-4} p_F$ of the Fermi surface — a thin shell, about $10^{-4}$ of all the electrons — can be doing the work, and they get their energy lowered by something like $kT_c$. Let me check that this is even self-consistent with the energetics: $10^{-4}$ of the electrons each lowered by $\sim kT_c$ gives a condensation energy of order $10^{-4}kT_c$ per electron. In density-of-states units, $kT_c \approx 10^{-3}$ eV and $N(0)kT_c \sim 10^{-4}$, so $N(0)(kT_c)^2 \sim 10^{-4}\cdot10^{-3} = 10^{-7}$ eV per electron — within an order of magnitude of the measured $\sim10^{-8}$, which is as close as these crude shell estimates get. The number hangs together. So the active reorganization is confined to a thin shell at the surface, and it is something happening in *momentum* space — not the electrons moving in real space, but the occupancy of momentum states near $k_F$ being rearranged. That has the flavor of what London called a solidification of the average momentum distribution; whether his guess is the right one I can't say yet, but it tells me where to point the calculation.

Before I try to organize all $10^{23}$ electrons, let me ask the smallest possible question that could expose the instability. Forget the many-body state. Take the filled Fermi sea, freeze it — don't let its electrons feel the interaction at all, their only role is to block the states below $k_F$ by the exclusion principle — and drop in just *two* extra electrons above the surface, let them interact with each other through this attractive $V$, and ask: is their lowest state above the sea, or below it? If two electrons added to a metal that is supposedly in its normal ground state would rather bind than sit at the Fermi energy, then the normal ground state was never the ground state. The instability would announce itself in the simplest two-body problem.

So: two electrons, total momentum zero (I'll justify the zero in a moment), opposite spins, occupying states $\mathbf{k}\uparrow, -\mathbf{k}\downarrow$ with $|\mathbf{k}| > k_F$ because the sea blocks everything below. Write the pair wavefunction as a superposition over the allowed relative momenta,

$$ |\psi\rangle = \sum_{k>k_F} a_\mathbf{k}\, |\mathbf{k}\uparrow,\, -\mathbf{k}\downarrow\rangle. $$

The Hamiltonian is the kinetic energy of the two electrons plus the attraction. Measuring single-particle energies $\epsilon_\mathbf{k}$ from the Fermi energy, the kinetic energy of this pair above $2\mathcal{E}_F$ is $2\epsilon_\mathbf{k}$ (each electron sits at $\epsilon_\mathbf{k} \ge 0$). I will call $E$ the pair energy measured relative to $2\mathcal{E}_F$. The Schrödinger equation, projected onto the state $\mathbf{k}$, is

$$ (E - 2\epsilon_\mathbf{k})\, a_\mathbf{k} = \sum_{k'>k_F} V_{\mathbf{k}\mathbf{k}'}\, a_{\mathbf{k}'}, $$

where $V_{\mathbf{k}\mathbf{k}'} = \langle \mathbf{k}\uparrow,-\mathbf{k}\downarrow|\, \hat V\, |\mathbf{k}'\uparrow,-\mathbf{k}'\downarrow\rangle$ is the amplitude to scatter the pair from relative momentum $\mathbf{k}'$ to $\mathbf{k}$. Now I want to know whether this relative energy $E$ can become negative, because $E<0$ means the absolute pair energy lies below $2\mathcal{E}_F$.

The full $V_{\mathbf{k}\mathbf{k}'}$ is messy. But I argued the interaction is attractive only in the shell $0 < \epsilon < \hbar\omega_D$, and it doesn't vary rapidly there, and the details of the band structure are going to wash out anyway because the answer is dominated by the shell, not by its shape. So I'll do the brutal, honest simplification: take the matrix element to be a constant attraction $-V$ (with $V>0$) for both $\mathbf{k}$ and $\mathbf{k}'$ in the shell, and zero outside.

$$ V_{\mathbf{k}\mathbf{k}'} = -V \quad \text{for } 0 < \epsilon_\mathbf{k},\,\epsilon_{\mathbf{k}'} < \hbar\omega_D, \qquad 0 \text{ otherwise}. $$

With that, the right-hand side becomes $-V\sum_{k'} a_{\mathbf{k}'}$, which is the *same constant* for every $\mathbf{k}$ — call it $-V C$ with $C = \sum_{k'>k_F} a_{\mathbf{k}'}$. So

$$ a_\mathbf{k} = \frac{V C}{2\epsilon_\mathbf{k} - E}. $$

Sum both sides over $\mathbf{k}$ in the shell. The $C$ on the left is just $\sum a_\mathbf{k}$, and it cancels:

$$ 1 = V \sum_{0 < \epsilon_\mathbf{k} < \hbar\omega_D} \frac{1}{2\epsilon_\mathbf{k} - E}. $$

That is the eigenvalue condition. It's clean enough to read the physics straight off it. Replace the sum by an integral over energy with the density of states per spin $N(\epsilon) \approx N(0)$, constant across the thin shell:

$$ 1 = N(0)\, V \int_0^{\hbar\omega_D} \frac{d\xi}{2\xi - E}. $$

I'm looking for a bound state, $E < 0$ — below the continuum bottom at $2\mathcal{E}_F$. Do the integral:

$$ \int_0^{\hbar\omega_D} \frac{d\xi}{2\xi - E} = \frac{1}{2}\ln\frac{2\hbar\omega_D - E}{-E}. $$

Now look at what happens as the attraction gets weak. The condition is

$$ 1 = \frac{N(0)V}{2}\ln\frac{2\hbar\omega_D - E}{-E}. $$

Here is the thing that stops me. As $E \to 0^-$, that logarithm *diverges*. $\ln(2\hbar\omega_D/|E|) \to +\infty$. So no matter how small $N(0)V$ is — no matter how feeble the attraction — there is always a value of $E < 0$ that satisfies the equation. The right-hand side runs over all positive values as $E$ sweeps from $0^-$ downward; it must cross $1$. A bound state *always* exists. There is no threshold strength. An arbitrarily weak attraction binds the pair.

I want to see *why* this is so different from binding two particles in free space, because in three dimensions a free pair needs a finite attraction strength to bind — a shallow well holds nothing. The difference is right there in the integral, and I should check it rather than assert it, because the whole conclusion rides on it. In free space the density of states near the bottom of the band goes like $\sqrt\epsilon$ and vanishes at $\epsilon=0$, so the analogous eigenvalue condition is $1 = gV\int_0^{\hbar\omega}\sqrt\epsilon\,d\epsilon/(2\epsilon - E)$. Does that integral blow up as $E\to0^-$ the way mine did? Let me put numbers on it (in shell units $\hbar\omega=1$): at $E=-10^{-1}$ the integral is $0.698$, at $E=-10^{-3}$ it is $0.965$, at $E=-10^{-6}$ it is $1.0000$, at $E=-10^{-9}$ still $1.0000$. It does *not* diverge — it saturates at the finite value $\int_0^1 \sqrt\epsilon\,d\epsilon/2\epsilon = \int_0^1 d\epsilon/(2\sqrt\epsilon) = 1$. So in free space there is a genuine threshold: you need $gV \ge 1$ before the right-hand side can reach unity, and a weak well binds nothing. Now contrast my Fermi-sea integral: with $N(0)V=0.1$ the bound-state energy is $|E|=2e^{-2/0.1}=4.1\times10^{-9}$, with $N(0)V=0.01$ it is $2.8\times10^{-87}$ — minuscule, but strictly positive for every $V>0$. The difference is not the well; it is that here the two electrons are not at the bottom of a band — they are sitting on top of a filled Fermi sea, and the Fermi sea hands them a *finite, nonzero density of states $N(0)$* at the very bottom of the energy window available to them, which is exactly what makes my integral $\int_0^{\hbar\omega} d\xi/(2\xi-E)$ log-divergent where the $\sqrt\epsilon$ one saturated. The exclusion principle has done me a favor: by forbidding the low-$k$ states, it forces the pair to live where the density of states is finite, and a finite density of states at the threshold is exactly what makes the integral log-divergent and the binding unconditional. The instability is a property of the Fermi *sea*, not of the two electrons by themselves.

Solve for the binding. With $|E| \ll \hbar\omega_D$ the log is $\ln(2\hbar\omega_D/|E|)$, so $\frac{2}{N(0)V} = \ln(2\hbar\omega_D/|E|)$, and

$$ |E| = 2\hbar\omega_D\, e^{-2/N(0)V}, \qquad E_{\text{pair}} = 2\mathcal{E}_F + E = 2\mathcal{E}_F - 2\hbar\omega_D\, e^{-2/N(0)V}. $$

Two things to sit with. First, the pair energy is genuinely below $2\mathcal{E}_F$ for every $V>0$ — the filled Fermi sea is *unstable* against forming this bound pair. The normal ground state is not the ground state. Second, look at how the binding depends on $V$: it's $e^{-2/N(0)V}$. That function has an essential singularity at $V = 0$ — every derivative in $V$ vanishes there. You can never reach $e^{-2/N(0)V}$ by expanding in powers of $V$; the whole effect is invisible to perturbation theory at every finite order. *That* is why Fröhlich's perturbation series and my own variational self-energy attempts never found a superconducting phase: the thing they were hunting is non-analytic in the coupling. The pair binding can't be reached by any number of orders of $V$. Now the earlier failures look less like bad luck and more like an obstruction in principle.

While I'm here — why total momentum zero, opposite spins? If I give the pair a finite center-of-mass momentum $\mathbf{q}$, then the two members live in $\mathbf{k}+\mathbf{q}/2$ and $-\mathbf{k}+\mathbf{q}/2$, and the constraint that *both* lie in the attractive shell shrinks the set of relative momenta $\mathbf{k}$ they can scatter through — fewer terms in the sum, weaker binding, and the binding is maximal at $\mathbf{q}=0$. For the spin, a momentum-independent attraction wants the even spatial channel; the Pauli principle then makes the spin wavefunction antisymmetric, so I pair $\mathbf{k}\uparrow$ with $-\mathbf{k}\downarrow$ in the singlet channel. The deep reason both choices win is the same: I want every pair to be able to scatter into the *same* large set of empty final states, so that all the scattering amplitudes can reinforce. Zero total momentum and time-reversed spin partners maximize that shared set.

So the two-body problem has told me the Fermi sea collapses. But it has also handed me a serious worry, and I have to face it before I celebrate. This bound pair is *huge*. The binding energy is of order $kT_c$, so by uncertainty its spatial extent is of order $\hbar v_F / kT_c \sim 10^{-4}$ cm — Pippard's coherence length, reassuringly. But if $\sim 10^{-4}$ of the electrons condense and each pair is $10^{-4}$ cm across, then the mean spacing between condensed electrons is $\sim 10^{-6}$ cm, and within the volume of a *single* pair sit the centers of roughly $(10^{-4}/10^{-6})^3 \sim 10^6$ other pairs. These are not little molecules I can Bose-condense the way Schafroth and Blatt and Butler wanted to. They overlap a million-fold. The very notion of "a bound pair wending its way through the metal" seems like it should be destroyed by the constant collisions of $10^6$ other pairs trying to occupy the same states. I can't just take a dilute gas of these pairs and condense it. The two-electron calculation is a signpost — *the normal state is unstable* — not a building block I can stack.

So I have to construct the real many-body ground state directly, as a state in which *all* the electrons near the surface are correlated this way at once, with the exclusion principle respected among all of them. Let me go back to the variational idea I keep circling: write the superconducting ground state as a coherent superposition of normal-state configurations,

$$ |\Psi\rangle = \sum_n a_n |\Phi_n\rangle, $$

and choose the coefficients to make the energy $E_0 = \langle\Psi|H|\Psi\rangle / \langle\Psi|\Psi\rangle$ as low as possible. The energy is a sum over matrix elements $a_n^* \langle\Phi_n|H|\Phi_m\rangle a_m$. If those off-diagonal matrix elements had random signs as I range over configurations, they'd cancel and I'd gain nothing — and indeed, for arbitrary configurations they *do* alternate in sign. I have $N$ coefficients $a_n$ but $N^2$ matrix elements to fix the signs of; I can't fix them all by choosing the $a_n$. So the naive superposition gains nothing. That's the wall.

The way around it: restrict which configurations I superpose so that the surviving matrix elements all have the *same* sign and add in phase. Build $|\Psi\rangle$ only from configurations in which, if a single-particle state $\mathbf{k}\uparrow$ is occupied, then its time-reversed mate $-\mathbf{k}\downarrow$ is also occupied — occupy states in *pairs* $(\mathbf{k}\uparrow, -\mathbf{k}\downarrow)$, never one without the other. Then the attractive interaction, which scatters a pair $(\mathbf{k}'\uparrow,-\mathbf{k}'\downarrow) \to (\mathbf{k}\uparrow,-\mathbf{k}\downarrow)$, connects these paired configurations through matrix elements of a single, definite sign — and they reinforce instead of cancelling. This is the same coherent-reinforcement logic the two-body problem ran on, now imposed as a constraint on the many-body state. And it is exactly London's momentum-space condensation: a correlated occupancy of pairs of momentum states. Momentum conservation in the lattice then says: to get the most reinforcing matrix elements, every pair should carry the *same* center-of-mass momentum, and the lowest energy is for that common momentum to be zero. So: zero-momentum, singlet, time-reversed pairs, all of them.

Let me make the pairs into objects. Define the pair operators

$$ b_\mathbf{k} = c_{-\mathbf{k}\downarrow}\, c_{\mathbf{k}\uparrow}, \qquad b_\mathbf{k}^\dagger = c_{\mathbf{k}\uparrow}^\dagger\, c_{-\mathbf{k}\downarrow}^\dagger, $$

so $b_\mathbf{k}^\dagger$ creates the pair. I should check their algebra explicitly, because everything will hinge on whether these behave as bosons, and the temptation to assume they do is exactly what sank the localized-pair picture. Start with the square: $b_\mathbf{k}^{\dagger\,2} = c_{\mathbf{k}\uparrow}^\dagger c_{-\mathbf{k}\downarrow}^\dagger c_{\mathbf{k}\uparrow}^\dagger c_{-\mathbf{k}\downarrow}^\dagger$, and since $c_{\mathbf{k}\uparrow}^{\dagger 2}=0$ I can anticommute the second $c_{\mathbf{k}\uparrow}^\dagger$ leftward past $c_{-\mathbf{k}\downarrow}^\dagger$ (one sign) to sit against the first, killing it: $b_\mathbf{k}^{\dagger\,2}=0$. A true boson has $(a^\dagger)^2\neq0$, so right there they are not bosons. Now the commutator on the same $\mathbf{k}$. Take it on the vacuum to be concrete: $b_\mathbf{k}b_\mathbf{k}^\dagger|0\rangle = c_{-\mathbf{k}\downarrow}c_{\mathbf{k}\uparrow}c_{\mathbf{k}\uparrow}^\dagger c_{-\mathbf{k}\downarrow}^\dagger|0\rangle$; using $c c^\dagger = 1 - c^\dagger c$ twice on the empty state gives $|0\rangle$, while $b_\mathbf{k}^\dagger b_\mathbf{k}|0\rangle=0$, so $[b_\mathbf{k},b_\mathbf{k}^\dagger]|0\rangle=|0\rangle$ — a boson would also give $+1$ here. But act instead on the filled pair state $b_\mathbf{k}^\dagger|0\rangle$: now $b_\mathbf{k}^\dagger$ can't act (square is zero) so $b_\mathbf{k}^\dagger b_\mathbf{k}$ returns the state, while $b_\mathbf{k}b_\mathbf{k}^\dagger$ kills it, and the commutator gives $-1\cdot b_\mathbf{k}^\dagger|0\rangle$. The eigenvalue flipped from $+1$ to $-1$ depending on occupancy — that is precisely the operator statement

$$ [b_\mathbf{k}, b_{\mathbf{k}'}^\dagger] = (1 - n_{\mathbf{k}\uparrow} - n_{-\mathbf{k}\downarrow})\,\delta_{\mathbf{k}\mathbf{k}'}, \qquad [b_\mathbf{k}, b_{\mathbf{k}'}] = 0, $$

with the boson value $+\delta_{\mathbf{k}\mathbf{k}'}$ recovered only when the pair is empty. So on different $\mathbf{k}$ they commute like bosons, but each one is hard-core: at most one pair per momentum state, the occupation-dependent $1-n-n$ being the fingerprint of the underlying fermions. This is the precise reason the localized-pair Bose-condensation picture is wrong — it treats as elementary bosons objects whose $(b^\dagger)^2=0$ forbids piling them up. I should not have expected them to be bosons — and this is the precise reason the localized-pair Bose-condensation picture is wrong. The fermionic inner structure is not a detail; it's encoded in $b_\mathbf{k}^2 = 0$, and it's going to give a gap not just against breaking a pair but against making a pair move with the wrong momentum, which is what will pin the whole condensate together.

Keep only the part of the Hamiltonian that scatters these zero-momentum pairs — the reduced Hamiltonian. Measuring from the Fermi sea,

$$ H_{\text{red}} = 2\sum_{k>k_F} \epsilon_\mathbf{k}\, b_\mathbf{k}^\dagger b_\mathbf{k} + 2\sum_{k<k_F} |\epsilon_\mathbf{k}|\, b_\mathbf{k} b_\mathbf{k}^\dagger - \sum_{\mathbf{k}\mathbf{k}'} V_{\mathbf{k}\mathbf{k}'}\, b_{\mathbf{k}'}^\dagger b_\mathbf{k}. $$

The first two terms are the cost of having a pair excited above the surface or a pair-hole below it; the last is the attractive scattering. I'm dropping all the finite-momentum scattering terms — they contribute little to the energy and can be handled as a perturbation later.

Now the ground state. I cannot say "pair state $\mathbf{k}$ is definitely occupied" or "definitely empty," because a definitely-occupied or definitely-empty configuration can't scatter — scattering needs an *amplitude* to be occupied and an amplitude to be empty, so the interaction term has something to act between. And here is the resolution of the overlap worry, which I'd been treating as a catastrophe. Precisely *because* $\sim 10^6$ pairs overlap and an enormous number of pair states feed into and out of any given one, the instantaneous occupancy of one pair state is essentially uncorrelated with the occupancy of the others at that instant — only the *average* occupancies are linked. The massive overlap, which looked like it would destroy the binding through collisions, is exactly what makes a *Hartree-like* product over pair states an excellent approximation. The disease is the cure. So write

$$ |\Psi\rangle = \prod_\mathbf{k} \left( u_\mathbf{k} + v_\mathbf{k}\, b_\mathbf{k}^\dagger \right) |0\rangle, $$

one independent factor per pair state: amplitude $v_\mathbf{k}$ that pair $\mathbf{k}$ is occupied, amplitude $u_\mathbf{k}$ that it is empty, with $u_\mathbf{k}^2 + v_\mathbf{k}^2 = 1$ for normalization, and write $h_\mathbf{k} \equiv v_\mathbf{k}^2$ for the occupation probability. This is a state of *indefinite* particle number — expanding the product gives pieces with different numbers of pairs — but for a macroscopic system the weights are sharply peaked about the mean $N$, and the errors from not fixing the number are of order $1/N$. I'll tolerate that; it's what lets each pair carry an occupied/empty amplitude and thereby scatter.

Now compute $W_0 = \langle\Psi|H_{\text{red}}|\Psi\rangle$ and minimize. The kinetic part: a pair above the surface costs $2\epsilon_\mathbf{k}$ and is present with probability $h_\mathbf{k}$; below the surface the energy is naturally written in terms of holes, and the algebra combines to (with $\epsilon_\mathbf{k}$ now signed about the surface)

$$ W_{\text{KE}} = 2\sum_{k>k_F} \epsilon_\mathbf{k} h_\mathbf{k} + 2\sum_{k<k_F} |\epsilon_\mathbf{k}| (1 - h_\mathbf{k}). $$

The interaction part: $-V_{\mathbf{k}\mathbf{k}'} b_{\mathbf{k}'}^\dagger b_\mathbf{k}$ takes its expectation value across the product state. The operator $b_{\mathbf{k}'}^\dagger$ needs pair $\mathbf{k}'$ empty (amplitude $u_{\mathbf{k}'} = \sqrt{1-h_{\mathbf{k}'}}$) and fills it (amplitude $v_{\mathbf{k}'} = \sqrt{h_{\mathbf{k}'}}$), so it brings a factor $\sqrt{h_{\mathbf{k}'}(1-h_{\mathbf{k}'})}$; likewise $b_\mathbf{k}$ brings $\sqrt{h_\mathbf{k}(1-h_\mathbf{k})}$. So

$$ W_I = -\sum_{\mathbf{k}\mathbf{k}'} V_{\mathbf{k}\mathbf{k}'}\, [h_\mathbf{k}(1-h_\mathbf{k})\, h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}, $$

and $W_0 = W_{\text{KE}} + W_I$. The interaction wants $h_\mathbf{k}(1-h_\mathbf{k})$ large, i.e. $h_\mathbf{k}$ near $1/2$ — it wants the occupancy *smeared* across the Fermi surface, because only a half-occupied pair state offers both the "empty" and the "occupied" amplitude that scattering feeds on. The kinetic energy wants the sharp step, $h_\mathbf{k}=1$ below and $0$ above. The competition between these is the whole physics: pay a little kinetic energy by smearing the step, gain more potential energy from the coherent scattering.

Minimize. Take $\partial W_0/\partial h_\mathbf{k} = 0$. Differentiating, the kinetic term gives $2\epsilon_\mathbf{k}$ and the interaction term gives, through $\frac{d}{dh_\mathbf{k}}[h_\mathbf{k}(1-h_\mathbf{k})]^{1/2} = \frac{1-2h_\mathbf{k}}{2[h_\mathbf{k}(1-h_\mathbf{k})]^{1/2}}$,

$$ 2\epsilon_\mathbf{k} = \frac{1-2h_\mathbf{k}}{[h_\mathbf{k}(1-h_\mathbf{k})]^{1/2}}\sum_{\mathbf{k}'} V_{\mathbf{k}\mathbf{k}'}\,[h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}. $$

Define the quantity that the right-hand sum keeps producing:

$$ \epsilon_0 \equiv \sum_{\mathbf{k}'} V_{\mathbf{k}\mathbf{k}'}\,[h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}, $$

where $V_{\mathbf{k}\mathbf{k}'}$ is now the positive attraction strength appearing in the reduced Hamiltonian through the explicit minus sign. For a constant attraction in the shell this is $\epsilon_0 = V\sum_{\mathbf{k}'}[h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}$. Then the minimization reads $2\epsilon_\mathbf{k}\,[h_\mathbf{k}(1-h_\mathbf{k})]^{1/2} = (1-2h_\mathbf{k})\,\epsilon_0$. This is a quadratic for $h_\mathbf{k}$; solving it,

$$ h_\mathbf{k} = \frac{1}{2}\left( 1 - \frac{\epsilon_\mathbf{k}}{\sqrt{\epsilon_\mathbf{k}^2 + \epsilon_0^2}} \right), \qquad [h_\mathbf{k}(1-h_\mathbf{k})]^{1/2} = \frac{\epsilon_0}{2\sqrt{\epsilon_\mathbf{k}^2 + \epsilon_0^2}}. $$

Look at $h_\mathbf{k}$: far below the surface ($\epsilon_\mathbf{k} \to -\infty$) it goes to $1$, far above it goes to $0$, and it passes smoothly through $1/2$ at the Fermi surface, smeared over a width $\epsilon_0$ in energy. The sharp step has been rounded over a scale $\epsilon_0$ — and $\epsilon_0$, whatever it turns out to be, is the energy scale of the whole effect.

Now close the loop. Substitute $[h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2} = \epsilon_0/(2\sqrt{\epsilon_{\mathbf{k}'}^2+\epsilon_0^2})$ back into the definition of $\epsilon_0$:

$$ \epsilon_0 = V\sum_{\mathbf{k}'} \frac{\epsilon_0}{2\sqrt{\epsilon_{\mathbf{k}'}^2+\epsilon_0^2}}. $$

The $\epsilon_0$ cancels — *provided it's nonzero* — and I'm left with the self-consistency condition

$$ \frac{1}{V} = \sum_\mathbf{k} \frac{1}{2\sqrt{\epsilon_\mathbf{k}^2 + \epsilon_0^2}}. $$

This is the gap equation. A nonzero $\epsilon_0$ solving it means the smeared state has lower energy than the sharp Fermi step — the normal state is unstable, exactly as the two-body problem warned, but now for the full many-body wavefunction. Turn the sum over the symmetric shell into an integral: the factor of two from $\epsilon$ and $-\epsilon$ cancels the explicit $1/2$ in the summand, leaving

$$ \frac{1}{N(0)V} = \int_0^{\hbar\omega} \frac{d\xi}{\sqrt{\xi^2 + \epsilon_0^2}} = \sinh^{-1}\!\frac{\hbar\omega}{\epsilon_0}. $$

So $\hbar\omega/\epsilon_0 = \sinh[1/N(0)V]$, and

$$ \epsilon_0 = \frac{\hbar\omega}{\sinh[1/N(0)V]} \;\xrightarrow{\;N(0)V \ll 1\;}\; 2\hbar\omega\, e^{-1/N(0)V}. $$

There is the gap, and there is the same non-analytic $e^{-1/N(0)V}$ — an essential singularity at $V=0$, beyond the reach of any perturbation series, which is why the honest perturbative theories could never have found it.

One factor demands an explanation: the two-body binding came out as $e^{-2/N(0)V}$, but the many-body gap is $e^{-1/N(0)V}$ — half the exponent, so parametrically *larger*. Why? Trace it to the integrals. The single added pair sat on a *rigid* sea, so its energy denominator was the bare $2\xi - E$, linear in $\xi$, giving $\int d\xi/(2\xi-E) = \frac12\ln(\cdots)$ — that factor $\frac12$ became the $2$ in the exponent. In the self-consistent many-body state every electron is *itself* paired, so the relevant single-particle energy is not the bare $\xi$ but $\sqrt{\xi^2+\epsilon_0^2}$, and the integral $\int d\xi/\sqrt{\xi^2+\epsilon_0^2}$ is $\sinh^{-1}$, with no leading $\frac12$ — exponent $1$. Let me make sure that halving of the exponent actually matters and isn't a wash. At $N(0)V=0.3$ the single-pair binding is $|E|=2e^{-2/0.3}=2.5\times10^{-3}$ while the gap is $\epsilon_0=2e^{-1/0.3}=7.1\times10^{-2}$, a ratio of $\sim28$; push to $N(0)V=0.15$ and the ratio is $\sim790$. So the collective gap doesn't just edge out the lone-pair binding — it runs away from it as the coupling weakens, by the factor $e^{+1/N(0)V}$. The collective state binds far more strongly than a lone pair, because each partner is supported by its own pairing rather than sitting on an inert sea.

Now read off the excitations, because the gap I've been calling $\epsilon_0$ should show up as a real gap in the spectrum. Take the ground state and break one pair — promote it to a genuine single-particle (quasiparticle) excitation. Computing the energy cost, the excitation energy of a quasiparticle of normal energy $\epsilon_\mathbf{k}$ is

$$ E_\mathbf{k} = \sqrt{\epsilon_\mathbf{k}^2 + \epsilon_0^2}. $$

Even right at the Fermi surface, $\epsilon_\mathbf{k} \to 0$, this does *not* go to zero — it bottoms out at $E_\mathbf{k} = \epsilon_0$. In the normal metal the single-particle excitation energy slides continuously to zero at the surface; here there is a floor. You cannot make a single-particle excitation for less than $\epsilon_0$, so the minimum energy to break a pair and produce two excitations is $2\epsilon_0$: an energy *gap* of width $2\epsilon_0$ opens in the single-particle spectrum, centered on the Fermi energy. The density of single-particle states piles up at the gap edge — $dN/dE = N(0)\, E/\sqrt{E^2-\epsilon_0^2}$, singular at $E=\epsilon_0$ and zero inside the gap. That gap is the thing the exponential specific heat $\exp(-T_0/T)$ has been telling me about all along. And because $b_\mathbf{k}^2=0$ made these pairs non-bosonic, there is also a gap against making a pair move with a momentum different from the common one — that is what locks the condensate's phase rigid over macroscopic distances and gives the London rigidity, the persistent currents, the Meissner effect.

Let me also pin the condensation energy, because it has to come out tiny and isotope-dependent or the whole thing is wrong. Evaluating $W_0$ at the optimal $h_\mathbf{k}$ and subtracting the normal-state energy, in the weak-coupling limit

$$ W_0 = -2N(0)(\hbar\omega)^2\, e^{-2/N(0)V}. $$

Two things to check here. First the magnitude, since the eight-order-of-magnitude smallness was my whole opening worry. The prefactor $2N(0)(\hbar\omega)^2$ is the dimensionally-natural scale and is *not* small — it is set by the phonon energy. What makes $W_0$ tiny is the factor $e^{-2/N(0)V}$. With $N(0)V=0.3$ that factor is $e^{-2/0.3}=e^{-6.67}=1.3\times10^{-3}$; for a more realistic weak coupling $N(0)V=0.2$ it is $e^{-10}=4.5\times10^{-5}$. So the exponential alone suppresses the natural scale by three-to-five orders of magnitude, and a slightly weaker coupling pushes it further — the smallness is *produced* by the non-analytic exponential, not put in by hand. Squaring it into $T_c\propto e^{-1/N(0)V}$ and the condensation energy $\propto e^{-2/N(0)V}$ is exactly the kind of explosive sensitivity that turns an eV phonon scale into a meV gap. Second, the isotope effect: $W_0\propto(\hbar\omega)^2\propto M^{-1}$, and identifying $W_0\sim N(0)(kT_c)^2$ forces $kT_c\propto\hbar\omega\propto M^{-1/2}$, which is the measured $T_c M^{1/2}=\text{const}$. The phonon mechanism didn't have to be inserted at the end; it is sitting in the answer through the single appearance of $\hbar\omega$.

Now extend to finite temperature, because I need $T_c$ and a second-order transition. At temperature $T$ some states are occupied by quasiparticle excitations with the Fermi probability $f_\mathbf{k}$; those states are blocked, so they're unavailable for the pair scattering and the available phase space shrinks. Re-minimize the free energy $F = \langle H_{\text{red}}\rangle - TS$ over both the pairing amplitudes and the excitation occupations $f_\mathbf{k}$. The quasiparticle energy keeps its form $E_\mathbf{k} = \sqrt{\epsilon_\mathbf{k}^2+\epsilon_0^2}$ with a now temperature-dependent $\epsilon_0$, the excitations fill according to $f_\mathbf{k} = 1/(e^{\beta E_\mathbf{k}}+1)$, and each occupied excitation removes its state from the pairing sum. The effect is to weight the gap equation by the factor $(1-2f) = \tanh(\tfrac12\beta E)$:

$$ \frac{1}{N(0)V} = \int_0^{\hbar\omega} \frac{d\xi}{\sqrt{\xi^2+\epsilon_0^2}}\,\tanh\!\left[\tfrac{1}{2}\beta\sqrt{\xi^2+\epsilon_0^2}\right]. $$

At $T=0$, $\tanh \to 1$ and this is the zero-temperature gap equation again. As $T$ rises the $\tanh$ erodes the right-hand side, so the $\epsilon_0$ that solves it shrinks, and at a critical temperature $\epsilon_0$ reaches zero — that is $T_c$, and the transition is continuous (second order) because $\epsilon_0(T)$ goes smoothly to zero, with no latent heat. Set $\epsilon_0 = 0$ to find $T_c$:

$$ \frac{1}{N(0)V} = \int_0^{\hbar\omega} \frac{d\xi}{\xi}\,\tanh\!\left(\tfrac{1}{2}\beta_c\, \xi\right), $$

and doing this integral in the weak-coupling limit $kT_c \ll \hbar\omega$ gives

$$ kT_c = 1.14\,\hbar\omega\, e^{-1/N(0)V}. $$

Same exponential, so $T_c$ inherits the isotope dependence too. And now a parameter-free prediction falls out by comparing the two: the zero-temperature gap $2\epsilon_0(0) = 4\hbar\omega\,e^{-1/N(0)V}$ against $kT_c = 1.14\,\hbar\omega\, e^{-1/N(0)V}$ — the $\hbar\omega$ and the exponential both cancel, leaving

$$ \frac{2\epsilon_0(0)}{kT_c} = \frac{4}{1.14} \approx 3.5. $$

A pure number, independent of $\hbar\omega$, of $V$, of the material — every weak-coupling superconductor should show the same ratio of gap to transition temperature. The arbitrary microscopic parameters have dropped out of a measurable dimensionless quantity. That is the kind of prediction I'd want to put against tin and vanadium.

But I got that $3.5$ from the *weak-coupling asymptotics* of two different equations — $4\hbar\omega\,e^{-1/N(0)V}$ for the gap, $1.14\,\hbar\omega\,e^{-1/N(0)V}$ for $T_c$ — and I'd like to know whether the cancellation is real or an artifact of replacing both equations by their leading exponentials. The honest check is to solve the *un-approximated* equations at a finite coupling and divide. So I take $N(0)V=0.3$ (not especially small) and solve, numerically, the full $1/N(0)V=\sinh^{-1}(\hbar\omega/\epsilon_0)$ for $\epsilon_0$ and the full $\tanh$-integral $1/N(0)V=\int_0^{\hbar\omega}(d\xi/\xi)\tanh(\xi/2kT_c)$ for $kT_c$, with no weak-coupling substitution anywhere. The root-finder returns $\epsilon_0=0.0714$ (the closed form $1/\sinh(1/0.3)=0.0714$ agrees to four figures, so my $\sinh^{-1}$ algebra was right) and $kT_c=0.0404$. Their ratio is $2\epsilon_0/kT_c=2(0.0714)/0.0404=3.53$. So the $3.5$ survives being computed from the exact equations at finite coupling — it is not an accident of the asymptotic forms. I'll also note the exact $kT_c=0.0404$ sits just below the weak-coupling estimate $1.14\,e^{-1/0.3}=0.0407$, the small gap being the difference between $\sinh^{-1}$ and its exponential approximation; nothing surprising, and it tells me the $\sim3.5$ ratio is robust, not knife-edge.

Let me say the causal chain back to myself in one breath. The condensation energy is eight orders below my error bar, so I isolate the residual quasiparticle interaction and find, from the phonon mechanism the isotope effect points to, that it is *attractive* in a thin shell at the Fermi surface. Adding two electrons to a frozen sea, I find that this attraction — however weak — always binds them below $2\mathcal{E}_F$, because the sea's finite density of states $N(0)$ makes the pairing integral log-divergent; the binding $e^{-2/N(0)V}$ is non-analytic in $V$, which is why perturbation theory never saw it. So the normal state is unstable. The pairs overlap a million-fold, which kills the localized-Bose picture but is precisely what licenses a Hartree-like product over pair states; minimizing its energy smears the Fermi step over a width $\epsilon_0$ set self-consistently by the gap equation $1/N(0)V = \int d\xi/\sqrt{\xi^2+\epsilon_0^2}$, giving $\epsilon_0 \approx 2\hbar\omega\,e^{-1/N(0)V}$. That $\epsilon_0$ is a real gap of width $2\epsilon_0$ in the single-particle spectrum (the exponential specific heat), the condensation energy scales as $(\hbar\omega)^2$ and so matches both the tiny scale and the isotope dependence through $T_c$, the non-bosonic pair structure locks the phase rigid (Meissner, persistent currents), and at finite $T$ the same equation with a $\tanh$ gives a second-order transition at $kT_c = 1.14\,\hbar\omega\,e^{-1/N(0)V}$ with the parameter-free ratio $2\epsilon_0(0)/kT_c \approx 3.5$. Every one of the five facts I started from is now an output.

Here is the calculation laid out so I can run all three checks against their closed forms in one place — the two-electron instability, the self-consistent gap, and $T_c$ — and confirm by machine the numbers I've been carrying by hand.

```python
import numpy as np
from scipy import integrate, optimize

# Energies in units of the phonon (Debye) cutoff:  hbar*omega_D = 1.
N0V = 0.3   # dimensionless coupling N(0)*V  (weak coupling: < 1)

# --- two-electron instability: one pair added above a frozen Fermi sea ---
# Eigenvalue condition  1 = N0V * \int_0^1 dxi / (2*xi - E_rel), with E_rel < 0.
def pair_condition(E_rel):
    integ, _ = integrate.quad(lambda xi: 1.0/(2*xi - E_rel), 0.0, 1.0)
    return N0V*integ - 1.0

E_rel = optimize.brentq(pair_condition, -10.0, -1e-12)      # relative to 2*E_F
print("pair energy relative to 2E_F:", E_rel)               # < 0 for any N0V > 0
print("  weak-coupling form -2*exp(-2/N0V):", -2*np.exp(-2/N0V))

# --- many-body gap equation: self-consistent order parameter ---
# 1/(N0V) = \int_0^1 dxi / sqrt(xi^2 + D^2)  = arcsinh(1/D)  ->  D = 1/sinh(1/N0V)
def gap_condition(D):
    integ, _ = integrate.quad(lambda xi: 1.0/np.sqrt(xi**2 + D**2), 0.0, 1.0)
    return integ - 1.0/N0V

D = optimize.brentq(gap_condition, 1e-12, 10.0)             # the half-gap eps_0
print("gap eps_0:", D, " closed form 1/sinh(1/N0V):", 1.0/np.sinh(1.0/N0V))
print("  weak-coupling form 2*exp(-1/N0V):", 2*np.exp(-1/N0V))

# --- finite T: 1/(N0V) = \int_0^1 dxi/sqrt(xi^2+D^2) * tanh(sqrt(xi^2+D^2)/(2*kT)) ---
def gap_T(D, kT):
    integ, _ = integrate.quad(
        lambda xi: np.tanh(np.sqrt(xi**2+D**2)/(2*kT))/np.sqrt(xi**2+D**2), 0.0, 1.0)
    return integ - 1.0/N0V

# T_c: where D -> 0.  1/(N0V) = \int_0^1 (dxi/xi) tanh(xi/(2 kTc)) -> kTc ~ 1.14 exp(-1/N0V)
def Tc_condition(kT):
    integ, _ = integrate.quad(lambda xi: np.tanh(xi/(2*kT))/xi, 1e-9, 1.0)
    return integ - 1.0/N0V

kTc = optimize.brentq(Tc_condition, 1e-6, 1.0)
print("kTc:", kTc, " weak-coupling 1.14*exp(-1/N0V):", 1.14*np.exp(-1/N0V))
print("gap-to-Tc ratio 2*eps_0(0)/kTc:", 2*D/kTc, " (weak-coupling universal ~3.5)")
```

Running it: the pair energy comes out $-2.55\times10^{-3}$ against the weak-coupling $-2e^{-2/N(0)V}=-2.55\times10^{-3}$ — negative, so the sea is unstable, and matched. The gap is $\epsilon_0=0.0714$, equal to $1/\sinh(1/N(0)V)=0.0714$ to four figures and close to $2e^{-1/N(0)V}=0.0713$. And $kT_c=0.0404$ against $1.14\,e^{-1/N(0)V}=0.0407$, with the ratio printing $3.53$. Every hand-derived form lands on its numerical solution, and the dimensionless gap-to-$T_c$ ratio is the $\approx3.5$ I'll take to the experiments. Nothing in the chain needed an adjustable parameter to make a number come out right.
