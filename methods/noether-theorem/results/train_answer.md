I present Noether's theorem as the general statement that ties a continuous symmetry of an action to a conservation law. The canonical name of this result is Noether's theorem; when the symmetry group is finite-dimensional, it is often called Noether's first theorem, while the companion statement for groups depending on arbitrary functions is Noether's second theorem. The theorem answers two questions at once: why energy, momentum, and angular momentum are conserved in ordinary mechanics, and why energy conservation loses its ordinary meaning in a generally-covariant theory such as general relativity. The key move is to work with the action itself, not with the equations of motion, because the action is defined for every field configuration and its variation carries a boundary term that becomes the conserved current.

I begin with an action integral I = ∫ f(x, u, ∂u/∂x, …) dx, where x = (x₁,…,x_n) are independent variables and u = (u₁,…,u_μ) are dependent variables. The physical configurations are the stationary points of I, which satisfy the Euler–Lagrange equations. The first step is a purely variational identity. If I vary u without imposing boundary conditions, integration by parts moves all derivatives off the variation and produces a total divergence:

Σᵢ ψᵢ δuᵢ = δf + Div A.

Here ψᵢ are the Lagrangian expressions, the left-hand sides of the Euler–Lagrange equations, and A is linear in δu and its derivatives. For a single integral with first derivatives this is the familiar central Lagrange equation, with A = −Σᵢ (∂f/∂uᵢ′) δuᵢ. The crucial point is that this identity holds off-shell: it is true for every field configuration, not only for solutions. The boundary term of the variation is already a divergence, and that divergence is the seed of every conservation law.

Next I bring in symmetry. A continuous symmetry, in Lie's sense, is a group of transformations that leaves the action invariant. Near the identity the transformation has the form yᵢ = xᵢ + Δxᵢ and vᵢ(y) = uᵢ + Δuᵢ, with Δx and Δu linear in the group parameters. The statement ΔI = 0 can be rewritten by pulling the transformed integral back to the original domain. The Jacobian of the infinitesimal change of variables contributes a term Div(f · Δx), and after subtracting the convective part of the field change one obtains the fixed-point variation

δ̄uᵢ = vᵢ(x) − uᵢ(x) = Δuᵢ − Σ_λ (∂uᵢ/∂x_λ) Δx_λ.

Lie's invariance equation then reads δ̄f + Div(f · Δx) = 0. Substituting the variational identity into this gives

Σᵢ ψᵢ δ̄uᵢ = Div B,   B = A − f · Δx.

This is the master identity. The left side contains the equations of motion contracted with the symmetry direction; the right side is a pure divergence. It holds for all configurations, so I have not yet used the Euler–Lagrange equations.

If the symmetry group is finite and depends on ρ parameters ε₁,…,ε_ρ, then B and δ̄u are linear in those parameters. Equating coefficients gives ρ separate relations:

Σᵢ ψᵢ δ̄uᵢ^(λ) = Div B^(λ),   λ = 1,…,ρ.

Now I impose the equations of motion, ψᵢ = 0. Each relation collapses to a conservation law:

Div B^(λ) = 0.

In one independent variable Div is simply d/dx, so each B^(λ) is a constant first integral. This is why a one-parameter symmetry gives one conserved quantity, a two-parameter symmetry gives two, and so on. The converse is obtained by reading the same calculation backward: given ρ divergence relations of this form, one can reconstruct the symmetry variation and recover invariance of the action.

The standard conservation laws drop out as special cases. In field-theory notation with Lagrangian density ℒ(φ, ∂_μφ) and spacetime coordinates x^μ, the boundary vector is A^μ = −(∂ℒ/∂(∂_μφ)) δ̄φ. For a translation x^μ → x^μ + ε^μ the field is carried along unchanged, so Δφ = 0 and δ̄φ = −ε^ν ∂_νφ. The current becomes

B^μ = ε^ν [ (∂ℒ/∂(∂_μφ)) ∂_νφ − δ^μ_ν ℒ ].

The coefficient of ε^ν is the canonical energy–momentum tensor T^μ_ν. The master identity reads −Σᵢ ψᵢ ∂_νφᵢ = ∂_μT^μ_ν, so on-shell ∂_μT^μ_ν = 0. The time-translation component ν = 0 gives the conserved energy E = ∫ T⁰₀ d³x = ∫ [ (∂ℒ/∂φ̇) φ̇ − ℒ ] d³x, which is the total Hamiltonian. The spatial components ν = i give the conserved momentum Pⁱ = ∫ T⁰ⁱ d³x = ∫ (∂ℒ/∂φ̇)(∂ⁱφ) d³x. Rotations and Lorentz boosts yield the angular-momentum tensor in the same way. Thus a single theorem produces energy, momentum, and angular momentum from translation and rotation invariance.

When the symmetry group is infinite-dimensional and depends on arbitrary functions rather than finitely many constants, the conclusion changes character. The symmetry variation δ̄u is now a linear differential expression in the arbitrary functions p^λ and their derivatives. The master identity still holds, but one must integrate by parts with respect to the derivatives of p^λ. This moves all derivatives onto the coefficients and produces ρ identities among the Lagrangian expressions and their derivatives:

Σ_{i,|α|≤σ} (−1)^{|α|} ∂_α(aᵢ^{λ,α} ψᵢ) = 0.

These are off-shell identities; they hold whether or not the equations of motion are satisfied. They are not independent conservation laws, because the apparent divergence relations for finite subgroups of the infinite group become linear combinations of these identities. In a generally-covariant theory such as general relativity, the energy law is therefore improper: its current is built from the field equations and identically divergenceless pieces, so the statement Div B = 0 carries no on-shell content beyond general covariance itself. This is exactly the characteristic feature that Hilbert asserted and that Noether's second theorem explains.

The following Python script is a small numerical illustration rather than a formal proof. It checks the conserved quantities in two simple systems. The first is a harmonic oscillator, whose Lagrangian L = ½ m q̇² − ½ k q² has no explicit time dependence, so Noether's theorem predicts conservation of energy. The second is a free particle, whose Lagrangian is invariant under spatial translations, so momentum is conserved. A crude discrete field update is also included to show how a lattice version of translation invariance leaves a discrete momentum approximately unchanged.

```python
import numpy as np

def energy(q, qdot, m=1.0, k=1.0):
    return 0.5 * m * qdot**2 + 0.5 * k * q**2

def simulate_harmonic(q0, p0, m=1.0, k=1.0, dt=0.01, steps=2000):
    q = np.empty(steps)
    p = np.empty(steps)
    q[0], p[0] = q0, p0
    for i in range(1, steps):
        p[i] = p[i-1] - k * q[i-1] * dt
        q[i] = q[i-1] + p[i] / m * dt
    return q, p

q, p = simulate_harmonic(1.0, 0.0)
E = energy(q, p)
print("Max |E - E0| (harmonic oscillator):", np.max(np.abs(E - E[0])))

def simulate_free(q0, p0, m=1.0, dt=0.01, steps=2000):
    q = np.empty(steps)
    p = np.empty(steps)
    q[0], p[0] = q0, p0
    for i in range(1, steps):
        p[i] = p[i-1]
        q[i] = q[i-1] + p[i] / m * dt
    return q, p

q_free, p_free = simulate_free(0.0, 2.0)
print("Momentum drift (free particle):", np.max(np.abs(p_free - p_free[0])))

N = 50
dx = 0.1
dt = 0.05
steps = 400
phi = np.sin(2 * np.pi * np.arange(N) * dx / N) + 0.1 * np.random.randn(N)
phi_prev = phi.copy()
for _ in range(steps):
    lap = np.roll(phi, 1) - 2 * phi + np.roll(phi, -1)
    phi_next = 2 * phi - phi_prev + (dt / dx)**2 * lap
    phi_prev, phi = phi, phi_next
pi = (phi - phi_prev) / dt
spatial_grad = (np.roll(phi, 1) - np.roll(phi, -1)) / (2 * dx)
P = np.sum(pi * spatial_grad) * dx
print("Discrete field momentum:", P)
```

In summary, Noether's theorem states that continuous symmetries of the action yield conservation laws. For a finite-dimensional symmetry group, the boundary term of the action's variation becomes a genuine conserved current, and the familiar conservation laws of energy, momentum, and angular momentum are recovered from translations and rotations. For an infinite-dimensional group depending on arbitrary functions, the same machinery produces identities among the field equations, and the associated conservation laws become improper. This distinction is what makes the energy law of general relativity qualitatively different from the energy law of ordinary mechanics.
