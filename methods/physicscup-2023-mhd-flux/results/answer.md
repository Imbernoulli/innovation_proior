# Magnets in a conducting liquid — flux freezing

## Problem

Two spherical permanent magnets of radius $R$ and magnetic dipole moment $m$ sit on a common axis, their dipole moments parallel to each other and to that axis, initially with centers a distance $3R$ apart. They are immersed in an incompressible liquid of effectively infinite conductivity (treat as superconductive, but with the pre-existing field *not* expelled), filling all space. The spheres are pulled apart along the axis to a separation $L\gg R$. Estimate the force $F$ needed to hold them there.

## Key idea: flux freezing (Alfvén's theorem)

In an infinitely conducting fluid the current density must stay finite, so Ohm's law forces $\mathbf{E}+\mathbf{v}\times\mathbf{B}=0$ in the fluid. The flux through any loop advected with the material then obeys

$$\frac{d\Phi}{dt}=-\oint(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot d\boldsymbol{\ell}=0,$$

so the magnetic flux through every co-moving loop is conserved: the field is **frozen into the fluid**, and field lines are carried with the material. A co-moving flux tube (zero flux through its sides) stays a flux tube and keeps its threading flux. Incompressibility ($\nabla\cdot\mathbf{v}=0$) additionally conserves each co-moving tube's **volume**.

## Derivation (energy method)

The aligned-axial geometry is cylindrically symmetric about the line of centers. Take a co-moving flux tube straddling the gap, with its base patches lying on the impenetrable sphere surfaces. The no-penetration condition keeps those material patches on the sphere boundaries, so the bridge between the surfaces stretches from its initial length $\sim R$ to $\sim L$ as the spheres separate.

- **Volume fixed (incompressible):** $V\sim A\,\ell=\text{const}\Rightarrow A\propto 1/\ell$. The tube thins as it lengthens; for a near-zone material volume $\sim R^3$, the final bridge area is $A(L)\sim R^3/L$.
- **Flux fixed (frozen-in):** $\Phi=B_\parallel A=\text{const}\Rightarrow B_\parallel\propto 1/A\propto \ell$, so at the end

$$B_\parallel=B_{\parallel,0}\,\frac{L}{R}.$$

Pulling the spheres apart *concentrates* the axial field by $L/R$ instead of diluting it.

- **Energy.** Density $B^2/2\mu_0$ rises by $(L/R)^2$ at fixed tube volume. The initial near-zone dipole energy is, by dimensions ($B\sim\mu_0 m/R^3$ over volume $\sim R^3$),

$$U_0\sim\frac{1}{\mu_0}\left(\frac{\mu_0 m}{R^3}\right)^2 R^3=\mu_0\frac{m^2}{R^3}.$$

For $L\gg R$ the stretched near-zone rope dominates the total energy, so

$$U(L)\sim U_0\left(\frac{L}{R}\right)^2=\mu_0\frac{m^2 L^2}{R^5}.$$

- **Force.** By cylindrical symmetry it is axial:

$$F_{\rm mag}=-\frac{dU}{dL},\qquad \boxed{\,F_{\rm hold}\sim\mu_0\dfrac{m^2 L}{R^5}\,}$$

The magnetic force is attractive because the stored energy grows with $L$; the external holding force points outward and *increases* linearly with separation. The field forms a stretched flux rope whose cross-section shrinks as it lengthens, so its field strength and tension grow with $L$.

## Cross-check (MHD stress tensor)

Enclose one sphere in a box with one face on the symmetry plane. In ideal MHD the stress is

$$\mathsf{T}=\frac{\mathbf{B}\mathbf{B}}{\mu_0}-\left(\frac{B^2}{2\mu_0}+(P-P_\infty)\right)\mathsf{I}.$$

Take the box large enough that, away from the stretched bridge, only the rapidly decaying dipolar tail reaches its surface; the leading contribution is the symmetry-plane face cutting through the bridge. Fluid equilibrium $\;(\mathbf{B}\cdot\nabla)\mathbf{B}/\mu_0=\nabla(B^2/2\mu_0+(P-P_\infty))\;$ with nearly straight axial field lines gives a tension/along-field term $\sim B^2/(\mu_0 L)$, negligible for transverse balance against pressure gradients $\sim B^2/(\mu_0 w)$ across a rope of width $w\ll L$. Thus $B^2/2\mu_0+(P-P_\infty)=\text{const}=0$ on the plane (evaluated far out, $B\to0,\ P\to P_\infty$). The bracket vanishes on the plane and the thermal pressure drops out:

$$\mathbf{F}=\int_{\rm plane}\frac{B^2}{\mu_0}\,\hat{\mathbf{n}}\,dS.$$

With $B\sim(\mu_0 m/R^3)(L/R)$ and plane cross-section $S\sim R^3/L$,

$$F\sim\frac{B^2}{\mu_0}S\sim\mu_0\frac{m^2 L}{R^5},$$

the same result by an independent route.

## Result

$$F_{\rm hold}\sim\mu_0\frac{m^2 L}{R^5},\qquad |F_{\rm mag}|\sim\mu_0\frac{m^2 L}{R^5}$$

with the magnetic force attractive and growing linearly with separation.
