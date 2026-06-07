# Differentiable simulation for additive-manufacturing process design

## Problem

In metal AM, the part's thermal history governs microstructure, residual stress, geometric accuracy, and melt-pool depth, and that history is set by the process parameters — laser power, beam radius, material properties, and especially a *time-series* laser power P(t) over tens of thousands of explicit time steps. Designing those parameters to hit a target thermal/melt-pool behavior is an inverse problem over a very high-dimensional space. The obstacle is gradient cost: finite-difference or derivative-free optimization needs one full (minutes-long) simulation per parameter, so its cost scales with the parameter count and becomes infeasible for thousands of time-series degrees of freedom.

## Key idea

Make the thermal **simulator itself differentiable** in Taichi. Implement the explicit time-stepping finite-element thermal solver inside a reverse-mode automatic-differentiation framework. Because the loss is a single scalar and the parameters number in the thousands (R^P → R), reverse-mode AD returns the full gradient ∂loss/∂(all parameters) in essentially one backward pass, at a cost independent of P. Backpropagating through the explicit recurrence is exactly the **discrete adjoint** of the solver, obtained automatically for any loss and any parameterization — no hand-derived continuous adjoint. The only memory cost is the stored state history, O(N) in the number of steps N, reducible to O(√N) by segment-wise checkpointing (recompute within a segment from a stored boundary state on the backward pass).

Two design rules make the gradients usable:
- **Freeze the discrete machinery.** Element birth, surface birth, and toolpath are precomputed and held fixed behind boolean masks (`if birth ≤ t·Δt`); the gradient flows only through the *continuous* power and material properties, never across a material-deposition discontinuity.
- **Smooth every step-like feature.** Melt-pool depth defined as "deepest molten node" is piecewise-constant → zero gradient. Replace it with a continuous surrogate: interpolate the temperature directly under the beam at four height levels from nine fixed-by-toolpath neighboring nodes, then locate the liquidus-isotherm crossing by pairwise-linear interpolation → a smooth depth with nonzero gradient away from degenerate equal-temperature pairs.

## Method

Thermal physics. Transient heat conduction ρ c_p ∂T/∂t − ∇·(k∇T) − s = 0 with Dirichlet, convection q = −h(T − T_amb), radiation q = −εσ(T⁴ − T_amb⁴), and a moving laser flux. Discretize over Hex8 elements (8 Gauss points, shape functions N, derivatives B), assemble the capacitance [M] and conduction [K] and the flux/convection/radiation vectors from active-element contributions, and integrate explicitly with a **lumped** (diagonal) capacitance so M⁻¹ is an elementwise divide and no linear solve is needed:

  {T^{n+1}} = {T^n} + Δt [M]⁻¹ [ {R_G} − {R_F} − {R_C} − {R_R} − [K]{T^n} ].

Laser. Moving surface Gaussian,

  q_s(x,t) = 3 Q P(t) on(t) / (π r_b²) · exp(−3||x − x_L(t)||²/r_b²),

where x_L(t) is the beam center and r_b is the beam radius. Under the outward-flux convention, incoming laser heat has R_F < 0, so the update's −R_F term is assembled as a positive heat-input contribution. Latent heat is folded into a piecewise c_p across the solidus–liquidus band as L/(T_liq − T_solid); the branch is not smooth at the thresholds, so it is treated as a fixed material law rather than a design variable.

Control parameterization. A tiny fully-connected network maps (normalized) time → P(t) in [0,1] (two hidden layers of 50, tanh), giving a compact, smooth schedule whose weights are the optimized variables; for static calibration the leaves are instead the scalar material/process parameters.

Optimization. Squared-error loss, averaged as MSE when reported, between achieved and target response (full thermal history, melt-pool depth, or a partial top-layer observation), minimized with **Adam** (lr 1e-2, β₁=0.9, β₂=0.999, ε=1e-7). The host AD framework (an imperative, flexibly-indexed, atomic-accumulating differentiable-programming system — chosen over array DL libraries because FE assembly needs scatter/atomics, not dense ops) records a lightweight tape of the forward kernels and replays their adjoints in reverse.

Three tasks: (i) infer static material/process scalars from partially observed thermal data; (ii) match an arbitrary target thermal history by shaping P(t); (iii) stabilize melt-pool depth through the build by shaping P(t).

## Code

```python
import numpy as np
import taichi as ti

# scales / properties (stainless steel)
density = 0.03; cp_init = 0.5; cond_init = 0.01
Qin_init = 250.0; r_beam_init = 1.0; h_conv_init = 0.00005; h_rad = 0.2
ambient_init = 300.0; max_temp = 2000.0
solidus, liquidus = 1533.15, 1609.15
latent_cp = 272.0 / (liquidus - solidus)     # L/(T_liq - T_sol) folded into c_p
pi = 3.141592653589793

n_input, n_hidden_1, n_hidden_2, n_hidden_3 = 3, 50, 50, 1
learning_rate, beta_1, beta_2, epsilon = 1e-2, 0.9, 0.999, 1e-7
loss_kind = "temperature_history"   # or "melt_pool" for the smooth-depth objective
n_depth_levels = 4

ti.init()
# every state/parameter is a differentiable field; reverse-mode AD fills .grad
temperature = ti.field(float, (steps, nn), needs_grad=True)   # full history -> O(N) memory (checkpointable)
m_vec = ti.field(float, nn, needs_grad=True)                  # lumped capacitance (diagonal)
rhs   = ti.field(float, nn, needs_grad=True)
ambient = ti.field(float, (), needs_grad=True)
cp   = ti.Vector.field(8, float, nel, needs_grad=True)
cond = ti.Vector.field(8, float, nel, needs_grad=True)
Qin  = ti.field(float, (), needs_grad=True)
r_beam = ti.field(float, (), needs_grad=True)
h_conv = ti.field(float, (), needs_grad=True)
loss = ti.field(float, (), needs_grad=True)
weight1 = ti.field(float, (n_hidden_1, n_input),  needs_grad=True); bias1 = ti.field(float, n_hidden_1, needs_grad=True)
weight2 = ti.field(float, (n_hidden_2, n_hidden_1), needs_grad=True); bias2 = ti.field(float, n_hidden_2, needs_grad=True)
weight3 = ti.field(float, (n_hidden_3, n_hidden_2), needs_grad=True); bias3 = ti.field(float, n_hidden_3, needs_grad=True)
weight1_m = ti.field(float, (n_hidden_1, n_input));  weight1_v = ti.field(float, (n_hidden_1, n_input))
weight2_m = ti.field(float, (n_hidden_2, n_hidden_1)); weight2_v = ti.field(float, (n_hidden_2, n_hidden_1))
weight3_m = ti.field(float, (n_hidden_3, n_hidden_2)); weight3_v = ti.field(float, (n_hidden_3, n_hidden_2))
bias1_m = ti.field(float, n_hidden_1); bias1_v = ti.field(float, n_hidden_1)
bias2_m = ti.field(float, n_hidden_2); bias2_v = ti.field(float, n_hidden_2)
bias3_m = ti.field(float, n_hidden_3); bias3_v = ti.field(float, n_hidden_3)
output1 = ti.field(float, (steps, n_hidden_1), needs_grad=True)
output2 = ti.field(float, (steps, n_hidden_2), needs_grad=True)
output3 = ti.field(float, (steps, n_hidden_3), needs_grad=True)   # normalized power P(t)
melt_depth = ti.field(float, steps, needs_grad=True)              # smooth depth proxy
target = ti.field(float, (steps, nn))
target_depth = ti.field(float, steps)
beam_interp_node = ti.field(int, (steps, n_depth_levels, 9))
beam_interp_weight = ti.field(float, (steps, n_depth_levels, 9))
depth_level = ti.field(float, n_depth_levels)

# control network: (normalized) time -> P(t) in [0,1]
@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden_1):
        act = 0.0
        for j in ti.static(range(n_input)):
            act += weight1[i, j] * (((t / steps) - 0.5))**j
        output1[t, i] = ti.tanh(act + bias1[i])
@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_hidden_2):
        act = 0.0
        for j in ti.static(range(n_hidden_1)):
            act += weight2[i, j] * output1[t, j]
        output2[t, i] = ti.tanh(act + bias2[i])
@ti.kernel
def nn3(t: ti.i32):
    for i in range(1):
        act = 0.0
        for j in ti.static(range(n_hidden_2)):
            act += weight3[i, j] * output2[t, j]
        output3[t, i] = (ti.tanh(act + bias3[i]) + 1) / 2.0

# temperature-dependent c_p with latent-heat band (forward-only branch)
@ti.func
def calc_cp(t, el_id, solidus, liquidus, latent_cp):
    cp_el = ti.Vector.zero(float, 8)
    for ip in ti.static(range(8)):
        theta_ip = 0.0; N = Nip_element[ip]
        for i in ti.static(range(8)):
            theta_ip += N[i] * temperature[t - 1, elements_node_ids[el_id][i]]
        cp_el[ip] = cp_init
        if theta_ip > solidus and theta_ip < liquidus:
            cp_el[ip] += latent_cp
    return cp_el
@ti.kernel
def update_matprop(t: ti.i32):
    for el_id in range(nel):
        cp[el_id] = calc_cp(t, el_id, solidus, liquidus, latent_cp)

# lumped capacitance over ACTIVE elements
@ti.func
def calc_mass(t, el_id, density):
    mass = ti.Matrix.zero(float, 8, 8); lump_mass = ti.Vector.zero(float, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        N = Nip_element[ip]; B = Bip_element[ip]
        detJac = (B @ nodes_pos).determinant()
        mass += density * cp[el_id][ip] * detJac * N @ N.transpose()
    for i in ti.static(range(8)):
        for j in ti.static(range(8)):
            lump_mass[i] += mass[i, j]            # row-sum lumping -> diagonal
    return lump_mass
@ti.kernel
def update_mvec(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:          # frozen discrete mask
            lump_mass = calc_mass(t, el_id, density)
            for i in ti.static(range(8)):
                m_vec[elements_node_ids[el_id][i]] += lump_mass[i]   # atomic accumulation

# conduction: accumulate -K T^n into rhs
@ti.func
def calc_stiffness(el_id):
    stiffness = ti.Matrix.zero(float, 8, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        B = Bip_element[ip]; Jac = B @ nodes_pos
        gradN = Jac.inverse() @ B
        stiffness += cond[el_id][ip] * Jac.determinant() * gradN.transpose() @ gradN
    return stiffness
@ti.kernel
def update_stiffness(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            stiffness = calc_stiffness(el_id)
            temperature_nodes = ti.Matrix.rows([[temperature[t-1, elements_node_ids[el_id][k]]] for k in range(8)])
            stiff_temp = stiffness @ temperature_nodes
            for i in ti.static(range(8)):
                rhs[elements_node_ids[el_id][i]] -= stiff_temp[i]

# moving Gaussian surface heat source: where P(t) (output3) enters
@ti.kernel
def update_fluxes_m(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            for sur_id in ti.static(range(6)):
                if surface_birth[el_id][sur_id, 0] <= t*dt and surface_birth[el_id][sur_id, 1] > t*dt:
                    for ip in ti.static(range(4)):
                        N = Nip_surface[ip]
                        ip_pos_x = N.dot(ti.Vector([
                            node_position[element_surface_ids[el_id][sur_id, 0]][0],
                            node_position[element_surface_ids[el_id][sur_id, 1]][0],
                            node_position[element_surface_ids[el_id][sur_id, 2]][0],
                            node_position[element_surface_ids[el_id][sur_id, 3]][0]]))
                        ip_pos_y = N.dot(ti.Vector([
                            node_position[element_surface_ids[el_id][sur_id, 0]][1],
                            node_position[element_surface_ids[el_id][sur_id, 1]][1],
                            node_position[element_surface_ids[el_id][sur_id, 2]][1],
                            node_position[element_surface_ids[el_id][sur_id, 3]][1]]))
                        ip_pos_z = N.dot(ti.Vector([
                            node_position[element_surface_ids[el_id][sur_id, 0]][2],
                            node_position[element_surface_ids[el_id][sur_id, 1]][2],
                            node_position[element_surface_ids[el_id][sur_id, 2]][2],
                            node_position[element_surface_ids[el_id][sur_id, 3]][2]]))
                        r2 = (ip_pos_x - laser_loc[t][0])**2 + (ip_pos_y - laser_loc[t][1])**2 + (ip_pos_z - laser_loc[t][2])**2
                        qmov = 3.0 * Qin[None] * output3[t, 0] * laser_on[t] \
                               / (pi * r_beam[None]**2) * ti.exp(-3.0 * r2 / (r_beam[None]**2))
                        for i in ti.static(range(4)):
                            rhs[element_surface_ids[el_id][sur_id, i]] += N[i] * qmov * surface_jac[el_id][sur_id, ip]

# convection + radiation on exposed surfaces (T^4 is smooth)
@ti.kernel
def update_fluxes_cr(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            for sur_id in ti.static(range(6)):
                if surface_birth[el_id][sur_id, 0] <= t*dt and surface_birth[el_id][sur_id, 1] > t*dt:
                    temperature_nodes = ti.Vector([
                        temperature[t-1, element_surface_ids[el_id][sur_id, 0]],
                        temperature[t-1, element_surface_ids[el_id][sur_id, 1]],
                        temperature[t-1, element_surface_ids[el_id][sur_id, 2]],
                        temperature[t-1, element_surface_ids[el_id][sur_id, 3]]])
                    for ip in ti.static(range(4)):
                        N = Nip_surface[ip]
                        temperature_ip = N.dot(temperature_nodes)
                        qconv = -1 * h_conv[None] * (temperature_ip - ambient[None])
                        qrad  = -1 * 5.67e-14 * h_rad * (temperature_ip**4 - ambient[None]**4)
                        for i in ti.static(range(4)):
                            rhs[element_surface_ids[el_id][sur_id, i]] += N[i] * (qconv + qrad) * surface_jac[el_id][sur_id, ip]

# explicit lumped update: T^{n+1} = T^n + dt * rhs / m_vec
@ti.kernel
def time_integrate(t: ti.i32, dt: ti.f32):
    for i in range(nn):
        if node_birth[i] <= t*dt:
            temperature[t, i] = ti.min(temperature[t-1, i] + dt * rhs[i] / m_vec[i], max_temp)

# smooth melt-pool depth: fixed local interpolation + linear isotherm crossing
@ti.func
def interp_under_beam(t, level):
    value = 0.0
    for j in ti.static(range(9)):
        node = beam_interp_node[t, level, j]       # precomputed from fixed mesh/toolpath
        value += beam_interp_weight[t, level, j] * temperature[t, node]
    return value
@ti.kernel
def update_melt_pool_depth(t: ti.i32):
    depth = 0.0
    for level in range(n_depth_levels - 1):
        t0 = interp_under_beam(t, level)
        t1 = interp_under_beam(t, level + 1)
        if t0 >= liquidus and t1 < liquidus:
            alpha = (t0 - liquidus) / (t0 - t1 + 1e-7)
            depth = depth_level[level] + alpha * (depth_level[level + 1] - depth_level[level])
    melt_depth[t] = depth

@ti.kernel
def compute_temperature_loss():
    for t, i in ti.ndrange(steps, nn):
        ti.atomic_add(loss[None], (temperature[t, i] - target[t, i])**2)   # squared-error numerator; partial obs masks i
@ti.kernel
def compute_melt_pool_loss():
    for t in range(1, steps):
        ti.atomic_add(loss[None], (melt_depth[t] - target_depth[t])**2)

def compute_loss():
    if loss_kind == "melt_pool":
        compute_melt_pool_loss()
    else:
        compute_temperature_loss()

def simulate():
    nn1(0); nn2(0); nn3(0)
    for time_step in range(1, steps):
        clear_vectors()
        nn1(time_step); nn2(time_step); nn3(time_step)   # P(t)
        update_matprop(time_step)
        update_mvec(time_step, dt)
        update_stiffness(time_step, dt)
        update_fluxes_m(time_step, dt)                   # laser injects P(t)
        update_fluxes_cr(time_step, dt)
        time_integrate(time_step, dt)
        update_melt_pool_depth(time_step)
    compute_loss()

# Adam over whichever parameters are the differentiable leaves
@ti.func
def adam(g, m, v, t):
    m = beta_1*m + (1-beta_1)*g
    v = beta_2*v + (1-beta_2)*(g*g)
    update = (m/(1-beta_1**t)) / (ti.sqrt(v/(1-beta_2**t)) + epsilon)
    return m, v, update

@ti.kernel
def update_weights_adam(t: ti.i32):
    for i in range(n_hidden_1):
        for j in range(n_input):
            weight1_m[i, j], weight1_v[i, j], step = adam(weight1.grad[i, j], weight1_m[i, j], weight1_v[i, j], t)
            weight1[i, j] -= learning_rate * step
        bias1_m[i], bias1_v[i], step = adam(bias1.grad[i], bias1_m[i], bias1_v[i], t)
        bias1[i] -= learning_rate * step
    for i in range(n_hidden_2):
        for j in range(n_hidden_1):
            weight2_m[i, j], weight2_v[i, j], step = adam(weight2.grad[i, j], weight2_m[i, j], weight2_v[i, j], t)
            weight2[i, j] -= learning_rate * step
        bias2_m[i], bias2_v[i], step = adam(bias2.grad[i], bias2_m[i], bias2_v[i], t)
        bias2[i] -= learning_rate * step
    for i in range(n_hidden_3):
        for j in range(n_hidden_2):
            weight3_m[i, j], weight3_v[i, j], step = adam(weight3.grad[i, j], weight3_m[i, j], weight3_v[i, j], t)
            weight3[i, j] -= learning_rate * step
        bias3_m[i], bias3_v[i], step = adam(bias3.grad[i], bias3_m[i], bias3_v[i], t)
        bias3[i] -= learning_rate * step

# optimize: one tape = the whole recurrence; one backward pass = the full gradient
for it in range(iterations):
    loss[None] = 0.0
    with ti.Tape(loss):     # records forward kernels; on exit replays adjoints in reverse (the discrete adjoint)
        simulate()
    update_weights_adam(it + 1)
```

## Notes / limitations

- Reverse-mode cost is O(1) in the parameter count but O(N) in memory over the N time steps (the stored temperature history); segment-wise checkpointing brings memory to O(√N) at the cost of one recomputation pass.
- Not everything is differentiable: dynamically searching for the last remelting time, for example, has no good differentiable form and must be precomputed/hard-coded; smoothing step-like features (melt-pool depth) is essential or the gradient vanishes.
- Gradient-based optimization is local and initialization-sensitive; the inverse problem is non-convex and parameters can be non-identifiable (e.g. low c_p vs high power produce similar responses), so the meaningful success criterion is collective loss reduction, not exact per-parameter recovery.
