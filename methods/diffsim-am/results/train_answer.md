In metal additive manufacturing, the transient thermal field that sweeps through a part as the laser moves determines nearly everything that matters: local cooling rates fix the microstructure, accumulated heat drives residual stress and distortion, and the melt-pool depth governs geometric accuracy and porosity. The practical problem is therefore the inverse one: given a target thermal response or melt-pool behavior, find the process parameters that produce it. The hard version is not a few static scalars but a full time-series laser power schedule P(t) over tens of thousands of explicit time steps. The obstacle is the cost of gradients. Finite differences needs one complete forward simulation per parameter, so the gradient cost scales with the number of parameters and becomes completely infeasible for a 20,000-dimensional power schedule. Derivative-free methods suffer the same curse. Pure data-driven surrogates are cheap to query but violate the known physics outside their training domain and cannot faithfully couple heat transfer to an arbitrary new geometry. Hand-derived continuous adjoints give the right asymptotic cost, but each new objective, boundary condition, or parameterization requires a bespoke and error-prone re-derivation.

The way out is to make the thermal simulator itself differentiable and use reverse-mode automatic differentiation. The loss is a scalar and the parameters are many, so reverse mode computes the full gradient with respect to all parameters in essentially one backward pass at a cost independent of the parameter count. Backpropagating through the explicit time-stepping finite-element recurrence is exactly the discrete adjoint of the solver, obtained automatically instead of by hand. The method is DiffSim-AM, a Taichi-based differentiable explicit thermal finite-element solver for additive-manufacturing process design. It represents every state and parameter as a differentiable field, records the forward kernels on a lightweight tape, and replays their adjoints in reverse to populate gradients for the continuous design variables. The simulator is kept physically faithful: Hex8 elements, lumped capacitance, moving Gaussian surface laser flux, convection, radiation, and latent heat folded into a temperature-dependent specific heat. The only approximation is the one that makes gradients meaningful: element birth, surface birth, and the toolpath are precomputed and frozen behind boolean masks, so the gradient flows only through continuous quantities such as laser power, beam radius, heat capacity, conductivity, and convection coefficient, never across the discrete material-deposition jumps. Step-like derived quantities are also smoothed: melt-pool depth is computed by interpolating temperature under the beam at fixed depth levels and finding the liquidus-isotherm crossing by pairwise-linear interpolation, giving a nonzero gradient where a discrete deepest-node definition would be piecewise constant and dead.

The control is parameterized compactly by a small fully-connected network that maps normalized time to a normalized laser power in [0, 1]. This produces a smooth schedule, regularizes the high-dimensional temporal control, and lets Adam optimize a few thousand network weights rather than 20,000 independent power values. The same differentiable machinery also handles static calibration: the leaves can instead be scalar material and process parameters inferred from partial top-layer observations. Three tasks fit naturally: inferring static scalars from sparse thermal data, matching a target full thermal history by shaping P(t), and holding a target melt-pool depth throughout a geometrically varying build such as an hourglass part. In each case the loop is the same: run the differentiable simulation under Taichi's tape, accumulate the scalar squared-error loss, exit the tape to trigger one reverse pass, and apply Adam to the differentiable leaves. Memory is O(N) in the number of time steps because the state history is needed by the adjoint, but segment-wise checkpointing can reduce this to O(sqrt(N)) by storing boundary states and recomputing within segments.

```python
import numpy as np
import taichi as ti

# scales / properties (stainless steel)
density = 0.03; cp_init = 0.5; cond_init = 0.01
Qin_init = 250.0; r_beam_init = 1.0; h_conv_init = 0.00005; h_rad = 0.2
ambient_init = 300.0; max_temp = 2000.0
solidus, liquidus = 1533.15, 1609.15
latent_cp = 272.0 / (liquidus - solidus)
pi = 3.141592653589793

n_input, n_hidden_1, n_hidden_2, n_hidden_3 = 3, 50, 50, 1
learning_rate, beta_1, beta_2, epsilon = 1e-2, 0.9, 0.999, 1e-7
loss_kind = "temperature_history"
n_depth_levels = 4

ti.init()

temperature = ti.field(float, (steps, nn), needs_grad=True)
m_vec = ti.field(float, nn, needs_grad=True)
rhs   = ti.field(float, nn, needs_grad=True)
ambient = ti.field(float, (), needs_grad=True)
cp   = ti.Vector.field(8, float, nel, needs_grad=True)
cond = ti.Vector.field(8, float, nel, needs_grad=True)
Qin  = ti.field(float, (), needs_grad=True)
r_beam = ti.field(float, (), needs_grad=True)
h_conv = ti.field(float, (), needs_grad=True)
loss = ti.field(float, (), needs_grad=True)

weight1 = ti.field(float, (n_hidden_1, n_input),  needs_grad=True)
bias1 = ti.field(float, n_hidden_1, needs_grad=True)
weight2 = ti.field(float, (n_hidden_2, n_hidden_1), needs_grad=True)
bias2 = ti.field(float, n_hidden_2, needs_grad=True)
weight3 = ti.field(float, (n_hidden_3, n_hidden_2), needs_grad=True)
bias3 = ti.field(float, n_hidden_3, needs_grad=True)

weight1_m = ti.field(float, (n_hidden_1, n_input))
weight1_v = ti.field(float, (n_hidden_1, n_input))
weight2_m = ti.field(float, (n_hidden_2, n_hidden_1))
weight2_v = ti.field(float, (n_hidden_2, n_hidden_1))
weight3_m = ti.field(float, (n_hidden_3, n_hidden_2))
weight3_v = ti.field(float, (n_hidden_3, n_hidden_2))
bias1_m = ti.field(float, n_hidden_1); bias1_v = ti.field(float, n_hidden_1)
bias2_m = ti.field(float, n_hidden_2); bias2_v = ti.field(float, n_hidden_2)
bias3_m = ti.field(float, n_hidden_3); bias3_v = ti.field(float, n_hidden_3)

output1 = ti.field(float, (steps, n_hidden_1), needs_grad=True)
output2 = ti.field(float, (steps, n_hidden_2), needs_grad=True)
output3 = ti.field(float, (steps, n_hidden_3), needs_grad=True)
melt_depth = ti.field(float, steps, needs_grad=True)
target = ti.field(float, (steps, nn))
target_depth = ti.field(float, steps)
beam_interp_node = ti.field(int, (steps, n_depth_levels, 9))
beam_interp_weight = ti.field(float, (steps, n_depth_levels, 9))
depth_level = ti.field(float, n_depth_levels)

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

@ti.func
def calc_cp(t, el_id, solidus, liquidus, latent_cp):
    cp_el = ti.Vector.zero(float, 8)
    for ip in ti.static(range(8)):
        theta_ip = 0.0
        N = Nip_element[ip]
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

@ti.func
def calc_mass(t, el_id, density):
    mass = ti.Matrix.zero(float, 8, 8)
    lump_mass = ti.Vector.zero(float, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        N = Nip_element[ip]
        B = Bip_element[ip]
        detJac = (B @ nodes_pos).determinant()
        mass += density * cp[el_id][ip] * detJac * N @ N.transpose()
    for i in ti.static(range(8)):
        for j in ti.static(range(8)):
            lump_mass[i] += mass[i, j]
    return lump_mass

@ti.kernel
def update_mvec(t: ti.i32, dt: ti.f32):
    for el_id in range(nel):
        if element_birth[el_id] <= t*dt:
            lump_mass = calc_mass(t, el_id, density)
            for i in ti.static(range(8)):
                m_vec[elements_node_ids[el_id][i]] += lump_mass[i]

@ti.func
def calc_stiffness(el_id):
    stiffness = ti.Matrix.zero(float, 8, 8)
    nodes_pos = ti.Matrix.rows([node_position[elements_node_ids[el_id][k]] for k in range(8)])
    for ip in ti.static(range(8)):
        B = Bip_element[ip]
        Jac = B @ nodes_pos
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

@ti.kernel
def time_integrate(t: ti.i32, dt: ti.f32):
    for i in range(nn):
        if node_birth[i] <= t*dt:
            temperature[t, i] = ti.min(temperature[t-1, i] + dt * rhs[i] / m_vec[i], max_temp)

@ti.func
def interp_under_beam(t, level):
    value = 0.0
    for j in ti.static(range(9)):
        node = beam_interp_node[t, level, j]
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
        ti.atomic_add(loss[None], (temperature[t, i] - target[t, i])**2)

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
        nn1(time_step); nn2(time_step); nn3(time_step)
        update_matprop(time_step)
        update_mvec(time_step, dt)
        update_stiffness(time_step, dt)
        update_fluxes_m(time_step, dt)
        update_fluxes_cr(time_step, dt)
        time_integrate(time_step, dt)
        update_melt_pool_depth(time_step)
    compute_loss()

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

for it in range(iterations):
    loss[None] = 0.0
    with ti.Tape(loss):
        simulate()
    update_weights_adam(it + 1)
```
