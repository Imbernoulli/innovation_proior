# TIER: strong
# Insight: amplitude is a resonance overlap integral (participation of each
# mode with the gust shapes, divided by how far that mode's frequency sits
# from the gust tones). So the lever is the STRUCTURE'S SPECTRUM, not "which
# plank looks like it swings the most". We build the mass/stiffness model
# ourselves, solve for the modes, and search ballast layouts with EXCHANGE
# moves (move one unit of ballast from plank j to plank i) that directly
# minimize the resulting peak resonance response -- not merely "greedy-add,
# never look back". Multiple restarts (including the naive center-load) plus
# an alternating ballast/cargo refinement round let the search escape the
# local optimum that traps a single fixed-point placement.
import sys, json, math
import numpy as np


def _modes(mass, stiff):
    n = len(mass)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = stiff[i] + stiff[i + 1]
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = -stiff[i + 1]
    linv = 1.0 / np.sqrt(mass)
    Ksym = (linv[:, None] * K) * linv[None, :]
    Ksym = 0.5 * (Ksym + Ksym.T)
    w, y = np.linalg.eigh(Ksym)
    w = np.clip(w, 1e-9, None)
    phi = linv[:, None] * y
    return w, phi


def _peak(mass, stiff, gusts, zeta, node):
    w, phi = _modes(mass, stiff)
    omega_k = np.sqrt(w)
    total = 0.0
    for g in gusts:
        shape = np.array(g["shape"], dtype=float)
        omega_g = float(g["omega"]); amp = float(g["amp"])
        p = phi.T.dot(shape)
        denom = np.sqrt((w - omega_g ** 2) ** 2 + (2.0 * zeta * omega_k * omega_g) ** 2)
        denom = np.maximum(denom, 1e-9)
        total += amp * float(np.sum(np.abs(p * phi[node, :]) / denom))
    return total


def _frac(peak, storms, thr):
    return sum(1 for s in storms if s * peak <= thr + 1e-12) / len(storms)


def _best_cargo(base_mass, stiff, ballast, cap, node, gusts, zeta, storms, thr):
    best_obj, best_c = 0.0, 0
    mass_b = base_mass + ballast
    for c in range(0, cap + 1):
        mass = mass_b.copy(); mass[node] += c
        peak = _peak(mass, stiff, gusts, zeta, node)
        obj = c * _frac(peak, storms, thr)
        if obj > best_obj:
            best_obj = obj; best_c = c
    return best_obj, best_c


def _local_search(base_mass, stiff, node, gusts, zeta, budget, n, start, cargo_ref, sweeps=6):
    ballast = start.copy()
    mass = base_mass + ballast; mass = mass.copy(); mass[node] += cargo_ref
    cur = _peak(mass, stiff, gusts, zeta, node)
    for _ in range(sweeps):
        improved = False
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                if ballast[j] <= 0:
                    break
                ballast[i] += 1; ballast[j] -= 1
                mass = base_mass + ballast; mass = mass.copy(); mass[node] += cargo_ref
                p = _peak(mass, stiff, gusts, zeta, node)
                if p < cur - 1e-9:
                    cur = p; improved = True
                else:
                    ballast[i] -= 1; ballast[j] += 1
        if not improved:
            break
    return ballast


def main():
    inst = json.load(sys.stdin)
    n = inst["n_nodes"]
    budget = inst["ballast_budget"]
    cap = inst["cargo_cap"]
    node = inst["cargo_node"]
    zeta = inst["damping_zeta"]
    gusts = inst["gust_components"]
    storms = inst["storm_scales"]
    thr = inst["amp_threshold"]
    base_mass = np.array(inst["base_mass"], dtype=float)
    stiff = np.array(inst["stiffness"], dtype=float)

    hump = np.sin(np.pi * (np.arange(n) + 1) / (n + 1))
    w0, phi0 = _modes(base_mass, stiff)
    proj0 = np.abs(phi0.T.dot(hump))
    m1 = int(np.argmax(proj0[:min(3, n)]))
    i_star = int(np.argmax(np.abs(phi0[:, m1])))

    starts = []
    b_center = np.zeros(n); b_center[i_star] = budget
    starts.append(b_center)
    q, rem = divmod(budget, n)
    b_uni = np.full(n, float(q))
    for i in range(rem):
        b_uni[i] += 1
    starts.append(b_uni)
    j_far = int(np.argmax(np.abs(np.arange(n) - i_star)))
    b_split = np.zeros(n); half = budget // 2
    b_split[i_star] = budget - half; b_split[j_far] += half
    starts.append(b_split)
    starts.append(np.zeros(n))

    best_obj, best_ballast, best_cargo = -1.0, None, 0
    for start in starts:
        # phase 1: local search under worst-case (full-cargo) loading
        ballast = _local_search(base_mass, stiff, node, gusts, zeta, budget, n, start, cap)
        obj, cargo = _best_cargo(base_mass, stiff, ballast, cap, node, gusts, zeta, storms, thr)
        # phase 2: refine ballast under the ACTUAL chosen cargo, then re-pick cargo
        if cargo > 0:
            ballast2 = _local_search(base_mass, stiff, node, gusts, zeta, budget, n, ballast, cargo)
            obj2, cargo2 = _best_cargo(base_mass, stiff, ballast2, cap, node, gusts, zeta, storms, thr)
            if obj2 > obj:
                ballast, obj, cargo = ballast2, obj2, cargo2
        if obj > best_obj:
            best_obj, best_ballast, best_cargo = obj, ballast, cargo

    ballast_out = [int(round(x)) for x in best_ballast]
    # guard the integer budget constraint against any rounding drift
    over = sum(ballast_out) - budget
    idx = 0
    while over > 0 and idx < n:
        take = min(over, ballast_out[idx])
        ballast_out[idx] -= take
        over -= take
        idx += 1

    print(json.dumps({"ballast": ballast_out, "cargo": int(best_cargo)}))


if __name__ == "__main__":
    main()
