# TIER: greedy
# The obvious textbook move: find the plank that swings hardest in the
# structure's own dominant mode (a simple modal analysis is fair game -- the
# masses and stiffnesses are public) and dump the WHOLE ballast budget there,
# on the theory that "weighing down the biggest swinger" must help. Then scan
# cargo loads to find the best one under that fixed placement.
#
# This recipe never asks what adding all that mass at one spot does to the
# OTHER modes -- it only ever looks at the single mode it started from.
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

    # dominant mode = the one (among the 3 lowest) most excited by a generic
    # single-hump wind profile; "biggest swinger" = its largest-amplitude plank.
    hump = np.sin(np.pi * (np.arange(n) + 1) / (n + 1))
    w0, phi0 = _modes(base_mass, stiff)
    proj0 = np.abs(phi0.T.dot(hump))
    m1 = int(np.argmax(proj0[:min(3, n)]))
    i_star = int(np.argmax(np.abs(phi0[:, m1])))

    ballast = [0] * n
    ballast[i_star] = budget
    ballast_arr = np.array(ballast, dtype=float)

    best_obj, best_cargo = 0.0, 0
    for cargo in range(0, cap + 1):
        mass = base_mass + ballast_arr
        mass = mass.copy()
        mass[node] += cargo
        peak = _peak(mass, stiff, gusts, zeta, node)
        cnt = sum(1 for s in storms if s * peak <= thr + 1e-12)
        obj = cargo * (cnt / len(storms))
        if obj > best_obj:
            best_obj = obj; best_cargo = cargo

    print(json.dumps({"ballast": ballast, "cargo": best_cargo}))


if __name__ == "__main__":
    main()
