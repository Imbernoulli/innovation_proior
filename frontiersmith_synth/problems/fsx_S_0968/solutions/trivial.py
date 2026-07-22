# TIER: trivial
# Do the bare minimum: spread MOST of the ballast budget evenly across every
# plank (no analysis of the structure's spectrum at all -- "just add some
# weight, roughly evenly"), then linearly scan cargo loads to find the
# largest one that still clears the danger threshold under this fixed,
# thoughtless ballast layout.
import sys, json, math

try:
    import numpy as np
except Exception:
    np = None


def _modes(mass, stiff):
    n = len(mass)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        K[i][i] = stiff[i] + stiff[i + 1]
    for i in range(n - 1):
        K[i][i + 1] = K[i + 1][i] = -stiff[i + 1]
    Kn = np.array(K, dtype=float)
    linv = 1.0 / np.sqrt(np.array(mass, dtype=float))
    Ksym = (linv[:, None] * Kn) * linv[None, :]
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

    used = int(round(0.9 * budget))
    q, rem = divmod(used, n)
    ballast = [q] * n
    for i in range(rem):
        ballast[i] += 1

    best_obj, best_cargo = 0.0, 0
    ballast_arr = np.array(ballast, dtype=float)
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
