"""plumelib.py -- shared deterministic instance builder + forward (advection-diffusion)
model for the plume-source-inversion problem. Imported by gen.py and verify.py ONLY
(never shipped to / readable by the sandboxed participant solutions).

Everything about one test instance -- grid size, wells, wind/diffusion, the TRUE sparse
source, and the held-out wells -- is a pure function of the integer testId via
build_instance(testId). gen.py prints the visible (non-secret) half to stdout; verify.py
calls build_instance() again on the testId it reads back from the .in file to regenerate
the TRUE source + held-out wells (which are never printed) and grade against them.
"""
import math
import random

MT = 3          # number of observation snapshot times
KH = 6          # number of held-out (never shown) monitoring wells
S_MAX = 3       # sparsity-prior cap: at most this many active leak cells
SALT = 20260726712  # fixed arbitrary salt (no wall-clock, no external entropy)

# testId -> (N grid side, K visible wells, S true active source cells)
_LADDER = {
    1:  (5, 6, 1),
    2:  (6, 5, 1),
    3:  (6, 4, 1),
    4:  (8, 5, 1),
    5:  (8, 4, 2),
    6:  (9, 5, 1),
    7:  (9, 4, 1),
    8:  (10, 5, 2),
    9:  (10, 4, 1),
    10: (11, 5, 3),
}


def cell_center(i, j):
    """Grid cell (row i, col j) -> its center in continuous (x, y) coordinates."""
    return (j + 0.5, i + 0.5)


def green(x0, y0, xw, yw, t, vx, vy, D):
    """2D advection-diffusion Green's function: concentration at (xw,yw) at time t
    after elapsed time from an instantaneous unit-rate release at (x0,y0), under
    uniform drift (vx,vy) and isotropic diffusivity D. (Heat kernel with drift.)"""
    dx = xw - x0 - vx * t
    dy = yw - y0 - vy * t
    denom = 4.0 * D * t
    return math.exp(-(dx * dx + dy * dy) / denom) / (math.pi * denom)


def forward_conc(rates_full, cells_enum, well_xy, times, vx, vy, D):
    """rates_full: list of per-cell release rates, parallel to cells_enum (row-major,
    index = i*N+j). Returns concentration at well_xy for each time in `times`."""
    xw, yw = well_xy
    out = []
    active = [(cell_center(i, j), r) for (i, j), r in zip(cells_enum, rates_full) if r != 0.0]
    for t in times:
        c = 0.0
        for (x0, y0), r in active:
            c += r * green(x0, y0, xw, yw, t, vx, vy, D)
        out.append(c)
    return out


def build_instance(test_id):
    t = int(test_id)
    if t < 1:
        t = 1
    if t > 10:
        t = 10
    N, K, S = _LADDER[t]
    M = N * N
    rnd = random.Random(SALT + 1009 * t)

    D = 0.85 + 0.05 * t + rnd.uniform(-0.03, 0.03)
    vx = 0.5 + 0.03 * t + rnd.uniform(-0.05, 0.05)
    vy = 0.25 - 0.015 * t + rnd.uniform(-0.05, 0.05)
    scale = N / 8.0
    times = [round(v * scale, 4) for v in (1.4, 2.3, 3.4)]

    lo, hi = 1, N - 2
    if hi < lo:
        lo, hi = 0, N - 1
    interior = [(i, j) for i in range(lo, hi + 1) for j in range(lo, hi + 1)]
    rnd.shuffle(interior)
    true_cells = []
    for cand in interior:
        if all(abs(cand[0] - c[0]) + abs(cand[1] - c[1]) >= 2 for c in true_cells):
            true_cells.append(cand)
        if len(true_cells) == S:
            break
    while len(true_cells) < S:
        cand = interior[len(true_cells) % len(interior)]
        if cand not in true_cells:
            true_cells.append(cand)
        else:
            true_cells.append(interior[(len(true_cells) + 1) % len(interior)])

    true_rates = [rnd.uniform(40.0, 100.0) for _ in range(S)]
    B_mass = sum(true_rates)

    cells_enum = [(i, j) for i in range(N) for j in range(N)]
    cell_index = {c: k for k, c in enumerate(cells_enum)}
    rates_full = [0.0] * M
    for c, r in zip(true_cells, true_rates):
        rates_full[cell_index[c]] = r

    pool = list(cells_enum)
    rnd.shuffle(pool)
    used = set(true_cells)
    visible_wells = []
    idx = 0
    while len(visible_wells) < K:
        cand = pool[idx % len(pool)]
        idx += 1
        if cand in used or cand in visible_wells:
            continue
        visible_wells.append(cand)
    held_wells = []
    while len(held_wells) < KH:
        cand = pool[idx % len(pool)]
        idx += 1
        if cand in visible_wells or cand in held_wells:
            continue
        held_wells.append(cand)

    sigma_rel = 0.02
    vis_readings = []
    for w in visible_wells:
        xw, yw = cell_center(*w)
        clean = forward_conc(rates_full, cells_enum, (xw, yw), times, vx, vy, D)
        noisy = []
        for c in clean:
            noise = rnd.gauss(0.0, sigma_rel * max(c, 1e-6) + 1e-4)
            v = c + noise
            if v < 0.0:
                v = 0.0
            noisy.append(round(v, 6))
        vis_readings.append(noisy)

    return dict(
        test_id=t, N=N, K=K, S=S, M=M, D=D, vx=vx, vy=vy, times=times, B_mass=B_mass,
        visible_wells=visible_wells, vis_readings=vis_readings,
        true_cells=true_cells, true_rates=true_rates, held_wells=held_wells,
        cells_enum=cells_enum,
    )
