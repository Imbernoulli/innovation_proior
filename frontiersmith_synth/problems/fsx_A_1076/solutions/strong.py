# TIER: strong
# Insight: connectivity/diversity must be reasoned about in the UNFOLDED
# orbit space, not the folded packet.  A cut placed generically (well clear
# of every reflection axis of the fold arrangement) is multiplied into
# |G| congruent, disjoint copies -- cheap area, but only ONE congruence
# class no matter how many such cuts you make.  To raise hole-motif
# diversity you must place SEVERAL differently-shaped cuts at genuinely
# different positions relative to the axes so their orbits land as
# separate, non-merging, mutually incongruent holes.  This solution keeps
# every cut cell at a safe distance from every axis of the fold group (so
# orbits stay clean and disjoint) while cycling through a menu of pairwise
# incongruent small polyomino motifs, spending just enough of the budget on
# each distinct motif to hit the diversity target, then spending the rest
# on repeats of already-used motifs to close the area gap toward 1/3.
import sys


def group_elements(N, t):
    def idf(r, c):
        return (r, c)

    def flipH(r, c):
        return (r, N - 1 - c)

    def flipV(r, c):
        return (N - 1 - r, c)

    def rot180(r, c):
        return (N - 1 - r, N - 1 - c)

    def flipD(r, c):
        return (c, r)

    def flipAD(r, c):
        return (N - 1 - c, N - 1 - r)

    def rot90(r, c):
        return (c, N - 1 - r)

    def rot270(r, c):
        return (N - 1 - c, r)

    if t == 2:
        return [idf, flipH]
    if t == 4:
        return [idf, flipH, flipV, rot180]
    return [idf, flipH, flipV, rot180, flipD, flipAD, rot90, rot270]


def orbit_cells(cells, G):
    s = set()
    for (r, c) in cells:
        for g in G:
            s.add(g(r, c))
    return s


MENU = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]


def is_cell_safe(r, c, N, t, axis_margin, boundary_margin):
    if not (boundary_margin <= r <= N - 1 - boundary_margin):
        return False
    if not (boundary_margin <= c <= N - 1 - boundary_margin):
        return False
    # vertical mirror axis (flipH reflects columns) present for t=2,4,8
    if abs(2 * c - (N - 1)) <= 2 * axis_margin:
        return False
    if t >= 4:
        # horizontal mirror axis (flipV reflects rows) present for t=4,8
        if abs(2 * r - (N - 1)) <= 2 * axis_margin:
            return False
    if t >= 8:
        if abs(r - c) <= axis_margin:
            return False
        if abs(r + c - (N - 1)) <= axis_margin:
            return False
    return True


def main():
    data = sys.stdin.read().split()
    N, t, V = int(data[0]), int(data[1]), int(data[2])
    G = group_elements(N, t)

    D_target = max(2, min(8, V // 6))
    target_cells = round(N * N / 3.0)
    M_target = min(V, max(1, round(target_cells / t)))

    axis_margin = 2
    boundary_margin = 2

    blocked = set()
    marks = []
    used = 0
    shape_idx = 0

    def place(cells):
        nonlocal used
        for cell in cells:
            marks.append(cell)
        used += len(cells)

    done = False
    for ar in range(0, N):
        if done:
            break
        for ac in range(0, N):
            if used >= V:
                done = True
                break
            shape = MENU[shape_idx % len(MENU)]
            cells = [(ar + dr, ac + dc) for dr, dc in shape]
            if not all(is_cell_safe(r, c, N, t, axis_margin, boundary_margin) for r, c in cells):
                continue
            if used + len(cells) > V:
                continue
            orb = orbit_cells(cells, G)
            if orb & blocked:
                continue
            place(cells)
            for cell in orb:
                for nr in range(cell[0] - 1, cell[0] + 2):
                    for nc in range(cell[1] - 1, cell[1] + 2):
                        blocked.add((nr, nc))
            shape_idx += 1
            if used >= M_target and shape_idx >= D_target:
                done = True
                break

    out = [str(len(marks))]
    for r, c in marks:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
