#!/usr/bin/env python3
# Deterministic checker for "Rosette: Lead-Line Partition & Glass Colouring" (format C).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys, math

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        itxt = open(sys.argv[1]).read().split("\n")
        head = itxt[0].split()
        R, A, k, p = int(head[0]), int(head[1]), int(head[2]), int(head[3])
        we, wh = [float(x) for x in itxt[1].split()]
        H = []
        for r in range(p):
            H.append([float(x) for x in itxt[2 + r].split()])
    except Exception:
        fail("bad instance")

    if R < 1 or A < 1 or k < 1 or p < 2 or A % k != 0:
        fail("bad instance params")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = R * A * 2
    if len(otoks) != need:
        fail("wrong token count: need %d got %d" % (need, len(otoks)))

    face = [[0] * A for _ in range(R)]
    color = [[0] * A for _ in range(R)]
    idx = 0
    for r in range(R):
        for a in range(A):
            ftok = otoks[idx]; ctok = otoks[idx + 1]; idx += 2
            try:
                fid = int(ftok); cid = int(ctok)
            except Exception:
                fail("non-integer token at cell (%d,%d)" % (r, a))
            if not (math.isfinite(fid) and math.isfinite(cid)):
                fail("non-finite token at cell (%d,%d)" % (r, a))
            if fid < 0 or fid > 10**7:
                fail("face id out of range at (%d,%d)" % (r, a))
            if cid < 1 or cid > p:
                fail("color id out of range at (%d,%d)" % (r, a))
            face[r][a] = fid
            color[r][a] = cid

    # ---- feasibility 1: every face_id is a single connected component -----
    # (grid adjacency: angular neighbours wrap mod A; radial neighbours do not wrap)
    seen = [[False] * A for _ in range(R)]
    comp_of_face = {}
    for r0 in range(R):
        for a0 in range(A):
            if seen[r0][a0]:
                continue
            fid = face[r0][a0]
            stack = [(r0, a0)]
            seen[r0][a0] = True
            while stack:
                r, a = stack.pop()
                nbrs = [(r, (a - 1) % A), (r, (a + 1) % A)]
                if r > 0: nbrs.append((r - 1, a))
                if r < R - 1: nbrs.append((r + 1, a))
                for nr, na in nbrs:
                    if not seen[nr][na] and face[nr][na] == fid:
                        seen[nr][na] = True
                        stack.append((nr, na))
            comp_of_face[fid] = comp_of_face.get(fid, 0) + 1
    if any(v > 1 for v in comp_of_face.values()):
        fail("a face id spans more than one connected component")

    # ---- feasibility 2: k-fold rotational symmetry of the partition ----
    step = A // k
    rot_map = {}
    for r in range(R):
        for a in range(A):
            a2 = (a + step) % A
            f1, f2 = face[r][a], face[r][a2]
            if f1 in rot_map:
                if rot_map[f1] != f2:
                    fail("partition not k-fold rotationally symmetric")
            else:
                rot_map[f1] = f2

    # ---- feasibility 3: each face has one colour; adjacent DIFFERENT faces differ ----
    color_of_face = {}
    for r in range(R):
        for a in range(A):
            fid, cid = face[r][a], color[r][a]
            if fid in color_of_face:
                if color_of_face[fid] != cid:
                    fail("face %d has inconsistent colour" % fid)
            else:
                color_of_face[fid] = cid

    edges = []  # (r,a,r2,a2)
    for r in range(R):
        for a in range(A):
            edges.append((r, a, r, (a + 1) % A))
            if r < R - 1:
                edges.append((r, a, r + 1, a))

    harmony_sum = 0.0
    for (r, a, r2, a2) in edges:
        f1, f2 = face[r][a], face[r2][a2]
        if f1 == f2:
            continue
        c1, c2 = color[r][a], color[r2][a2]
        if c1 == c2:
            fail("adjacent faces %d,%d share colour %d" % (f1, f2, c1))
        harmony_sum += H[c1 - 1][c2 - 1]

    # ---- objective: face-area entropy (bits) + harmony sum ----
    areas = {}
    for r in range(R):
        for a in range(A):
            fid = face[r][a]
            areas[fid] = areas.get(fid, 0) + 1
    total = R * A
    Hent = 0.0
    for cnt in areas.values():
        pr = cnt / total
        Hent -= pr * math.log2(pr)

    F = we * Hent + wh * harmony_sum

    # ---- internal trivial baseline: R full-ring faces, colours alternate 1,2 ----
    Hb = math.log2(R) if R > 1 else 0.0
    harmony_b = 0.0
    for r in range(R - 1):
        c1 = 1 + (r % 2)
        c2 = 1 + ((r + 1) % 2)
        harmony_b += A * H[c1 - 1][c2 - 1]
    B = we * Hb + wh * harmony_b
    if B <= 1e-9:
        fail("degenerate baseline")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f faces=%d Ratio: %.6f" % (F, B, len(areas), ratio))


if __name__ == "__main__":
    main()
