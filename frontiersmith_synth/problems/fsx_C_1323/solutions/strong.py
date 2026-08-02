# TIER: strong
# The decoupling insight: a pharmacophore anchor only constrains the TYPE and
# CHAIN POSITION of an atom, never the identity of the fragment that carries
# it. The library is deliberately redundant -- for every feature type there
# is an "off-reference" fragment (never used anywhere in the known active)
# that places its atom at the exact same canonical offset, and it is also
# cheaper than the reference's own choice. So: at every anchor position, drop
# in the off-reference feature fragment of the required type instead of the
# reference's own fragment; at every other position, drop in the cheapest
# off-reference filler. Every fragment id used is then completely absent from
# the reference's multiset (novelty saturates at N=1) while every anchor is
# still hit exactly (P=1) and the total cost is strictly below the
# reference's (efficiency saturates at 1) -- activity preservation and
# novelty stop trading off against each other entirely.
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0
    M = int(tok[p]); p += 1
    L_max = int(tok[p]); p += 1
    STEP = int(tok[p]); p += 1
    BUDGET = int(tok[p]); p += 1

    costs = []
    types = []
    offs = []
    for _ in range(M):
        fid = int(tok[p]); p += 1
        cost = int(tok[p]); p += 1
        typ = tok[p]; p += 1
        dx = float(tok[p]); p += 1
        dy = float(tok[p]); p += 1
        dz = float(tok[p]); p += 1
        costs.append(cost)
        types.append(typ)
        offs.append((dx, dy, dz))

    L_ref = int(tok[p]); p += 1
    ref_seq = [int(tok[p + i]) for i in range(L_ref)]; p += L_ref
    K = int(tok[p]); p += 1
    anchors_by_index = {}
    for _ in range(K):
        x = float(tok[p]); p += 1
        y = float(tok[p]); p += 1
        z = float(tok[p]); p += 1
        typ = tok[p]; p += 1
        tol = float(tok[p]); p += 1
        idx = round(x / STEP)
        anchors_by_index[idx] = (x, y, z, typ, tol)

    ref_ids = set(ref_seq)

    def dist(a, b):
        return sum((u - v) ** 2 for u, v in zip(a, b)) ** 0.5

    def cheapest_off_reference(pred):
        cands = [fid for fid in range(M) if pred(fid) and fid not in ref_ids]
        if not cands:
            # fallback (should not trigger given the fixed library): allow
            # reuse of a reference id rather than emit something infeasible.
            cands = [fid for fid in range(M) if pred(fid)]
        return min(cands, key=lambda fid: (costs[fid], fid))

    filler_id = cheapest_off_reference(lambda fid: types[fid] == 'X')

    seq = []
    for i in range(L_ref):
        if i in anchors_by_index:
            ax, ay, az, atyp, atol = anchors_by_index[i]

            def hits(fid, i=i, ax=ax, ay=ay, az=az, atyp=atyp, atol=atol):
                if types[fid] != atyp:
                    return False
                dx, dy, dz = offs[fid]
                gx, gy, gz = i * STEP + dx, dy, dz
                return dist((gx, gy, gz), (ax, ay, az)) <= atol + 1e-9

            seq.append(cheapest_off_reference(hits))
        else:
            seq.append(filler_id)

    print(len(seq))
    print(" ".join(str(x) for x in seq))


if __name__ == "__main__":
    main()
