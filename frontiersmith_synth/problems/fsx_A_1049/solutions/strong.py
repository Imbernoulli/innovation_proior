# TIER: strong
# The insight: build the rotation-compatibility graph over the dictionary
# FIRST -- for each letter u, decode reverse(u) against the whole
# dictionary to find its forced rotation partner v(u) (an edge u ~ v(u) in
# the compatibility graph, weight = own margin + rotation margin), and
# throw out any u whose rotation doesn't decode at all (that silently
# removes the trap letter, since its rotation is unreadable by
# construction). The center must land on a SELF-loop of this graph (a near
# fixed point of the rotation) -- rare, so it is chosen deliberately, not
# by chance. The remaining k-1 slots are filled by walking the graph:
# take each distinct usable letter once (by descending edge weight) to
# harvest the distinct-letter bonus, then top up any leftover slots with
# the single best edge for pure margin.
import sys


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    L, k, d = int(head[0]), int(head[1]), int(head[2])
    letters = []
    pos = 1
    for _ in range(L):
        rows = data[pos:pos + 7]
        pos += 7
        letters.append("".join(r.strip() for r in rows))

    def decode(bitmap):
        dists = [hamming(bitmap, w) for w in letters]
        s = sorted(dists)
        if s[0] > d or s[0] >= s[1]:
            return None
        return dists.index(s[0]), s[1] - s[0]

    def own_margin(i):
        return decode(letters[i])[1]  # letters[i] decodes to itself exactly

    # ---- build the rotation-compatibility graph: u -> (v(u), weight) ----
    edge = {}
    for i in range(L):
        rd = decode(letters[i][::-1])
        if rd is None:
            continue  # unusable letter (this prunes the planted trap)
        v_idx, rot_margin = rd
        edge[i] = (v_idx, own_margin(i) + rot_margin)

    self_loops = [(i, w) for i, (v, w) in edge.items() if v == i]
    assert self_loops, "instance guarantees >=1 self-compatible letter"
    self_loops.sort(key=lambda t: (-t[1], t[0]))
    center_letter, _ = self_loops[0]

    usable = sorted(edge.items(), key=lambda kv: (-kv[1][1], kv[0]))  # (u,(v,w))
    order = [u for u, _ in usable]

    center = (k + 1) // 2
    slots_needed = k - 1
    plan = []
    for u in order:
        if len(plan) >= slots_needed:
            break
        plan.append(u)
    while len(plan) < slots_needed:
        plan.append(order[0])  # top up with the single best edge

    rows_out = [[""] * k for _ in range(7)]
    plan_iter = iter(plan)
    for j in range(1, k + 1):
        pick_i = center_letter if j == center else next(plan_iter)
        pick = letters[pick_i]
        for r in range(7):
            rows_out[r][j - 1] = pick[5 * r:5 * r + 5]

    print("\n".join("".join(rows_out[r]) for r in range(7)))


if __name__ == "__main__":
    main()
