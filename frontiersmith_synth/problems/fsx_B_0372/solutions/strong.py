# TIER: strong
# SABRE-style routing over the pending operation set (all operations commute, so the
# whole pending set is the "front layer"). Repeatedly: (1) perform any pending op whose
# batches are already adjacent for free; (2) otherwise apply the canal exchange that most
# reduces the total distance of the nearest lookahead operations -- one exchange can
# advance several pending pairs at once. Several lookahead widths (plus plain nearest-first)
# are run and the cheapest plan is emitted.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    rows = int(next(it)); cols = int(next(it)); K = int(next(it))
    V = rows * cols
    ops = [(int(next(it)), int(next(it))) for _ in range(K)]

    def dist_nodes(u, w):
        ru, cu = divmod(u, cols); rw, cw = divmod(w, cols)
        return abs(ru - rw) + abs(cu - cw)

    def neighbors(nd):
        r, c = divmod(nd, cols)
        res = []
        if r > 0: res.append(nd - cols)
        if r < rows - 1: res.append(nd + cols)
        if c > 0: res.append(nd - 1)
        if c < cols - 1: res.append(nd + 1)
        return res

    def path_manhattan(src, dst):
        r0, c0 = divmod(src, cols); r1, c1 = divmod(dst, cols)
        path = [src]; r, c = r0, c0
        sr = 1 if r1 > r else -1
        while r != r1:
            r += sr; path.append(r * cols + c)
        sc = 1 if c1 > c else -1
        while c != c1:
            c += sc; path.append(r * cols + c)
        return path

    def apply_swap(pos, inv, u, w, moves):
        tu, tw = inv[u], inv[w]
        inv[u], inv[w] = tw, tu
        pos[tu], pos[tw] = w, u
        moves.append("S %d %d" % (u, w))

    def run_nearest():
        pos = list(range(V)); inv = list(range(V))
        pending = set(range(K))
        moves = []
        while pending:
            best = None; bestd = None
            for t in pending:
                a, b = ops[t]
                d = dist_nodes(pos[a], pos[b])
                if bestd is None or d < bestd or (d == bestd and t < best):
                    bestd = d; best = t
            t = best
            a, b = ops[t]
            path = path_manhattan(pos[a], pos[b])
            for i in range(len(path) - 2):
                apply_swap(pos, inv, path[i], path[i + 1], moves)
            moves.append("X %d" % t)
            pending.discard(t)
        return moves

    def run_sabre(L):
        pos = list(range(V)); inv = list(range(V))
        pending = set(range(K))
        moves = []
        while pending:
            # fulfill all currently-adjacent ops
            progressed = True
            while progressed:
                progressed = False
                for t in list(pending):
                    a, b = ops[t]
                    if dist_nodes(pos[a], pos[b]) == 1:
                        moves.append("X %d" % t)
                        pending.discard(t)
                        progressed = True
            if not pending:
                break
            # lookahead set = L nearest pending ops
            order = sorted(pending, key=lambda t: (dist_nodes(pos[ops[t][0]], pos[ops[t][1]]), t))
            look = order[:L]
            base = sum(dist_nodes(pos[ops[t][0]], pos[ops[t][1]]) for t in look)
            involved = set()
            for t in look:
                a, b = ops[t]
                involved.add(pos[a]); involved.add(pos[b])
            cand = set()
            for nd in involved:
                for nb in neighbors(nd):
                    cand.add((nd, nb) if nd < nb else (nb, nd))
            best_swap = None; best_sum = None
            for (u, w) in sorted(cand):
                tu, tw = inv[u], inv[w]
                # tentatively swap
                inv[u], inv[w] = tw, tu
                pos[tu], pos[tw] = w, u
                s = sum(dist_nodes(pos[ops[t][0]], pos[ops[t][1]]) for t in look)
                # undo
                inv[u], inv[w] = tu, tw
                pos[tu], pos[tw] = u, w
                if best_sum is None or s < best_sum:
                    best_sum = s; best_swap = (u, w)
            if best_swap is not None and best_sum < base:
                apply_swap(pos, inv, best_swap[0], best_swap[1], moves)
            else:
                # fallback: route the single nearest pending op directly
                t = order[0]
                a, b = ops[t]
                path = path_manhattan(pos[a], pos[b])
                for i in range(len(path) - 2):
                    apply_swap(pos, inv, path[i], path[i + 1], moves)
                moves.append("X %d" % t)
                pending.discard(t)
        return moves

    plans = [run_nearest(), run_sabre(3), run_sabre(8), run_sabre(max(4, K))]
    best = min(plans, key=len)
    sys.stdout.write("%d\n%s\n" % (len(best), "\n".join(best)))

if __name__ == "__main__":
    main()
