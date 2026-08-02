# TIER: strong
"""
Insight: a call site's true value is not its own removed call overhead -- it is that
overhead PLUS every constant-propagation bonus its inlining enables further down its
unlock chain. So instead of scoring call sites individually, we reformulate: group each
chain (a maximal parent-follows-parent path) into a small set of DEPENDENT "prefix"
options -- inline the first t links of the chain, t = 0..length -- and compute the exact
post-optimization value (overhead saved by every inlined link, PLUS every bonus that a
full prefix actually unlocks) and the exact size cost of that whole prefix. Standalone
call sites are simply length-1 chains.

This turns the decision into a GROUPED knapsack: pick at most one prefix option per
chain, respecting the total icache budget, maximizing total value -- solved exactly with
a DP over the size budget (m and ICACHE_CAP are both small). That already beats
frequency-greedy because it prices the whole chain's payoff, not each link alone.

Then, since the icache penalty is a multiplier on the WHOLE dynamic cost rather than a
hard wall, we probe a bounded number of deterministic "overshoot" moves on top of the
capped DP solution -- try adding each not-yet-taken chain's best remaining option even
though it pushes S past ICACHE_CAP, keep it only if the actual replayed F improves. This
is local, deterministic, and does not attempt an exhaustive global search (that coupled
nonlinear search is the genuinely open problem here) -- it is a bounded refinement on top
of the chain-aware knapsack, not "greedy plus more iterations."
"""
import sys


def unlocked_mask(chosen, parent, m):
    ok_cache = {}

    def chain_ok(i):
        if i in ok_cache:
            return ok_cache[i]
        if i not in chosen:
            ok_cache[i] = False
            return False
        p = parent[i]
        if p == 0:
            ok_cache[i] = True
            return True
        r = chain_ok(p)
        ok_cache[i] = r
        return r

    unlocked = {}
    for i in range(1, m + 1):
        if i in chosen and parent[i] != 0:
            unlocked[i] = chain_ok(parent[i])
        else:
            unlocked[i] = False
    return unlocked


def evaluate(chosen, freq, base_cost, inline_size, parent, bonus, m,
             S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF):
    unlocked = unlocked_mask(chosen, parent, m)
    D = 0
    for i in range(1, m + 1):
        eff = base_cost[i]
        if unlocked[i]:
            eff -= bonus[i]
            if eff < 1:
                eff = 1
        overhead = 0 if i in chosen else CALL_OVERHEAD
        D += freq[i] * (eff + overhead)
    S = S_base + sum(inline_size[i] for i in chosen)
    excess = max(0, S - ICACHE_CAP)
    F = (D * (ICACHE_CAP + PENALTY_COEF * excess)) // ICACHE_CAP
    return F, S


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    S_base = int(next(it))
    ICACHE_CAP = int(next(it))
    CALL_OVERHEAD = int(next(it))
    PENALTY_COEF = int(next(it))

    freq = [0] * (m + 1)
    base_cost = [0] * (m + 1)
    inline_size = [0] * (m + 1)
    parent = [0] * (m + 1)
    bonus = [0] * (m + 1)
    children = [[] for _ in range(m + 1)]
    for i in range(1, m + 1):
        freq[i] = int(next(it))
        base_cost[i] = int(next(it))
        inline_size[i] = int(next(it))
        parent[i] = int(next(it))
        bonus[i] = int(next(it))
        if parent[i] != 0:
            children[parent[i]].append(i)

    # --- build chains: follow root -> single child -> single child -> ... ---
    chains = []
    for r in range(1, m + 1):
        if parent[r] == 0:
            path = [r]
            cur = r
            while len(children[cur]) == 1:
                cur = children[cur][0]
                path.append(cur)
            chains.append(path)

    # --- for each chain, cumulative (value, size) per prefix length t = 0..len ---
    # value_t = total D-reduction vs "nothing in this chain inlined": overhead saved by
    # every inlined link, plus every bonus a FULL prefix (t links from the root) unlocks.
    chain_options = []  # list of (weight, value, path_prefix) tuples, one list per chain
    for path in chains:
        opts = [(0, 0, [])]  # t = 0: take nothing
        size_acc = 0
        value_acc = 0
        for pos, idx in enumerate(path):
            size_acc += inline_size[idx]
            value_acc += freq[idx] * CALL_OVERHEAD
            if pos > 0:  # bonus fires because the full prefix up to idx is inlined
                value_acc += freq[idx] * bonus[idx]
            opts.append((size_acc, value_acc, path[:pos + 1]))
        chain_options.append(opts)

    CAP = max(0, ICACHE_CAP - S_base)

    # --- grouped 0/1 knapsack DP over the icache budget: pick at most one prefix
    #     option per chain, maximize total value, weight <= CAP ---
    dp = [0] * (CAP + 1)
    choice = [[-1] * (CAP + 1) for _ in range(len(chain_options))]  # option index picked
    for gi, opts in enumerate(chain_options):
        new_dp = dp[:]
        for budget in range(CAP + 1):
            best_val = dp[budget]
            best_opt = -1
            for oi in range(1, len(opts)):
                w, v, _ = opts[oi]
                if w <= budget and dp[budget - w] + v > best_val:
                    best_val = dp[budget - w] + v
                    best_opt = oi
            if best_val > new_dp[budget]:
                new_dp[budget] = best_val
                choice[gi][budget] = best_opt
        dp = new_dp

    # recover best budget and the selection via backtracking
    best_budget = max(range(CAP + 1), key=lambda b: dp[b])
    selected = [0] * len(chain_options)  # option index per chain, -1/0 = none
    budget = best_budget
    for gi in range(len(chain_options) - 1, -1, -1):
        oi = choice[gi][budget]
        if oi is not None and oi != -1:
            w, v, _ = chain_options[gi][oi]
            selected[gi] = oi
            budget -= w
        else:
            selected[gi] = 0

    chosen = set()
    for gi, opts in enumerate(chain_options):
        oi = selected[gi]
        if oi > 0:
            for idx in opts[oi][2]:
                chosen.add(idx)

    # --- bounded, deterministic overshoot probe: for each chain not fully taken, try
    #     bumping it to its best remaining higher option (including its max option) and
    #     keep the move only if the REPLAYED F actually improves ---
    best_F, _ = evaluate(chosen, freq, base_cost, inline_size, parent, bonus, m,
                          S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF)
    for gi, opts in enumerate(chain_options):
        cur_oi = selected[gi]
        for oi in range(len(opts) - 1, cur_oi, -1):
            trial = set(chosen)
            for idx in opts[cur_oi][2]:
                trial.discard(idx)
            for idx in opts[oi][2]:
                trial.add(idx)
            trial_F, _ = evaluate(trial, freq, base_cost, inline_size, parent, bonus, m,
                                   S_base, ICACHE_CAP, CALL_OVERHEAD, PENALTY_COEF)
            if trial_F < best_F:
                best_F = trial_F
                chosen = trial
                selected[gi] = oi
            break  # only probe the single largest jump per chain (bounded work)

    print(len(chosen))
    print(" ".join(map(str, sorted(chosen))))


if __name__ == "__main__":
    main()
