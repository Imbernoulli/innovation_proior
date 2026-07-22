# TIER: strong
# The insight: since the whole process (graph + dynamics) is fully deterministic and known
# up front, the program can SIMULATE its own intervention forward round by round, and at
# each round immunize the RING JUST AHEAD OF THE FRONTIER -- the currently-susceptible
# nodes that are adjacent to an already-infectious node, i.e. exactly the nodes that would
# be infected next if left alone -- ranked by how much further exposure they would cause
# (their own count of still-susceptible neighbours), rather than by static whole-graph
# degree. This reallocates the scarce total budget to wherever the outbreak is ABOUT to go,
# cutting transmission paths before they are traversed instead of protecting the
# already-exposed core or an irrelevant, far-away hub.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
T = inst["T"]
D = inst["D"]
rate_cap = inst["rate_cap"]
total_budget = inst["total_budget"]

adj = [[] for _ in range(N)]
for a, b in inst["edges"]:
    adj[a].append(b)
    adj[b].append(a)

state = [0] * N          # 0=S 1=I 2=R 3=V
infect_round = [-1] * N
for s in inst["seeds"]:
    state[s] = 1
    infect_round[s] = 0

schedule = [[] for _ in range(T)]
spent = 0
for t in range(T):
    for i in range(N):
        if state[i] == 1 and infect_round[i] + D <= t:
            state[i] = 2

    infset = set(i for i in range(N) if state[i] == 1)

    frontier = []
    for u in range(N):
        if state[u] == 0:
            for v in adj[u]:
                if v in infset:
                    frontier.append(u)
                    break

    def future_exposure(u):
        return sum(1 for v in adj[u] if state[v] == 0)

    frontier.sort(key=lambda u: (-future_exposure(u), u))

    budget_left = total_budget - spent
    take = frontier[:max(0, min(rate_cap, budget_left))]
    schedule[t] = take
    for nid in take:
        state[nid] = 3
    spent += len(take)

    infset = set(i for i in range(N) if state[i] == 1)
    newly = []
    for u in range(N):
        if state[u] == 0:
            for v in adj[u]:
                if v in infset:
                    newly.append(u)
                    break
    for u in newly:
        state[u] = 1
        infect_round[u] = t

print(json.dumps({"schedule": schedule}))
