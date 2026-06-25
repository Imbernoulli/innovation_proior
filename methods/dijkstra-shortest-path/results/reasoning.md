I am sitting on the café terrace, tired from shopping, with coffee in front of me, so let me just think about whether I can actually do this. The ARMAC needs a demonstration. For non-computing people the problem has to be one they can state and one whose answer they can check — so: the shortest way to travel from Rotterdam to Groningen, on a road map. Everyone understands "what's the shortest route." Good. Now can I find that route on the machine? I have no pencil, no paper, only the cup in front of me and what I can hold in my head, which is probably for the best — if I can't keep it simple enough to carry in my head, it won't be simple enough to be a clean demo. So I want the simplest thing that works, nothing avoidable.

Let me set up the picture. Cities are points; roads are segments between them, each with a length; lengths are distances on a map, so they're never negative, and a road might be one-way so the length out of a city need not equal the length back in. I'm given a source P (Rotterdam) and a target Q (Groningen), and I want the minimum total length over all routes from P to Q. The map is reduced to 64 cities — six bits names a city — and the machine has almost no store, so whatever I do I must *not* need the whole branch list sitting in memory at once. That last constraint is going to do a lot of the design for me.

The brute thing — enumerate every route from P to Q and take the shortest — is obviously hopeless; there are far too many routes, and laypeople saying "obviously there's a shortest one" is exactly the trap: existence is trivial, *finding it cheaply* is the whole job. So I need structure.

Suppose I somehow know the genuine shortest route from Rotterdam to Groningen, and suppose some city R sits on it — say Utrecht is on the way. Then the piece of that route from Rotterdam to Utrecht must itself be the shortest route from Rotterdam to Utrecht. Because if there is a shorter way to get from Rotterdam to Utrecht, I can splice it in and get a shorter Rotterdam–Groningen route, contradicting that I started with the shortest. So a shortest route is made of shortest routes to its own intermediate cities. Knowing the shortest path to Q implies knowing the shortest path to every R along it.

That suggests I don't aim straight at Groningen at all. If I want the shortest route to Groningen, I am forced to know the shortest route to whatever city precedes it, and to that city's predecessor, and so on back to Rotterdam — the substructure pulls the whole problem backward toward the source. So maybe the natural direction is the reverse: grow knowledge of shortest routes from Rotterdam *outward* — to nearby cities first, then farther ones — and let Groningen's answer simply appear when I reach it. The nearest city to Rotterdam I ought to be able to nail down; then the next nearest; and so on, until Groningen's turn comes. Whether I'm actually *allowed* to nail them down one at a time is the thing I have to check, but the order — increasing distance from the source — at least matches the substructure, which only ever points to closer cities.

Why would I be allowed to "nail down" a city — declare its distance final and never revisit it? That's the crux, and it's where I have to be careful, because if I settle a city too early and a cheaper way to it turns up later, the whole thing is wrong. Let me think about what "settle the nearest" actually buys me.

Picture the cities split into two groups. Group A: cities whose true shortest distance from Rotterdam I already know for certain. At the start that's just Rotterdam itself, distance 0. Everything else is unknown. Now, every city not in A that's reachable by a single road from some city in A — call those the *frontier*. For each frontier city v I can compute a *tentative* distance: the smallest value of "known shortest distance to an A-city x plus the road length from x to v" over all roads crossing from A into v. It is not just a number; it is an actual route, because the route to x is already known and then I append that one road. It is tentative because maybe a better route exists that goes deeper before coming back out — I haven't explored everything yet.

Among all the frontier cities, look at the one with the *smallest* tentative distance; call it u and call that distance t(u). I need to prove that t(u) is not merely the best route I have seen so far but the true shortest route to u. One inequality is immediate: t(u) is the length of a real route to u, so the true shortest distance to u is at most t(u). The dangerous direction is the other one. Take any route from Rotterdam to u. Since Rotterdam is in A and u is outside A before I settle it, the route has a first city y outside A; just before y it is at some city x in A. Let the length of the route's prefix to x be L. Because x is already in A, the shortest distance d(x) is known, and no actual prefix to x can be shorter than d(x), so L ≥ d(x). After the road x to y, that prefix has length L + length(x, y) ≥ d(x) + length(x, y). But t(y) is the smallest value of d(a) + length(a, y) over all roads from any settled city a into y, so d(x) + length(x, y) ≥ t(y). Therefore the route has already cost at least t(y) by the time it first reaches y. Since u is the frontier city with minimum tentative distance, t(y) ≥ t(u). From y onward to u, all remaining road lengths are nonnegative, so the suffix cannot reduce the length. Therefore every route to u has length at least t(u). Together with the real route of length t(u), that pins the true shortest distance to u exactly. Now I can settle u.

Let me hold that thought up to the light, because it's the entire correctness of the method and it rests on one fact: lengths are nonnegative. If a road could have negative length, then "the rest of the trip only adds length" would be false — a faraway detour could come back and *reduce* the total, and a city I settled as nearest might actually be reachable more cheaply through some not-yet-explored loop. Then settling-the-nearest would be wrong. But these are road distances on a map. Distances are nonnegative, full stop; the question never even comes up for the demo. So I get to lean on it completely, and I will. (That's the kind of avoidable complexity I'm not going to invent: no negative roads exist here, so I won't carry machinery to handle them.)

So the loop is taking shape. Start with A = {Rotterdam}, distance 0. Whenever a city enters A, inspect the roads out of that newly settled city and refresh the frontier candidates they create. Then pick the frontier city with the smallest tentative distance, move it into A with that distance as final, and repeat. Stop when Groningen enters A.

Now the part the small machine forces me to get right: how much do I actually have to *store*? The lazy thing would be to keep, for every city, a full record of how to reach it, or to keep all the branch data in memory and rescan it. I can't — there isn't room, and anyway it's more than I need. Let me ask what's truly necessary.

For a frontier city, do I need to remember *all* the roads coming into it from A? No. All I care about is its current best tentative distance, and which single road achieves it. If two A-cities both have roads into the same frontier city, only the cheaper of the two combined distances can ever matter — the other can be thrown away immediately, because it'll never give a shorter route. So per frontier city I keep exactly one candidate road: the best one found so far. One branch per frontier city. The frontier has fewer cities than the whole map, so that's a small set — far less than storing every road.

And for a city already in A — settled — I keep the one road by which its shortest route finally enters it (its predecessor edge), so I can trace the route back at the end, plus its distance. That's it. In the mathematical bookkeeping, at any instant I'm holding: the known-route branches (one per settled city) and the candidate branches (one per frontier city). The number of live branches is bounded by the number of cities — always less than the full map, never the full edge list. The frugality falls straight out of "only the best candidate per frontier city can matter."

Compare that to scanning *all* the roads over and over looking for any that can be improved — relax the whole edge set, repeat until nothing changes. That works and it's simple to state, but it stores and rescans every branch and corrects a city's distance many times before it stabilizes, with no notion of a city ever being *done*. My settle-the-nearest order gives me, for free, the thing that method lacks: once a city is in A it is finished, never touched again, and I only ever keep a candidate per frontier city rather than the whole branch list. Fewer branches in store, and less work, because I never reprocess a settled city.

When a city is newly settled, the only new certified information enters through that city. Say I just move city u into A with its final distance. The only candidate distances I can now trust that I could not trust before are the ones that append a single outgoing road to the now-known route to u. So I look at the roads leading out of u to cities not yet settled. For each such road, to a city v: the distance "u's final distance plus this road's length" is a candidate route to v. If v is already on the frontier, I compare this candidate against v's current best tentative distance and keep whichever is smaller (and remember the road that gives it). If v is not on the frontier yet — it is off in the unexplored region — then this road is the first way I've found to reach it, so it joins the frontier with this candidate distance. Cities I never reach this way just stay unexplored; I never have to touch them. I only need the outgoing roads of the *one* city I just settled; the live search state is the settled routes plus the single best candidate for each frontier city, not the whole branch list.

I think that's the whole thing, but I have argued myself into it and I do not entirely trust an argument I have only argued. Let me actually run the order on a small map and watch the numbers, because the place this could break is the "settle the nearest, never revisit" step — if I can find one settled city whose distance later turns out wrong, the method is dead. Take Rotterdam with roads to Utrecht (57) and Amsterdam (78); Utrecht to Zwolle (90) and back to Amsterdam (40); Amsterdam to Utrecht (40) and to Zwolle (112); Zwolle to Groningen (100). This is small enough to carry but it has the feature I want to stress: two different cities (Amsterdam and Utrecht) both feed Zwolle, so Zwolle's tentative distance is a genuine competition, and there's a back-road Amsterdam→Utrecht that could in principle undercut the direct Rotterdam→Utrecht.

Settle Rotterdam at 0. Its roads put Utrecht on the frontier at 57 and Amsterdam at 78. The smaller is Utrecht at 57, so Utrecht settles — final, I'm claiming. Now, is that safe? The only other way into Utrecht is Rotterdam→Amsterdam→Utrecht, which already costs 78 just to reach Amsterdam, more than 57, so no route through Amsterdam can beat the direct one. Good — the nonnegativity argument predicted exactly this, and here it is concretely: the rival route is already over budget before it even turns toward Utrecht. Utrecht settling stirs its outgoing roads: Zwolle joins the frontier at 57+90 = 147, and Utrecht→Amsterdam offers Amsterdam 57+40 = 97, which is *worse* than Amsterdam's standing 78, so I discard it and keep Amsterdam at 78. (There — a candidate computed and thrown away because it didn't improve the best-per-city, which is the store discipline doing its job.) Frontier now: Amsterdam 78, Zwolle 147. Settle Amsterdam at 78. Amsterdam's roads: to Utrecht (already settled, skip) and to Zwolle at 78+112 = 190, which is worse than Zwolle's standing 147, so discard. Frontier: Zwolle 147. Settle Zwolle at 147; its road gives Groningen 147+100 = 247. Settle Groningen at 247, and the recorded predecessors trace Groningen←Zwolle←Utrecht←Rotterdam.

So the method returns 247 along Rotterdam→Utrecht→Zwolle→Groningen. Now I want to know if that's actually the optimum, independent of my own machinery, so I'll just enumerate the routes by hand. Rotterdam→Utrecht→Zwolle→Groningen = 57+90+100 = 247. Rotterdam→Amsterdam→Zwolle→Groningen = 78+112+100 = 290. Rotterdam→Amsterdam→Utrecht→Zwolle→Groningen = 78+40+90+100 = 308. Those are all the simple routes that reach Groningen, and 247 is the smallest. It matches. And notice *where* the method saved work: it settled Utrecht permanently at step two and never reconsidered it, even though the Amsterdam→Utrecht road existed; the by-hand check confirms that road never helps, so settling early lost nothing. Each settling stayed permanent, each settling only stirred the roads out of that one city, and the answer is right. That is the evidence I wanted before trusting the loop.

It doesn't need the full branch list in memory, doesn't sort all the roads up front, and doesn't revisit a finished city. That's about as few moving parts as I can manage — which is exactly the point: with no paper I have to keep it small enough to hold, and small enough to hold has turned out to be small enough to be efficient.

One more look at "pick the smallest tentative distance on the frontier," because that's the operation that repeats. On 64 cities I can just scan the frontier each time and take the minimum; that is only about V choices per settled city, so the simple version is O(V²) plus the road inspections, and it keeps the one-candidate-per-frontier-city store discipline exactly. But the clean way to say what that step *is* in a program is repeatedly extracting the minimum from a changing collection of (distance, city) pairs, and inserting a new key when a better candidate is found. That's a min-priority-queue. I keep the best tentative distance and predecessor for each city in ordinary tables; the heap is only the device for pulling out the small key.

The awkward part is what happens when a candidate *improves*. In my Rotterdam map nothing ever did — every frontier city's tentative distance was set once and never beaten, so a plain heap would have behaved perfectly and I'd have learned nothing about the hard case. Let me deliberately build one where a distance does improve, so I can see what the heap does wrong. Three cities: A→B costs 10, A→C costs 1, C→B costs 2. Settle A at 0; push B at 10 and C at 1. The frontier holds (1,C) and (10,B). Settle the smaller, C at 1. C's road to B offers 1+2 = 3, which beats B's standing 10, so I update B's best to 3 and its predecessor to C. But the heap still physically holds the old (10,B) — a heap has no cheap way to reach in and rewrite that entry. So now the heap contains *two* entries for B: the fresh (3,B) and the stale (10,B). If I trust the heap blindly I'll eventually pop (10,B) and report B at 10, which is wrong; the true answer is A→C→B = 3.

So I can't trust the bare pop. The fix that costs nothing: when I pop a (dist, city), I check dist against the best tentative distance I've recorded for that city. Pop (3,B): 3 equals B's best, so it's the live entry — settle B at 3. The stale (10,B) is still down in the heap; when it surfaces, 10 is greater than B's recorded best of 3, so I throw it away unexamined. And if a city has already been settled, I skip it too, which catches the same staleness from the other side. So the lazy heap may carry duplicate copies, but the recorded best-per-city table is the real frontier, and the comparison filters the junk. I just ran this case through the loop and it returns B at 3 via A→C→B, with the (10,B) entry correctly discarded — so the lazy scheme really does give the right answer without any decrease-key surgery.

Counting that way: each outgoing road of a settled city is inspected once, and each successful improvement pushes one heap entry, so the number of heap entries is bounded by E, and the lazy-heap code takes O((E + V) log(E + 1)) time using O(E) heap slots. On a simple graph log(E + 1) is O(log V), so this is usually written O((E + V) log V), or O(E log V) when the edge term dominates. A heap with true decrease-key would keep the queue itself to O(V) entries, but I do not need that complication for the code — the comparison-on-pop is cheaper to get right.

Let me write it down as it would actually run.

```python
import heapq

# outgoing_roads(city) -> iterable of (neighbour, road_length).
# Road lengths are nonnegative; roads may be one-way.

def shortest_path(outgoing_roads, start, end):
    # frontier keys; best/predecessor hold the single live candidate per city.
    heap = [(0, start)]
    best = {start: 0}
    predecessor = {start: None}
    settled = set()          # cities whose shortest distance is now final

    while heap:
        dist, u = heapq.heappop(heap)
        if dist > best.get(u, float("inf")):
            # stale key left over from an older, worse route
            continue
        if u in settled:
            continue
        settled.add(u)        # u joins A; dist is its true shortest distance

        if u == end:
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = predecessor[node]
            path.reverse()
            return dist, path

        # Only routes through the newly settled city can improve the frontier.
        for v, length in outgoing_roads(u):
            if v in settled:
                continue
            candidate = dist + length
            if candidate < best.get(v, float("inf")):
                best[v] = candidate
                predecessor[v] = u
                heapq.heappush(heap, (candidate, v))

    return None


if __name__ == "__main__":
    road_map = {
        "Rotterdam": [("Utrecht", 57), ("Amsterdam", 78)],
        "Amsterdam": [("Utrecht", 40), ("Zwolle", 112)],
        "Utrecht":   [("Zwolle", 90), ("Amsterdam", 40)],
        "Zwolle":    [("Groningen", 100)],
        "Groningen": [],
    }

    def outgoing_roads(city):
        return road_map.get(city, ())

    print(shortest_path(outgoing_roads, "Rotterdam", "Groningen"))
```

Looking back over how it came together: the demo had to be a problem laypeople grasp and the machine can run, which forced shortest road route, tiny map, no room for the whole branch list. The substructure fact — a shortest route is built from shortest routes to its intermediate cities — kept pointing back toward the source, which is what made growing the distances *outward* in increasing order the natural move rather than aiming at the target. The risky step was declaring the nearest frontier city final, and the nonnegativity argument is what licenses it; when I ran the Rotterdam map by hand, settling Utrecht early really did cost nothing and the by-hand enumeration agreed at 247, so I believe the argument rather than just having written it. The frugality came from noticing only the single best candidate per frontier city can matter, so the store is one branch per frontier city and one per settled city, and each settling stirs only the roads out of that one city. Reading "pick the nearest" as extract-min and "found a cheaper way" as an improved key let a plain min-heap stand in for the frontier scan — once I checked the three-city case, the lazy skip-on-pop handled the stale keys that improvement leaves behind. What I did not expect at the café is that the version small enough to hold in my head with no pencil would turn out to be the efficient one too; but that is how it came out.
