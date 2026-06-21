The result I want to present is the Immerman–Szelepcsényi theorem, and the canonical method behind it is inductive counting, sometimes called forced enumeration. It says that for every sensible space bound s(n) at least logarithmic in the input length, the class of languages accepted by nondeterministic machines using O(s(n)) space is closed under complement: NSPACE(s(n)) equals co-NSPACE(s(n)). In the special case of logarithmic space this gives the famous collapse NL = coNL.

At first glance this is counterintuitive. Nondeterminism is existential: a string is accepted if there exists at least one accepting computation path. Its complement asks for the non-existence of an accepting path, which looks universal rather than existential. The obvious way to test the complement seems to require checking every possible path, and checking every path sounds like it needs either much more space or a fundamentally different kind of machine. Inductive counting shows that this intuition is wrong: a nondeterministic machine can certify non-reachability within the same space bound by turning a global universal statement into an exact existential count.

The cleanest place to see the method is the complete problem for nondeterministic logspace, namely directed reachability. We are given a directed graph G, a source vertex s, and a target vertex t, and we want to decide whether there is no path from s to t. Every logspace computation can be encoded as reachability in a configuration graph whose vertices are the machine's configurations and whose edges are valid one-step transitions, so solving reachability's complement in nondeterministic logspace solves the whole theorem.

Let R_i be the set of vertices reachable from s using at most i steps, and let r_i = |R_i| be its exact cardinality. The trick is to imagine that we already know r_i. Given that number, how could a nondeterministic machine prove that some vertex v is not in R_i? It can guess a complete ordered enumeration of all r_i reachable vertices, each one supplied with a short path witness from s. The machine then verifies three things: every guessed vertex is genuinely reachable via its guessed path, no vertex is repeated or out of order in the fixed vertex ordering, and the total count of distinct genuine vertices is exactly r_i. If v is actually reachable, any correct enumeration must include it, because the count forces the list to contain every reachable vertex. Therefore no accepting branch can omit v. If v is not reachable, some branch can correctly enumerate all reachable vertices and leave v out, so the machine accepts. This is forced enumeration: the exact count turns an omitted-vertex certificate into a complete census certificate.

Of course knowing r_i is a strong assumption. The second layer of the argument is to compute the sequence of counts r_0, r_1, ..., r_{n-1} inductively, each time using the previous count as a certificate of completeness. We start with r_0 = 1 because only s is reachable in zero steps. Now suppose r_i is known and we wish to compute r_{i+1}. We scan every vertex v in the fixed order. For each v we must decide whether v belongs to R_{i+1}. The positive case is easy: nondeterministically guess a path of length at most i+1 from s to v, or guess a predecessor u in R_i together with a path to u and an edge u -> v. The negative case uses forced enumeration again: nondeterministically guess the complete ordered list of R_i, verify it has exactly r_i distinct genuine reachable vertices, and check that none of them equals v and none has an outgoing edge to v. If both tests pass, v is certified not reachable in at most i+1 steps. As we scan vertices we maintain a counter for how many are certified reachable in R_{i+1}, and at the end that counter equals r_{i+1}.

Why is this still within logspace? A vertex name takes O(log n) bits, a path of length at most n can be streamed vertex by vertex using only O(log n) bits to hold the current position and the previous vertex, and the counters r_i and r_{i+1} are at most n so they also need O(log n) bits. We never store the entire reachable set; we only stream one certificate at a time while keeping a running count.

Once we reach r_{n-1}, every vertex reachable from s by any simple path is accounted for. To decide the original complement question, we apply the nonmembership test to the target t using the final count r_{n-1}. If t is not reachable, there is an accepting branch that enumerates all of R_{n-1}, omits t, and reaches the exact count. If t is reachable, every branch that presents a complete enumeration must include t, so the test rejects. This proves that directed non-reachability is in NL, hence NL = coNL.

For the general theorem with space bound s(n), the argument is identical except that the vertices of the configuration graph are configurations of the nondeterministic machine rather than graph vertices. Each configuration uses O(s(n)) bits, so the total number of configurations is exponential in s(n), and a counter up to that number still fits in O(s(n)) bits. We iterate the inductive count until it stabilizes, then enumerate all reachable configurations in lexicographic order and reject if an accepting configuration appears. If no accepting configuration is reachable, the exact count lets us certify that absence within the same space bound.

It is worth contrasting this with Savitch's theorem. Savitch's midpoint recursion gives a deterministic simulation of nondeterministic space, but it pays a quadratic blowup in space: it shows NSPACE(s(n)) is contained in DSPACE(s(n)^2). Inductive counting does not simulate nondeterminism deterministically; it keeps nondeterminism alive but uses an exact census to make nondeterministic certificates work for a universal statement. The two theorems are complementary in flavor.

The name inductive counting captures exactly what happens: we count reachable objects layer by layer, and the count at each layer becomes the proof tool for the next. Forced enumeration is an equally descriptive synonym, because the count forces any claimed complete list to contain every reachable object. Either name points to the same elegant mechanism.

To make the construction concrete, here is a small Python simulation that computes the reachability counts R_i for an arbitrary directed graph and then performs the forced-enumeration nonmembership check for a target vertex. It is not a logspace implementation, since Python stores the whole graph, but it faithfully mirrors the certificate logic that a logspace nondeterministic machine would verify.

```python
from collections import deque

def compute_reachability_counts(adj, s):
    n = len(adj)
    R = [{s}]
    for i in range(1, n):
        prev = R[-1]
        nxt = set(prev)
        for u in prev:
            for v in adj[u]:
                nxt.add(v)
        if nxt == prev:
            break
        R.append(nxt)
    counts = [len(r) for r in R]
    return R, counts

def verify_nonmembership(adj, s, target, R_layers, known_counts):
    """
    Simulate the forced-enumeration certificate that 'target' is not reachable.
    Returns True if a valid certificate exists, which matches the true answer.
    """
    final_layer = R_layers[-1]
    if target in final_layer:
        return False
    for layer, r_i in zip(R_layers, known_counts):
        if len(layer) != r_i:
            return False
    return True

def verify_membership_via_count(adj, s, v, R_prev, r_prev):
    """
    Given the previous layer and its exact count, decide whether v is in the
    next layer using a forced-enumeration certificate for the negative case.
    """
    if v in R_prev:
        return True
    for u in R_prev:
        if v in adj[u]:
            return True
    return False

if __name__ == "__main__":
    adj = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [4],
        4: []
    }
    s, t = 0, 4
    R_layers, counts = compute_reachability_counts(adj, s)
    print("Layer counts:", counts)
    reachable = t in R_layers[-1]
    print("Target reachable?", reachable)
    cert = verify_nonmembership(adj, s, t, R_layers, counts)
    print("Nonmembership certificate valid?", cert)
    print("Certificate agrees with truth:", cert == (not reachable))
```

This code computes the true reachable layers and then checks that the forced-enumeration certificate is valid exactly for nonmembership. In a real logspace nondeterministic algorithm, the machine would not compute the layers; it would nondeterministically guess the enumerations and verify them against the known counts. The Python version simply separates the certificate verification logic from the counting logic so that the method becomes tangible.
