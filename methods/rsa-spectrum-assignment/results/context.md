# Context: allocating spectrum in a flexible-grid optical network

## Research question

A backbone optical network is a mesh of nodes joined by single-mode fibre links. Each fibre carries an optical spectrum a few terahertz wide, and that spectrum is the scarce resource: every end-to-end connection ("lightpath") that the network carries must be given a slice of it on every link along its route. The job is to take a set of traffic demands — each a source, a destination, and a requested bit-rate — and decide, for each demand, *which route* it takes and *which part of the spectrum* it occupies, so that all demands fit and the total spectrum the network must light up is as small as possible.

In the older fixed-grid world this is the Routing-and-Wavelength-Assignment problem: the spectrum is pre-carved into fixed-width wavelength channels (a 50 GHz ITU grid), every demand needs exactly one wavelength, and the same wavelength must be free on every link of its route because there are no wavelength converters in the line. The flexible-grid (elastic) world breaks the fixed channelisation: the spectrum is instead sliced into many fine frequency slots (12.5 GHz each), and a demand is given *as many adjacent slots as its bit-rate needs* — a 400 Gb/s demand might take eight slots, a 100 Gb/s demand two. The promise is much finer-grained, more efficient packing; the cost is a far harder allocation problem, because now the unit of allocation is a *variable-width contiguous block* of slots rather than a single wavelength, and the network must track the spectral position and width of every signal.

What a solution must achieve: given the topology, the per-link spectral width (a number of slots), and the demand matrix, produce a route and a block of contiguous slots for every demand such that (i) each demand's block is contiguous, (ii) it occupies the identical slots on every link of its route, (iii) demands that share a link never overlap in the spectrum, and (iv) adjacent blocks on a link are separated by a small guard band. The figure of merit is how much spectrum gets used — the highest slot index any link is forced to occupy. The problem must be solvable exactly for small planning instances and by fast heuristics at network scale.

## Background

**The fixed grid and wavelength routing.** In a wavelength-division-multiplexed (WDM) network the fibre spectrum is divided into a fixed set of wavelength channels of equal width. A lightpath is an all-optical channel from source to destination occupying one wavelength on each link of its physical route. Because the line has no wavelength converters, the *wavelength-continuity constraint* holds: a lightpath must use the same wavelength on every hop. Two lightpaths sharing a fibre must use different wavelengths. Establishing a set of lightpaths is the routing-and-wavelength-assignment (RWA) problem: pick a route and a continuous wavelength for each demand. Wavelength assignment alone — routes given — is a graph-colouring problem: build a conflict graph whose vertices are lightpaths and whose edges join lightpaths that share a fibre, and colour it with as few wavelengths (colours) as possible. RWA is NP-hard in general; on a single path the wavelength-assignment subproblem is solvable in polynomial time (interval graphs are perfectly colourable).

**The elastic grid.** A flex-grid (elastic) optical network replaces the coarse fixed channels with a fine slot grid. The ITU flex-grid fixes a reference at 193.1 THz and lets a channel occupy `12.5 GHz × m` of spectrum for a positive integer `m`; the basic unit, the frequency slot, is 12.5 GHz wide. Bandwidth-variable transponders generate a signal of just the width a demand needs by grouping the right number of adjacent slots, and bandwidth-variable wavelength-selective switches at the nodes pass that exact width through. So a demand no longer maps to one wavelength but to a *block of `n` adjacent slots*, with `n` set by its bit-rate and the modulation format it uses.

**Distance-adaptive modulation.** A transponder can trade spectral efficiency against reach by choosing its modulation format. A high-order format (16-QAM, four bits per symbol per polarisation) packs the bit-rate into few slots but needs a high signal-to-noise margin and so only reaches short distances; a robust format (BPSK, one bit per symbol) reaches far but spends many slots. The number of slots a demand needs is `n = ceil(rate / (efficiency × slot_bandwidth))`, and the choice of modulation is constrained by whether the format's reach covers the demand's route length. Whether a given modulation actually survives a given route — the optical-signal-to-noise-ratio left after amplifier noise and Kerr nonlinear interference along that path — is a physical-layer question answered by an analytic noise model of the fibre (the Gaussian-noise model of nonlinear propagation); that model supplies the reach/feasibility input. Here that input is taken as a modulation-to-reach table: the noise computation is upstream and is not redone at the allocation layer.

**The two new constraints.** The fine grid imposes two constraints absent from single-wavelength routing. *Spectrum contiguity*: if a demand needs `t` units of spectrum, the `t` slots assigned to it must be consecutive — a single interval, not `t` scattered slots — because one transponder emits one continuous band. *Spectrum continuity*: those same `t` contiguous slots must be free and assigned on every link along the demand's route, the direct analogue of wavelength continuity, again because there is no spectrum conversion in the line. Contiguity is the genuinely new difficulty: in plain colouring any free colour will do, but here the `t` colours must be adjacent *and* identical across all the route's links. The visible symptom is fragmentation — a link may have many free slots in total yet no contiguous aligned block large enough for an incoming demand, so the demand is blocked even though capacity exists.

**Guard bands.** The filters in a bandwidth-variable switch do not have brick-wall edges; their roll-off and the optical crosstalk between neighbouring channels mean two adjacent demands on a link cannot be packed slot-against-slot. A small guard band of empty slots — commonly one or two — must separate adjacent occupied blocks so the filters can isolate each signal.

**Why this is computationally hard.** The spectrum-assignment subproblem (routes fixed) can be read as a scheduling problem on parallel machines: a slot index plays the role of time, a demand is a task needing a contiguous time block of length equal to its slot count, and the machines it must run on simultaneously are the links of its route — the `P|fix_j|C_max` problem of scheduling tasks each pinned to a fixed subset of machines, minimising the makespan `C_max = max_j C_j`. The three-machine unit-time-free version `P3|fix_j|C_max` is strongly NP-hard, and through this correspondence the spectrum-assignment problem is NP-hard on a unidirectional ring of three links and on any path with four or more links, while it is solvable in polynomial time on paths of at most three links. The same correspondence to interval-graph colouring gives a `(2+ε)`-approximation via the interval chromatic number on a path, and `(4+2ε)` on a ring. So even with routes fixed the assignment is intractable in general, and adding the routing choice only enlarges the search — yet RWA, the special case where every width is one slot, is itself NP-hard, so the elastic problem inherits hardness from below as well.

## Baselines

**Routing-and-wavelength assignment (RWA), fixed grid.** Pick a route and one continuous wavelength per demand; assignment with routes fixed is graph colouring on the lightpath conflict graph. Core algorithm: route on `k` shortest paths, colour greedily (First-Fit: lowest-indexed free wavelength). Gap for the elastic problem: it allocates exactly one channel of fixed width per demand, so it cannot express variable-width demands, has no notion of a *contiguous block*, and therefore cannot capture the contiguity constraint or the fragmentation it causes. It is the special case `width = 1`, not a solution to the general problem.

**Graph / interval colouring for spectrum assignment.** Treat each demand as an interval to be coloured, where a "colour" is now a band of `t` adjacent slots. With routes fixed on a path, the assignment is the interval-chromatic-number problem and admits a `(2+ε)` approximation. Core idea: order the intervals and pack them. Gap: this captures contiguity on a single path but does not by itself enforce continuity across a multi-link route, nor the non-overlap of demands sharing arbitrary links in a mesh, and the general mesh problem with four-plus-link paths is NP-hard, so the path-only colouring guarantee does not transfer.

**Greedy spectrum-allocation policies (routes given).** With a route chosen, a policy decides which free contiguous block to take. First-Fit scans slot indices from zero and takes the lowest-indexed block of the required width that is free on every link of the route; it needs no global state, packs demands toward index zero, and leaves a long contiguous free tail. Random-Fit takes a random feasible block; Last-Fit takes the highest-indexed; Exact-Fit takes the smallest free gap that exactly fits (to limit fragmentation); Most-Used takes the slot index that is already occupied on the most fibres network-wide, to concentrate reuse and keep other indices clean (it needs global state and costs more to compute); Least-Used spreads load the opposite way. Gap: each is a fast online rule with no optimality guarantee; they react to the demand order and routing they are handed, and decoupling routing from assignment can route a demand onto links that then force a high slot index.

**Joint and decomposed integer programs.** The exact route is to write an integer linear program. A *link-based* multicommodity-flow ILP assigns slots to demands subject to contiguity, continuity, non-overlap, and guard constraints, minimising the maximum slot index (or maximum slots per fibre, or total slots). It is exact but the explicit contiguity constraints inflate it badly, so it solves only for very small topologies. A *path-based* ILP pre-computes `k` candidate paths per demand and selects one path plus a slot block per demand, which makes continuity implicit along the chosen path but still carries explicit contiguity constraints. The decomposed `R+SA` route solves routing first (choose a path per demand minimising max link load) then spectrum assignment second (assign blocks minimising max slot index) as two smaller ILPs — compact and scalable, but the sequential split is not guaranteed optimal for the joint problem. Gap: the joint ILP is intractable past a handful of nodes; the decompositions trade optimality for size.

## Evaluation settings

The natural test instances are standard backbone topologies — a 14-node, 21-link NSFNET, or small rings and meshes — with a per-fibre spectral width given as a fixed number of 12.5 GHz slots. A demand set is a list of (source, destination, bit-rate) triples, either a static traffic matrix for the offline planning problem or a stream of randomly arriving and departing requests for the online problem. A modulation table maps (format → spectral efficiency, reach) so that a route length determines the slot count of each demand; the guard band is fixed at one or two slots. For the offline problem the metric is the spectrum used — the maximum occupied slot index over all fibres, or equivalently the total slots, with a route and contiguous block reported per demand. For the online problem the metric is blocking probability — the fraction of requests that arrive and cannot be given a contiguous, continuous, non-overlapping block. Runtime/scalability (largest topology solvable exactly; heuristic speed) is the secondary axis.

## Code framework

The primitives that already exist: a graph library for the topology and `k`-shortest-paths, a per-link occupancy record, and a general-purpose ILP solver. The pieces the method must supply are the slot-count rule, the constraint-respecting block search, and the exact program; they are left as stubs below.

```python
from math import ceil
import networkx as nx

SLOT_GHZ = 12.5      # frequency-slot width
GUARD = 1            # guard-band slots between adjacent blocks on a link
NUM_SLOTS = 40       # spectral width per link

# Distance-adaptive modulation table: (name, bits/s/Hz, reach_km).
# reach is the demand-feasibility input supplied by the physical-layer
# noise model; treated here as a fixed table.
MODULATIONS = [
    ("16-QAM", 4.0, 500),
    ("8-QAM",  3.0, 1000),
    ("QPSK",   2.0, 2000),
    ("BPSK",   1.0, 4000),
]

def k_shortest_paths(G, s, t, k=3):
    out = []
    for i, p in enumerate(nx.shortest_simple_paths(G, s, t, weight="km")):
        if i >= k:
            break
        out.append(p)
    return out

def path_links(path):
    return [(min(u, v), max(u, v)) for u, v in zip(path, path[1:])]

def path_len(G, path):
    return sum(G[u][v]["km"] for u, v in zip(path, path[1:]))

def demand_slots(rate_gbps, path_len_km):
    # TODO: pick the most efficient modulation whose reach >= path length,
    # then return ceil(rate / (efficiency * slot_bandwidth)).
    pass

def free_block(occ, links, start, block):
    # TODO: is [start, start+block) free on EVERY link of the route?
    # (continuity across links + non-overlap with what's already placed)
    pass

def pick_start(occ, links, block):
    # TODO: the spectrum-allocation rule -- which feasible contiguous
    # start index to take (e.g. the lowest one).
    pass

def heuristic_rsa(G, demands, k=3):
    # TODO: order demands, route on k paths, place a contiguous block per
    # demand respecting all constraints; report the max occupied slot index.
    pass

def exact_rsa(G, demands, k=3):
    # TODO: integer program -- one route+block per demand, no overlaps on
    # shared links incl. guard, minimise the maximum occupied slot index.
    pass
```
