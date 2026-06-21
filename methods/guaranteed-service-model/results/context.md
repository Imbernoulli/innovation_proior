# Context: strategic safety-stock placement in a multi-echelon supply chain

## Research question

A product flows through a network of stages — raw-material procurement, component fabrication, subassembly, final assembly, distribution centers, retail — and each stage is a potential place to hold inventory. End-customer demand is uncertain. The question is strategic, not operational: across the *whole* network, **where should safety stock be positioned, and how much at each location**, so that the final customer is served at a high level while the total inventory holding cost is minimized?

The stages are coupled. If a stage carries little safety stock and instead leans on its supplier, then the time it is exposed to demand variability — and so the stock it needs — depends on how fast that supplier can replenish it, which depends in turn on how much safety stock the supplier carries, and so on up the chain. A decision at one stage changes the inventory requirement at every adjacent stage. The setting calls for setting the inventory at all stages *jointly*, respecting these inter-stage dependencies, and fast enough to run on supply chains with tens of stages — as a planning tool a cross-functional team can re-run under different scenarios.

## Background

**The single-stage building block.** For one location with a periodic-review base-stock policy: each period you observe demand and order back up to a target level $B$ (the base-stock / order-up-to level). With no ordering delay, every order placed on the supplier exactly reflects observed demand. If replenishment lead time is $L$ and demand over any $L$-period window is bounded by $D(L)$, then the least base stock that guarantees service is $B=D(L)$. For the normal-style bound $D(L)=\mu L+z\,\sigma\sqrt{L}$, the expected safety stock is $D(L)-\mu L=z\,\sigma\sqrt{L}$. This formula is exact for one guaranteed-service stage.

**Two ways to extend to many echelons.** By the late 1990s the multi-echelon literature had split into two philosophies for how a downstream stage experiences its supplier.

*Stochastic-service (Clark & Scarf 1960).* The upstream stage may itself stock out, so the time a downstream stage waits for a replenishment is a *random variable*. Clark & Scarf introduced **echelon inventory** (the stock at a stage plus everything downstream of it) and showed that a serial system decomposes exactly into single-stage problems linked by induced penalty-cost functions, giving optimal echelon base-stock levels. It makes no bounded-demand assumption and models how stockouts propagate. The induced cost functions and the random delays couple the stages; the formulation is developed for serial (and, via Rosling 1989, assembly) systems. Service times are *outcomes* of the policy.

*Guaranteed-service (Simpson 1958).* Simpson studied the simplest network — a serial production line — and posed a different question: should two adjacent operations be coupled or separated by an inventory buffer? In his framing each stage *promises* a service time to its customer and *always* honors it, holding enough stock to do so. He observed that the optimal safety-stock placement has an **all-or-nothing** character: a stage either holds enough safety stock to fully decouple itself from downstream (and quote service time zero), or it holds no safety stock at all (and simply passes its replenishment time through to its customer). Kimball (1988) gives the general single-stage base-stock principles in the same spirit.

**The pieces already assembled along the guaranteed-service line.** Graves (1988) observed that for a serial line the optimal placement can be found as a **shortest path**. Inderfurth (1991, 1993), Inderfurth & Minner (1998), and Minner (1997) extended the dynamic-programming idea to **assembly** networks (many suppliers feeding one stage) and **distribution** networks (one stage feeding many). Graves & Willems (1996) developed comparable results for assembly and distribution. So by 1998 the guaranteed-service framework had efficient algorithms for serial, assembly, and distribution structures separately.

**The motivating observation from practice.** In real product flows (e.g., the assembly of a consumer electronics product with tens of components, several internal subassemblies, a distribution center, and shipment to customers) firms tended to spread safety stock across nearly every stage, partly because each function optimized its own buffer locally. The operational hypothesis is that some stages should act as decoupling points with inventory, while others may be able to replenish-to-order.

**The bounded-demand idea and operating flexibility.** The guaranteed-service philosophy assumes demand over any horizon $\tau$ is **bounded** by a known function $D(\tau)$ (e.g. $D(\tau) = \tau\mu + z\,\sigma\sqrt{\tau}$ for the normal case), sets safety stock to cover that bound, and treats the rare occasions demand exceeds the bound as handled by **extraordinary measures** — expediting, overtime, subcontracting — rather than letting the stockout ripple through the network. This is contrary to most stochastic-demand inventory models. In practice managers are often more comfortable specifying "100% service over a covered range of demand" than estimating a shortage cost for an external customer.

## Baselines

**Simpson (1958), "In-process inventories."** Serial line; each stage promises a guaranteed service time and holds enough safety stock to honor it; the external service time is taken to be zero. Simpson formulated the safety-stock placement and showed the optimal solution is an extreme point with the all-or-nothing property: at each stage either $S_i = 0$ or $S_i = S_{i+1} + T_i$. Stated as a structural property of the serial line.

**Clark & Scarf (1960), echelon stochastic-service.** Exact decomposition of a serial (later assembly) system into single-stage subproblems via echelon inventory and induced penalties; yields optimal echelon base-stock levels under stochastic service. Service times are outcomes of the policy.

**Graves (1988), serial as shortest path.** Recognized that the optimal serial-line guaranteed-service placement is a shortest-path problem on a graph of service-time states.

**Inderfurth (1991, 1993); Inderfurth & Minner (1998); Minner (1997); Graves & Willems (1996).** Dynamic-programming algorithms for the guaranteed-service problem on **assembly** networks and on **distribution** networks. These establish the DP-over-service-times approach for the two pure topologies, each handling one structure.

**Performance-evaluation base-stock models (Lee & Billington 1993; Glasserman & Tayur 1995; Ettl et al. 2000).** A parallel line that determines optimal base-stock levels in a supply chain under stochastic service, the central challenge being to approximate the replenishment lead time inside the chain (which is random because suppliers stock out), then solve a nonlinear program for the base stocks. These use different demand and service-level assumptions and rely on simulation / IPA / approximation, aimed at evaluation.

## Evaluation settings

The natural test bed is a supply chain modeled as a network of stages with: a deterministic production lead time $T_j$ per stage (waiting + processing + transport to put the item in inventory, assumed independent of order size — no capacity limit); a per-unit holding cost $h_j$; arc multipliers $\phi_{ij}$ (units of upstream item $i$ per downstream unit $j$); independent end-item demand only at nodes with no successors, stationary with mean $\mu_j$ and standard deviation $\sigma_j$; and a maximum service time $s_j$ at each demand node, set by the marketplace. The topology classes of interest are serial, assembly, distribution, and general networks that mix convergent and divergent regions. The yardsticks present at the time are the prior guaranteed-service DPs for serial / assembly / distribution, and the stochastic-service (Clark–Scarf echelon) solution where it applies.

The metric is the **total safety-stock holding cost** (and, secondarily, total inventory holding cost including the pipeline/work-in-process stock $T_j\mu_j$, which is fixed). A diagnostic comparison of interest is the cost penalty of *requiring* guaranteed internal service times versus letting internal service levels emerge from the chosen base stocks; for a small set of serial test problems (Poisson demand, mean $\lambda\in\{10,50\}$, truncation percentile $\alpha\in\{0.90,0.98\}$; lead-time triples $(T_1,T_2,T_3)\in\{(4,4,4),(1,3,8),(8,3,1)\}$; holding triples $(1,0.5,0.2),(1,0.66,0.33),(1,0.8,0.5)$) this penalty quantifies how much the guaranteed-service simplification costs.

## Code framework

The pieces that already exist before the placement algorithm: a graph data structure for the network, per-stage parameters, the single-stage safety-stock primitive, and a spot where the network optimizer will go.

```python
import math
import networkx as nx

class Stage:
    """One stage (node) of the supply chain."""
    def __init__(self, index, processing_time, holding_cost,
                 demand_mean=0.0, demand_std=0.0,
                 demand_bound_constant=None,     # z / k
                 max_service_time=None):         # s_j, demand nodes only
        self.index = index
        self.processing_time = processing_time   # T_j
        self.holding_cost = holding_cost         # h_j
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.demand_bound_constant = demand_bound_constant
        self.max_service_time = max_service_time

class Network:
    """Directed graph; arc (i, j) means i supplies j. phi[(i,j)] = units of i per unit j."""
    def __init__(self):
        self.G = nx.DiGraph()
        self.stages = {}
        self.phi = {}

    def add_stage(self, stage):
        pass

    def add_arc(self, i, j, phi=1.0):
        pass

    def predecessors(self, k):
        pass

    def successors(self, k):
        pass


def single_stage_safety_stock(stage, exposure_window):
    """Safety stock at one stage for a given exposure window."""
    z = stage.demand_bound_constant
    sigma = stage.demand_std
    tau = exposure_window
    return z * sigma * math.sqrt(max(0, tau))


def place_safety_stock(network):
    """Decide where in the network to hold safety stock, and how much, at minimum
    total holding cost while serving the end customer. (To be designed.)"""
    pass
```
