#!/usr/bin/env python3
"""Canonical coarse-category taxonomy for the site's sidebar.

The corpus carries ~150 fine-grained `domain` strings, which is far too many to
browse. This module collapses every domain into ONE of five coarse categories
(the level the sidebar groups by); the fine `domain` is kept on each entry and
shown as the per-item subtitle, so nothing is lost.

The five buckets (in sidebar order):
  1. Mathematics & Physics              — basic science: pure/applied math,
                                          physics, statistics, chemistry, quantum.
  2. Combinatorial & Competitive Algorithms
                                        — "Circle-Packing"-style combinatorial /
                                          geometric optimization, metaheuristic
                                          search, and competitive-programming algos.
  3. Empirical Machine Learning         — the empirical ML stack (vision, RL, LLMs,
                                          generative, optimizers/HPO, GNNs, …).
  4. Theory (CS & ML)                   — TCS + learning theory: complexity, crypto,
                                          coding/information theory, learning theory.
  5. Applied & Engineering              — everything applied that fits none of the
                                          above: security, optics/imaging, control,
                                          economics/finance, OR, applied bio, etc.

Run from the repo root to (re)stamp `category` onto methods.json + trajectories.json:
    python3 tools/categorize.py
Then refresh the agentic index:  python3 tools/build_site_data.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sidebar order is this list's order.
CATEGORIES = [
    "Mathematics & Physics",
    "Combinatorial & Competitive Algorithms",
    "Empirical Machine Learning",
    "Theory (CS & ML)",
    "Applied & Engineering",
]
M, C, ML, TH, AP = CATEGORIES

# Every fine `domain` string seen in methods.json / trajectories.json maps here.
# Decisions documented inline where a domain straddles buckets.
DOMAIN_TO_CATEGORY = {
    # ---- 1. Mathematics & Physics (basic science) ----
    "Physics": M,
    "Pure mathematics": M,
    "Combinatorics": M,
    "Extremal combinatorics": M,
    "Probability & combinatorics": M,
    "Probability": M,
    "Probabilistic combinatorics": M,
    "Statistics": M,                 # mathematical statistics: bootstrap, EM, Cramér-Rao, James-Stein…
    "Mathematical logic": M,
    "Algebraic geometry": M,
    "Algebra": M,
    "Functional analysis": M,
    "Geometry and topology": M,
    "Graph theory": M,               # pure graph theory (cf. "Graph algorithms" -> Combinatorial)
    "Order theory": M,
    "Game theory": M,                # Shapley value etc.
    "Competition math": M,
    "Math olympiad": M,
    "Physics olympiad": M,
    "Mathematical physics": M,
    "Particle Physics": M,
    "Quantum Computing": M,
    "Quantum information": M,
    "Quantum Control": M,
    "Theoretical chemistry": M,
    "Computational Chemistry": M,
    "Molecular Mechanics": M,
    "Convex Optimization": M,        # ADMM / convexification — mathematical optimization

    # ---- 2. Combinatorial & Competitive Algorithms ----
    "Informatics olympiad": C,
    "Combinatorial optimization": C,
    "Combinatorics & discrete optimization": C,
    "Heuristic combinatorial optimization": C,
    "Heuristic combinatorial optimization (ALE-Bench / AtCoder Heuristic Contest)": C,
    "Algorithms & heuristic optimization": C,
    "Geometric optimization": C,
    "Computational geometry": C,
    "Graph algorithms": C,
    "Fast algorithms": C,
    "Strings & text": C,
    "Sorting & selection": C,
    "Scheduling": C,
    "Analysis & continuous optimization": C,  # AlphaEvolve-style extremal-search records (C1/C2)
    "Global Optimization": C,                 # metaheuristic / differential-evolution search
    "Multi-Objective Optimization": C,        # evolutionary many-objective search
    "Swarm Intelligence": C,                  # PSO/ACO metaheuristics

    # ---- 3. Empirical Machine Learning ----
    "Optimization": ML,              # dominated by ML optimizers/HPO (Adam, AdaGrad, BOHB, …)
    "Reinforcement learning": ML,
    "Reinforcement Learning": ML,
    "Language models": ML,
    "Language Models": ML,
    "Computer vision": ML,
    "Generative models": ML,
    "Generative Models": ML,
    "Classical ML": ML,
    "Graph & mesh": ML,              # GNNs (GCN/GAT/GraphSAGE/Graphormer…)
    "Causal inference": ML,
    "Causal discovery": ML,
    "Systems & efficiency": ML,      # ML systems (efficient training/inference)
    "Machine learning systems": ML,
    "Sequence modeling": ML,
    "AI for science": ML,
    "AI for biology": ML,
    "Computational biology / single-cell genomics": ML,
    "Self-supervised": ML,
    "Deep learning": ML,
    "Training & normalization": ML,
    "Meta-learning": ML,
    "Time series": ML,
    "Quantization": ML,
    "Retrieval": ML,
    "Parameter-Efficient Fine-Tuning": ML,
    "LLM Reasoning": ML,
    "LLM Inference": ML,
    "LLM Agents": ML,
    "LLM internals": ML,
    "Speech Recognition": ML,
    "Speech Synthesis": ML,
    "Audio Generation": ML,
    "Model Compression": ML,
    "Knowledge Distillation": ML,
    "Distributed Training": ML,
    "Alignment": ML,
    "Instruction Tuning": ML,
    "Robotics": ML,                  # robot learning / RL (cf. Legged/Swarm Robotics -> Applied)
    "Neural Architecture Search": ML,
    "Operator learning": ML,
    "Differentiable solvers": ML,
    "Optimal Transport": ML,         # Sinkhorn/Wasserstein in ML
    "Bayesian Optimization": ML,     # HPO for ML
    "Vision-Language": ML,
    "Tokenization": ML,
    "Object Detection": ML,
    "Semantic Segmentation": ML,
    "Neural Rendering": ML,
    "Neural rendering": ML,
    "Efficient Attention": ML,
    "Efficient Architectures": ML,
    "Long-Context Transformers": ML,
    "Sentence Embeddings": ML,
    "Geometric Deep Learning": ML,
    "CNN Architectures": ML,
    "Calibration": ML,
    "Probabilistic graphical models": ML,
    "Probabilistic inference": ML,
    "Artificial intelligence": ML,
    "": ML,                          # untyped methods — all neural-net learning papers

    # ---- 4. Theory (CS & ML) ----
    "Learning theory": TH,
    "ML theory": TH,
    "Computational complexity": TH,
    "Computability theory": TH,
    "Theoretical computer science": TH,
    "Approximation algorithms": TH,
    "Online algorithms": TH,
    "Randomized & streaming": TH,
    "Streaming & sketching": TH,
    "Cryptography": TH,
    "Cryptographic algorithms": TH,
    "Coding & information theory": TH,
    "Coding Theory": TH,
    "Information theory": TH,
    "Distributed computing": TH,     # FLP impossibility — distributed-algorithms theory
    "Automated reasoning": TH,

    # ---- 5. Applied & Engineering ----
    "Security": AP,
    "Economics": AP,
    "Economics & finance": AP,
    "Portfolio Optimization": AP,
    "Inventory Optimization": AP,
    "PDE-Constrained Optimization": AP,
    "Structural Optimization": AP,
    "Operations Research": AP,
    "Maritime Routing": AP,
    "Sensor Placement": AP,
    "Power Systems": AP,
    "Additive Manufacturing": AP,
    "Electrochemical Modeling": AP,
    "Rare-Event Simulation": AP,
    "Control": AP,
    "State Estimation": AP,
    "Signal processing & estimation": AP,
    "Signal Processing": AP,
    "Compressed Sensing": AP,
    "Computational Optics": AP,
    "Optical Communications": AP,
    "Optical Design": AP,
    "Adaptive Optics": AP,
    "Computational Imaging": AP,
    "Medical Physics": AP,
    "Life sciences": AP,
    "Bioinformatics": AP,
    "Computational Biology": AP,
    "Legged Robotics": AP,
    "Swarm Robotics": AP,
    "Linguistics olympiad": AP,      # language puzzles — fits none of the algorithmic/science buckets
    "Databases": AP,
    "Systems": AP,                   # systems programming (e.g. memory allocator)
}


def category_for(domain):
    """Coarse category for a fine `domain` string (Applied & Engineering if unseen)."""
    return DOMAIN_TO_CATEGORY.get((domain or "").strip(), AP)


def _annotate(path):
    data = json.load(open(path))
    unseen = set()
    for x in data:
        dom = x.get("domain", "")
        if (dom or "").strip() not in DOMAIN_TO_CATEGORY:
            unseen.add(dom)
        x["category"] = category_for(dom)
    with open(path, "w") as f:
        # match the files' existing style (2-space indent, raw UTF-8) so the
        # only diff is the added per-entry `category` line.
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    name = os.path.basename(path)
    counts = {}
    for x in data:
        counts[x["category"]] = counts.get(x["category"], 0) + 1
    print(f"{name}: {len(data)} entries")
    for cat in CATEGORIES:
        if cat in counts:
            print(f"    {counts[cat]:5d}  {cat}")
    if unseen:
        print(f"    ! UNMAPPED domains (defaulted to '{AP}'): {sorted(unseen)}")
    return unseen


def main():
    miss = set()
    for f in ("methods.json", "trajectories.json"):
        miss |= _annotate(os.path.join(ROOT, f))
    if miss:
        raise SystemExit(f"\nUnmapped domains found: {sorted(miss)} — add them to DOMAIN_TO_CATEGORY.")


if __name__ == "__main__":
    main()
