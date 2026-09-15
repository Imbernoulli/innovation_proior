# -*- coding: utf-8 -*-
"""重组(recombination)指标。
问题:“我们的模型是不是更不倾向于把已有技术拼起来?”
做法:把每份代码映射成一个“技术家族集合”(类似 Uzzi 2013 把论文映射成引用期刊集合),然后算
  R1 n_tech        : 一份解里出现的不同技术家族数(越低 = 拼的零件越少)
  R2 P(n_tech>=2)  : 真正发生“组合”的比例
  R3 new_pair_rate : 该解的技术对里,在 frontier 解池中从未共现过的比例(越高 = 新组合)
  R4 med_z / p10_z : Uzzi 原子性 z 值。以 frontier 解池为“既有文献”,做度数保持的随机零模型,
                     每个技术对算 z=(obs-mu)/sd;med_z 高 = 用的都是常规搭配;p10_z 低 = 伸手去够冷门搭配。
参照语料 = 官方 Frontier-CS 解池(gemini3pro / gpt5.x / deepseekreasoner / grok4 / trinity ...),与我们的模型无关。
"""
import json, glob, os, re, sys, random, math
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
O = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/Frontier-CS-official"

# ---------------- 技术家族词表 ----------------
# 每个家族 = 若干正则;命中任意一个即算该家族出现。对 C++ 与 Python 都给了标识符线索。
T = {
 "sim_anneal":      r"simulated[ _-]?anneal|\banneal|temperature\s*[*/=]|\btemp\s*\*=|exp\s*\(\s*-?\s*\(?\s*(delta|diff|d_?score)",
 "hill_climb":      r"hill[ _-]?climb|local[ _-]?search|\bimprove[d]?\s*=\s*true|neighbou?r\s*solution",
 "beam_search":     r"beam[ _-]?search|beam[ _-]?width|\bbeam\b",
 "genetic":         r"genetic|evolutionar|\bmutat|crossover|\bpopulation\b|\bgeneration\b\s*<",
 "tabu":            r"tabu",
 "mcts":            r"monte[ _-]?carlo|\bMCTS\b|playout|rollout",
 "random_restart":  r"restart|random[ _-]?seed|mt19937|rand_r|\brandom_device|\brng\b|uniform_int_distribution|np\.random",
 "branch_bound":    r"branch[ _-]?and[ _-]?bound|\bprune\b|\bbound\b\s*<",
 "greedy":          r"\bgreed",
 "bfs":             r"\bBFS\b|breadth[ _-]?first|queue<[^>]*>\s*q\b|deque\(\[",
 "dfs":             r"\bDFS\b|depth[ _-]?first|void\s+dfs|def\s+dfs",
 "dijkstra":        r"dijkstra",
 "bellman_floyd":   r"bellman[ _-]?ford|floyd[ _-]?warshall",
 "mst":             r"kruskal|\bprim\b|minimum[ _-]?spanning",
 "toposort":        r"topolog",
 "scc":             r"tarjan|strongly[ _-]?connected|\bSCC\b|articulation|bridge",
 "maxflow":         r"max[ _-]?flow|dinic|ford[ _-]?fulkerson|min[ _-]?cut|push[ _-]?relabel",
 "matching":        r"hungarian|kuhn|hopcroft|bipartite[ _-]?match|\bmatching\b",
 "dsu":             r"union[ _-]?find|disjoint[ _-]?set|\bDSU\b|find_?set|\bparent\[",
 "segment_tree":    r"segment[ _-]?tree|segtree|lazy[ _-]?propagat",
 "fenwick":         r"fenwick|binary[ _-]?indexed|\bBIT\b",
 "heap":            r"priority_queue|heapq|make_heap|\bheap\b",
 "hashmap":         r"unordered_map|unordered_set|\bdict\(|defaultdict|std::map<|HashMap",
 "trie":            r"\btrie\b|prefix[ _-]?tree",
 "sparse_table":    r"sparse[ _-]?table|\bRMQ\b",
 "balanced_bst":    r"treap|splay|\bAVL\b|red[ _-]?black|policy_based|order_of_key",
 "bitset":          r"bitset|__builtin_popcount|bit_count|popcount",
 "sorting":         r"\bsort\(|sorted\(|stable_sort|nth_element|\.sort\(",
 "dp":              r"\bdp\[|\bdp\b\s*=|dynamic[ _-]?programm|memo",
 "bitmask_dp":      r"bitmask|1\s*<<\s*n|\(1<<|mask\b",
 "knapsack":        r"knapsack",
 "sieve":           r"\bsieve\b|eratosthen|is_?prime",
 "modpow":          r"modpow|power_?mod|pow_?mod|fast[ _-]?pow|binpow",
 "gcd":             r"__gcd|\bgcd\(|math\.gcd|\blcm\b",
 "crt":             r"chinese[ _-]?remainder|\bCRT\b",
 "matrix_expo":     r"matrix[ _-]?(exponent|power)|matmul|matrix[ _-]?multipl",
 "fft":             r"\bFFT\b|\bNTT\b|fast[ _-]?fourier|convolution",
 "gauss":           r"gaussian[ _-]?elimin|row[ _-]?reduc|\brank\b\s*=|linalg\.solve|\bLU\b",
 "combinatorics":   r"binomial|\bnCr\b|factorial|\bcomb\(|catalan|inclusion[ _-]?exclusion",
 "probability":     r"probabilit|expected[ _-]?value|\bexpectation\b",
 "lp":              r"linear[ _-]?program|simplex|\bLP\b\s|scipy\.optimize\.linprog|pulp",
 "convex_hull":     r"convex[ _-]?hull|graham[ _-]?scan|andrew.?s monotone",
 "geometry":        r"cross[ _-]?product|\bdot\(|atan2|\bhypot|orientation\(|ccw\(",
 "kmp":             r"\bKMP\b|knuth[ _-]?morris|failure[ _-]?function|prefix[ _-]?function",
 "z_algo":          r"z[ _-]?algorithm|z[ _-]?function",
 "suffix":          r"suffix[ _-]?(array|automat|tree)|\bSA[ _-]?IS\b",
 "hashing_str":     r"rolling[ _-]?hash|polynomial[ _-]?hash|string[ _-]?hash",
 "aho":             r"aho[ _-]?corasick",
 "two_pointer":     r"two[ _-]?pointer|\bl\+\+.*\br\+\+|sliding[ _-]?window",
 "binary_search":   r"binary[ _-]?search|lower_bound|upper_bound|bisect",
 "ternary_search":  r"ternary[ _-]?search|golden[ _-]?section",
 "meet_middle":     r"meet[ _-]?in[ _-]?the[ _-]?middle",
 "divide_conquer":  r"divide[ _-]?and[ _-]?conquer|merge[ _-]?sort",
 "coord_compress":  r"coordinate[ _-]?compress|compress\(",
 "diff_array":      r"difference[ _-]?array|imos|prefix[ _-]?sum|cumsum|accumulate\(",
 "sqrt_decomp":     r"sqrt[ _-]?decomp|mo.?s algorithm|block[ _-]?size\s*=",
 "simulation":      r"\bsimulat",
 "caching":         r"lru_cache|\bcache\[|memoiz",
 "parallel":        r"thread|openmp|#pragma omp|multiprocessing|concurrent\.futures",
 "simd":            r"\bSIMD\b|__m256|avx|#pragma GCC target",
 # ---- ML / 科研赛道 ----
 "numpy":           r"\bimport numpy|\bnp\.",
 "torch":           r"\bimport torch|torch\.|\bnn\.Module",
 "sklearn":         r"sklearn|scikit",
 "pandas":          r"\bimport pandas|\bpd\.",
 "scipy":           r"\bimport scipy|scipy\.",
 "gradient_descent":r"gradient[ _-]?descent|\bSGD\b|\bAdam\b|optimizer\.|backward\(\)|lr_scheduler",
 "neural_net":      r"\bMLP\b|Linear\(|Conv\d?d|BatchNorm|LayerNorm|ReLU|activation",
 "attention":       r"attention|softmax\(.*QK|\bQKV?\b|transformer|rotary|RoPE",
 "regression":      r"LinearRegression|Ridge|Lasso|least[ _-]?squares|polyfit|curve_fit",
 "clustering":      r"KMeans|DBSCAN|AgglomerativeClustering|\bcluster",
 "pca":             r"\bPCA\b|SVD|truncated_svd|eigen",
 "ann_index":       r"faiss|\bHNSW\b|IVFFlat|annoy|nmslib|nearest[ _-]?neighbou?r",
 "decision_tree":   r"DecisionTree|RandomForest|GradientBoosting|XGB|LightGBM|CatBoost",
 "ensemble":        r"\bensemble|bagging|voting|stacking",
 "cross_val":       r"cross[ _-]?val|KFold|train_test_split",
 "bayes_opt":       r"bayesian[ _-]?optim|optuna|hyperopt|gaussian[ _-]?process",
 "symbolic_reg":    r"PySR|symbolic[ _-]?regress|\bsympy\b",
 "triton":          r"\btriton\b|tl\.load|tl\.store|@triton\.jit",
 "cuda":            r"\bcuda\b|__global__|threadIdx|blockIdx|cudaMalloc",
 "quantization":    r"quantiz|\bint8\b|\bfp8\b|bfloat16",
 "physics_sim":     r"n[ _-]?body|barnes[ _-]?hut|verlet|leapfrog|force[ _-]?field|octree",
 "info_theory":     r"entropy|mutual[ _-]?information|kullback|KL[ _-]?diverg",
 "control_theory":  r"\bPID\b|kalman|lyapunov|model[ _-]?predictive",
 "economics":       r"auction|nash|regret[ _-]?minim|multiplicative[ _-]?weights|game[ _-]?theor",
}
PAT = {k: re.compile(v, re.I) for k, v in T.items()}
FAMS = sorted(T)

def techset(code):
    if not code: return frozenset()
    c = code[:60000]
    return frozenset(k for k, p in PAT.items() if p.search(c))

# ---------------- 参照语料:frontier 解池 ----------------
def pool_docs():
    docs = []
    for f in glob.glob(f"{O}/algorithmic/solutions/*/*.cpp") + \
             glob.glob(f"{O}/research/solutions/*/*.py") + \
             glob.glob(f"{O}/research/solutions/*/*.cpp") + \
             glob.glob(f"{O}/research/solutions/*/*/*.py") + \
             glob.glob(f"{O}/research/solutions/*/*/*.cpp"):
        if re.search(r"reference", os.path.basename(f), re.I): continue
        try: s = open(f, errors="ignore").read()
        except Exception: continue
        t = techset(s)
        if len(t) >= 1: docs.append(t)
    return docs

def null_z(docs, nshuf=200, seed=0):
    """度数保持零模型:保持每篇文档的技术数与每个技术的文档频次(期望意义上),
    通过打乱全局 incidence 列表实现。返回 obs[pair], mu[pair], sd[pair]。"""
    rnd = random.Random(seed)
    sizes = [len(d) for d in docs]
    flat = [t for d in docs for t in d]
    obs = Counter()
    for d in docs:
        ds = sorted(d)
        for i in range(len(ds)):
            for j in range(i+1, len(ds)):
                obs[(ds[i], ds[j])] += 1
    acc = defaultdict(list)
    for s in range(nshuf):
        rnd.shuffle(flat)
        pos = 0
        cnt = Counter()
        for sz in sizes:
            d = set(flat[pos:pos+sz]); pos += sz
            ds = sorted(d)
            for i in range(len(ds)):
                for j in range(i+1, len(ds)):
                    cnt[(ds[i], ds[j])] += 1
        for k, v in cnt.items(): acc[k].append(v)
    Z = {}
    allpairs = set(obs) | set(acc)
    for k in allpairs:
        v = acc.get(k, [])
        v = v + [0]*(nshuf-len(v))
        mu = float(np.mean(v)); sd = float(np.std(v))
        Z[k] = ((obs.get(k, 0) - mu) / sd) if sd > 0 else (0.0 if obs.get(k,0) == mu else float("inf"))
    return obs, Z

def main():
    sys.stderr.write("building frontier pool corpus...\n")
    docs = pool_docs()
    sys.stderr.write("pool docs %d\n" % len(docs))
    obs, Z = null_z(docs)
    json.dump({"%s|%s" % k: v for k, v in Z.items()}, open(os.path.join(HERE, "recomb_pairz.json"), "w"))

    rows = []
    for line in open(os.path.join(HERE, "samples.jsonl")):
        d = json.loads(line)
        if not d.get("in_common") or not d.get("complete") or not d.get("has_code"): continue
        t = techset(d.get("code") or "")
        ts = sorted(t)
        pairs = [(ts[i], ts[j]) for i in range(len(ts)) for j in range(i+1, len(ts))]
        zs = [Z.get(p, None) for p in pairs]
        zs = [z for z in zs if z is not None and math.isfinite(z)]
        newp = sum(1 for p in pairs if obs.get(p, 0) == 0)
        rows.append(dict(bench=d["bench"], fam=d["fam"], arm=d["arm"], stage=d["stage"],
                         problem=str(d["problem"]), sample_idx=d["sample_idx"], score=d["score"],
                         n_tech=len(ts), n_pair=len(pairs),
                         new_pair_rate=(newp/len(pairs) if pairs else float("nan")),
                         med_z=(float(np.median(zs)) if zs else float("nan")),
                         p10_z=(float(np.percentile(zs, 10)) if zs else float("nan")),
                         techs="|".join(ts)))
    import csv as _csv
    with open(os.path.join(HERE, "recomb_metrics.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sys.stderr.write("rows %d\n" % len(rows))

if __name__ == "__main__":
    main()
