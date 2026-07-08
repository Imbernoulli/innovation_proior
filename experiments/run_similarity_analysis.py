#!/usr/bin/env python3
"""Similarity-to-baselines analysis for MLS-Bench task logs.

The script intentionally uses lightweight regex/string heuristics so it can run
in the evaluation environment without new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


TASKS_DIR = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks")
LOG_DIRS = {
    "OURS_methodtraj": Path(
        "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/"
        "cc_eval_all_r3_methodtraj_v4_r3_a10/mls/task_logs"
    ),
    "OURS_methodv4": Path(
        "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/"
        "cc_eval_all_r3_methodv4_r3_a10/mls/task_logs"
    ),
    "BASE": Path(
        "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/"
        "cc_mlsbench_cpu_q35_start_devfix/task_logs"
    ),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COMMON_INFRA = {
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "pandas",
    "math",
    "random",
    "collections",
    "typing",
    "itertools",
}
GENERIC_METHOD_STOPWORDS = {
    "selection",
    "parent_selection",
    "variation",
    "environmental_selection",
    "task",
    "metric",
    "metrics",
}


def _alias(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


# Includes the requested vocabulary plus task-specific MLS-Bench baseline names.
ALIASES: Dict[str, Sequence[re.Pattern[str]]] = {
    "random_forest": [_alias(r"\brandom[\s_-]*forest\b"), _alias(r"\bRandomForest(?:Classifier|Regressor)?\b")],
    "xgboost": [_alias(r"\bxgboost\b"), _alias(r"\bxgb(?:oost)?[\s_-]*style\b")],
    "lightgbm": [_alias(r"\blightgbm\b"), _alias(r"\blgbm\b")],
    "catboost": [_alias(r"\bcatboost\b")],
    "tpe": [_alias(r"\btpe\b"), _alias(r"\btree[\s_-]*structured[\s_-]*parzen\b")],
    "hyperband": [_alias(r"\bhyperband\b")],
    "optuna": [_alias(r"\boptuna\b")],
    "smac": [_alias(r"\bsmac\b")],
    "bohb": [_alias(r"\bbohb\b")],
    "dehb": [_alias(r"\bdehb\b")],
    "bayesian_optimization": [_alias(r"\bbayesian[\s_-]*optimization\b")],
    "cma_es": [_alias(r"\bcma[\s_-]*es\b"), _alias(r"\bcmaes\b")],
    "optuna_cma": [_alias(r"\boptuna[\s_-]*cma(?:[\s_-]*es)?\b")],
    "neural_network": [_alias(r"\bneural[\s_-]*network\b")],
    "mlp": [_alias(r"\bmlp\b"), _alias(r"\bMLP(?:Classifier|Regressor)?\b")],
    "cnn": [_alias(r"\bcnn\b"), _alias(r"\bconvolutional[\s_-]*network\b")],
    "lstm": [_alias(r"\blstm\b")],
    "transformer": [_alias(r"\btransformer\b")],
    "attention": [_alias(r"\battention\b"), _alias(r"\bself[\s_-]*attention\b")],
    "ensemble": [_alias(r"\bensemble\b"), _alias(r"\bensembling\b")],
    "stacking": [_alias(r"\bstacking\b")],
    "blending": [_alias(r"\bblending\b")],
    "gradient_boosting": [_alias(r"\bgradient[\s_-]*boosting\b"), _alias(r"\bGBM\b")],
    "adaboost": [_alias(r"\badaboost\b")],
    "svm": [_alias(r"\bsvm\b"), _alias(r"\bSVC\b"), _alias(r"\bone[\s_-]*class[\s_-]*svm\b")],
    "knn": [_alias(r"\bk[\s_-]*nearest[\s_-]*neighbors?\b"), _alias(r"\bknn\b"), _alias(r"\bNearestNeighbors\b")],
    "lasso": [_alias(r"\blasso\b")],
    "ridge": [_alias(r"\bridge\b")],
    "elastic_net": [_alias(r"\belastic[\s_-]*net\b")],
    "causal_forest": [_alias(r"\bcausal[\s_-]*forest\b")],
    "double_ml": [_alias(r"\bdouble[\s_-]*(?:ml|machine[\s_-]*learning)\b"), _alias(r"\bdml\b")],
    "s_learner": [_alias(r"\bs[\s_-]*learner\b")],
    "t_learner": [_alias(r"\bt[\s_-]*learner\b")],
    "x_learner": [_alias(r"\bx[\s_-]*learner\b")],
    "dr_learner": [_alias(r"\bdr[\s_-]*learner\b"), _alias(r"\bdoubly[\s_-]*robust\b")],
    "r_learner": [_alias(r"\br[\s_-]*learner\b")],
    "propensity": [_alias(r"\bpropensity\b")],
    "ipw": [_alias(r"\bipw\b"), _alias(r"\binverse[\s_-]*propensity\b")],
    "matching": [_alias(r"\bmatching\b")],
    "did": [_alias(r"\bdifference[\s_-]*in[\s_-]*differences\b"), _alias(r"\bdid\b")],
    "synthetic_control": [_alias(r"\bsynthetic[\s_-]*control\b")],
    "regression_discontinuity": [_alias(r"\bregression[\s_-]*discontinuity\b")],
    "meta_learner": [_alias(r"\bmeta[\s_-]*learner\b")],
    "rnn": [_alias(r"\brnn\b")],
    "gru": [_alias(r"\bgru\b")],
    "bert": [_alias(r"\bbert\b")],
    "gpt": [_alias(r"\bgpt\b")],
    "rl": [_alias(r"\breinforcement[\s_-]*learning\b")],
    "q_learning": [_alias(r"\bq[\s_-]*learning\b")],
    "ppo": [_alias(r"\bppo\b")],
    "dqn": [_alias(r"\bdqn\b")],
    "actor_critic": [_alias(r"\bactor[\s_-]*critic\b")],
    "random_search": [_alias(r"\brandom[\s_-]*search\b"), _alias(r"\bsample_uniform\b")],
    "random_sampling": [_alias(r"\brandom[\s_-]*sampling\b")],
    "grid_search": [_alias(r"\bgrid[\s_-]*search\b")],
    "successive_halving": [_alias(r"\bsuccessive[\s_-]*halving\b")],
    "asha": [_alias(r"\basha\b")],
    "pbt": [_alias(r"\bpopulation[\s_-]*based[\s_-]*training\b"), _alias(r"\bpbt\b")],
    "median_stopping": [_alias(r"\bmedian[\s_-]*stopping\b")],
    "differential_evolution": [_alias(r"\bdifferential[\s_-]*evolution\b"), _alias(r"\bDE/rand/1/bin\b")],
    "evolution_strategy": [_alias(r"\bevolution(?:ary)?[\s_-]*strategy\b")],
    "genetic_algorithm": [_alias(r"\bgenetic[\s_-]*algorithms?\b"), _alias(r"\bga_sbx\b")],
    "regularized_evolution": [_alias(r"\bregularized[\s_-]*evolution\b"), _alias(r"\brea\b")],
    "lshade": [_alias(r"\bl[\s_-]*shade\b")],
    "mutation": [_alias(r"\bmutation\b"), _alias(r"\bmutate\b")],
    "crossover": [_alias(r"\bcrossover\b"), _alias(r"\brecombination\b")],
    "elitism": [_alias(r"\belitism\b"), _alias(r"\belite\b")],
    "tournament_selection": [_alias(r"\btournament[\s_-]*selection\b")],
    "nsga2": [_alias(r"\bnsga[\s_-]*ii\b"), _alias(r"\bnsga2\b")],
    "nsga3": [_alias(r"\bnsga[\s_-]*iii\b"), _alias(r"\bnsga3\b")],
    "moead": [_alias(r"\bmoea[\s_/-]*d\b"), _alias(r"\bmoead\b")],
    "spea2": [_alias(r"\bspea2\b")],
    "rvea": [_alias(r"\brvea\b")],
    "age_moea": [_alias(r"\bage[\s_-]*moea\b")],
    "pareto": [_alias(r"\bpareto\b")],
    "hypervolume": [_alias(r"\bhypervolume\b")],
    "crowding_distance": [_alias(r"\bcrowding[\s_-]*distance\b")],
    "non_dominated_sorting": [_alias(r"\bnon[\s_-]*dominated[\s_-]*sorting\b")],
    "weighted_sum": [_alias(r"\bweighted[\s_-]*sum\b")],
    "epsilon_constraint": [_alias(r"\bepsilon[\s_-]*constraint\b")],
    "pc": [_alias(r"\bpc\b"), _alias(r"\bpeter[\s_-]*clark\b")],
    "ges": [_alias(r"\bges\b"), _alias(r"\bgreedy[\s_-]*equivalence[\s_-]*search\b")],
    "grasp": [_alias(r"\bgrasp\b"), _alias(r"\bgreedy[\s_-]*relaxations\b")],
    "boss": [_alias(r"\bboss\b"), _alias(r"\bbest[\s_-]*order[\s_-]*score[\s_-]*search\b")],
    "hc": [_alias(r"\bhill[\s_-]*climbing\b"), _alias(r"\bhc\b")],
    "icalingam": [_alias(r"\bica[\s_-]*lingam\b"), _alias(r"\bicalingam\b")],
    "directlingam": [_alias(r"\bdirect[\s_-]*lingam\b"), _alias(r"\bdirectlingam\b")],
    "lingam": [_alias(r"\blingam\b"), _alias(r"\blinear[\s_-]*non[\s_-]*gaussian\b")],
    "notears": [_alias(r"\bnotears\b"), _alias(r"\bno[\s_-]*tears\b")],
    "notears_mlp": [_alias(r"\bnotears[\s_-]*mlp\b")],
    "cam": [_alias(r"\bcausal[\s_-]*additive[\s_-]*models?\b"), _alias(r"\bcam\b")],
    "grandag": [_alias(r"\bgran[\s_-]*dag\b"), _alias(r"\bgrandag\b")],
    "anm": [_alias(r"\badditive[\s_-]*noise[\s_-]*models?\b"), _alias(r"\banm\b")],
    "independence_test": [_alias(r"\bindependence[\s_-]*test\b"), _alias(r"\bhsic\b")],
    "constraint_based": [_alias(r"\bconstraint[\s_-]*based\b")],
    "score_based": [_alias(r"\bscore[\s_-]*based\b")],
    "permutation_search": [_alias(r"\bpermutation[\s_-]*(?:based|search)\b")],
    "bdeu": [_alias(r"\bbdeu\b")],
    "chi_squared": [_alias(r"\bchi[\s_-]*squared\b")],
    "least_confidence": [_alias(r"\bleast[\s_-]*confidence\b")],
    "uncertainty_sampling": [_alias(r"\buncertainty[\s_-]*sampling\b")],
    "bald": [_alias(r"\bbald\b"), _alias(r"\bbayesian[\s_-]*active[\s_-]*learning[\s_-]*by[\s_-]*disagreement\b")],
    "badge": [_alias(r"\bbadge\b"), _alias(r"\bgradient[\s_-]*embeddings?\b")],
    "bait": [_alias(r"\bbait\b"), _alias(r"\bfisher[\s_-]*embeddings?\b")],
    "core_set": [_alias(r"\bcore[\s_-]*set\b"), _alias(r"\bcoreset\b")],
    "kmeans": [_alias(r"\bk[\s_-]*means\b"), _alias(r"\bkmeans\b")],
    "kmeans_plus_plus": [_alias(r"\bk[\s_-]*means\+\+\b"), _alias(r"\bkmeans\+\+\b")],
    "mc_dropout": [_alias(r"\bmc[\s_-]*dropout\b")],
    "entropy": [_alias(r"\bentropy\b")],
    "diversity": [_alias(r"\bdiversity\b"), _alias(r"\bdiverse\b")],
    "representativeness": [_alias(r"\brepresentativeness\b"), _alias(r"\brepresentative\b")],
    "information_gain": [_alias(r"\binformation[\s_-]*gain\b"), _alias(r"\bmutual[\s_-]*information\b")],
    "fisher_information": [_alias(r"\bfisher[\s_-]*information\b")],
    "isolation_forest": [_alias(r"\bisolation[\s_-]*forest\b"), _alias(r"\biforest\b")],
    "local_outlier_factor": [_alias(r"\blocal[\s_-]*outlier[\s_-]*factor\b"), _alias(r"\blof\b")],
    "one_class_svm": [_alias(r"\bone[\s_-]*class[\s_-]*svm\b"), _alias(r"\bocsvm\b")],
    "ecod": [_alias(r"\becod\b"), _alias(r"\bempirical[\s_-]*cumulative[\s_-]*distribution[\s_-]*outlier\b")],
    "copod": [_alias(r"\bcopod\b"), _alias(r"\bcopula[\s_-]*based[\s_-]*outlier\b")],
    "hbos": [_alias(r"\bhbos\b")],
    "loda": [_alias(r"\bloda\b")],
    "pca": [_alias(r"\bpca\b"), _alias(r"\bprincipal[\s_-]*components?\b")],
    "autoencoder": [_alias(r"\bautoencoder\b")],
    "gaussian_mixture": [_alias(r"\bgaussian[\s_-]*mixture\b"), _alias(r"\bGaussianMixture\b")],
    "kernel_density": [_alias(r"\bkernel[\s_-]*density\b"), _alias(r"\bKDE\b")],
    "empirical_cdf": [_alias(r"\bempirical[\s_-]*(?:cdf|cumulative)\b")],
    "tail_probability": [_alias(r"\btail[\s_-]*probabilit(?:y|ies)\b")],
    "platt_scaling": [_alias(r"\bplatt[\s_-]*scaling\b")],
    "isotonic_regression": [_alias(r"\bisotonic[\s_-]*regression\b"), _alias(r"\bIsotonicRegression\b")],
    "temperature_scaling": [_alias(r"\btemperature[\s_-]*scaling\b")],
    "beta_calibration": [_alias(r"\bbeta[\s_-]*calibration\b")],
    "histogram_binning": [_alias(r"\bhistogram[\s_-]*binning\b")],
    "vector_scaling": [_alias(r"\bvector[\s_-]*scaling\b")],
    "conformal_prediction": [_alias(r"\bconformal\b")],
    "ece": [_alias(r"\bece\b"), _alias(r"\bexpected[\s_-]*calibration[\s_-]*error\b")],
    "brier": [_alias(r"\bbrier\b")],
    "group_temperature_scaling": [_alias(r"\bgroup[\s_-]*wise[\s_-]*temperature[\s_-]*scaling\b")],
    "group_calibration": [_alias(r"\bgroup[\s_-]*calibration\b"), _alias(r"\bsubgroup[\s_-]*calibration\b")],
    "multicalibration": [_alias(r"\bmulticalibration\b")],
    "dbscan": [_alias(r"\bdbscan\b")],
    "hdbscan": [_alias(r"\bhdbscan\b")],
    "spectral_clustering": [_alias(r"\bspectral[\s_-]*clustering\b")],
    "agglomerative": [_alias(r"\bagglomerative\b"), _alias(r"\bhierarchical[\s_-]*clustering\b")],
    "mean_shift": [_alias(r"\bmean[\s_-]*shift\b")],
    "affinity_propagation": [_alias(r"\baffinity[\s_-]*propagation\b")],
    "umap": [_alias(r"\bumap\b")],
    "tsne": [_alias(r"\bt[\s_-]*sne\b"), _alias(r"\btsne\b")],
    "trimap": [_alias(r"\btrimap\b")],
    "pacmap": [_alias(r"\bpacmap\b")],
    "kernel_pca": [_alias(r"\bkernel[\s_-]*pca\b")],
    "ica": [_alias(r"\bindependent[\s_-]*component[\s_-]*analysis\b")],
    "nmf": [_alias(r"\bnmf\b"), _alias(r"\bnon[\s_-]*negative[\s_-]*matrix[\s_-]*factorization\b")],
    "lle": [_alias(r"\blle\b"), _alias(r"\blocal[\s_-]*linear[\s_-]*embedding\b")],
    "isomap": [_alias(r"\bisomap\b")],
    "random_projection": [_alias(r"\brandom[\s_-]*projection\b")],
    "svd": [_alias(r"\btruncated[\s_-]*svd\b"), _alias(r"\bTruncatedSVD\b")],
    "mean_imputation": [_alias(r"\bmean[\s_-]*imputation\b")],
    "median_imputation": [_alias(r"\bmedian[\s_-]*imputation\b")],
    "knn_imputation": [_alias(r"\bknn[\s_-]*imputation\b"), _alias(r"\bk[\s_-]*nearest[\s_-]*neighbors[\s_-]*imputation\b")],
    "mice": [_alias(r"\bmice\b"), _alias(r"\bchained[\s_-]*equations\b")],
    "iterative_imputer": [_alias(r"\biterative[\s_-]*imputer\b"), _alias(r"\bIterativeImputer\b")],
    "missforest": [_alias(r"\bmissforest\b")],
    "matrix_completion": [_alias(r"\bmatrix[\s_-]*completion\b")],
    "softimpute": [_alias(r"\bsoft[\s_-]*impute\b"), _alias(r"\bsoftimpute\b")],
    "gain": [_alias(r"\bgenerative[\s_-]*adversarial[\s_-]*imputation\b")],
    "selective_classification": [_alias(r"\bselective[\s_-]*classification\b")],
    "reject_option": [_alias(r"\breject[\s_-]*option\b")],
    "confidence_threshold": [_alias(r"\bconfidence[\s_-]*threshold\b")],
    "conformal_abstention": [_alias(r"\bconformal[\s_-]*abstention\b")],
    "groupwise_thresholding": [_alias(r"\bgroup[\s_-]*wise[\s_-]*thresholding\b")],
    "learned_deferral": [_alias(r"\blearned[\s_-]*deferral\b")],
    "cost_sensitive": [_alias(r"\bcost[\s_-]*sensitive\b")],
    "deferral": [_alias(r"\bdeferral\b"), _alias(r"\bdefer\b")],
    "human_in_the_loop": [_alias(r"\bhuman[\s_-]*in[\s_-]*the[\s_-]*loop\b")],
    "standard_gp": [_alias(r"\bstandard[\s_-]*gp\b")],
    "parsimony_gp": [_alias(r"\bparsimony[\s_-]*gp\b"), _alias(r"\bparsimony[\s_-]*pressure\b")],
    "lexicase_gp": [_alias(r"\blexicase[\s_-]*gp\b"), _alias(r"\bepsilon[\s_-]*lexicase\b"), _alias(r"\blexicase[\s_-]*selection\b")],
    "genetic_programming": [_alias(r"\bgenetic[\s_-]*programming\b"), _alias(r"\bexpression[\s_-]*tree\b")],
    "symbolic_regression": [_alias(r"\bsymbolic[\s_-]*regression\b")],
    "load_balance": [_alias(r"\bload[\s_-]*balanc(?:e|ing)\b")],
    "expert_replication": [_alias(r"\bexpert[\s_-]*replication\b"), _alias(r"\breplicate[\s_-]*experts\b")],
    "balanced_packing": [_alias(r"\bbalanced[\s_-]*packing\b")],
    "bin_packing": [_alias(r"\bbin[\s_-]*packing\b")],
    "eplb": [_alias(r"\beplb\b"), _alias(r"\bexpert[\s_-]*parallelism[\s_-]*load[\s_-]*balancer\b")],
    "moe": [_alias(r"\bmixture[\s_-]*of[\s_-]*experts\b"), _alias(r"\bmoe\b")],
    "top_k_routing": [_alias(r"\btop[\s_-]*k[\s_-]*routing\b")],
    "sinkhorn": [_alias(r"\bsinkhorn\b")],
    "auxiliary_loss": [_alias(r"\bauxiliary[\s_-]*loss\b")],
    "bananas": [_alias(r"\bbananas\b")],
    "npenas": [_alias(r"\bnpenas\b")],
    "nas_bench_201": [_alias(r"\bnas[\s_-]*bench[\s_-]*201\b")],
    "surrogate": [_alias(r"\bsurrogate\b"), _alias(r"\bpredictor[\s_-]*guided\b")],
    "path_encoding": [_alias(r"\bpath[\s_-]*encoding\b")],
    "ucb": [_alias(r"\bucb\b"), _alias(r"\bupper[\s_-]*confidence[\s_-]*bound\b")],
    "expected_improvement": [_alias(r"\bexpected[\s_-]*improvement\b"), _alias(r"\bei\b")],
    "thompson_sampling": [_alias(r"\bthompson[\s_-]*sampling\b")],
    "acquisition_function": [_alias(r"\bacquisition[\s_-]*function\b")],
    "gaussian_process": [_alias(r"\bgaussian[\s_-]*process\b")],
    "gnn": [_alias(r"\bgnn\b"), _alias(r"\bgraph[\s_-]*neural[\s_-]*network\b")],
    "zero_cost_proxy": [_alias(r"\bzero[\s_-]*cost[\s_-]*prox(?:y|ies)\b")],
    "local_search": [_alias(r"\blocal[\s_-]*search\b")],
    "multi_fidelity": [_alias(r"\bmulti[\s_-]*fidelity\b")],
    "sample_weighting": [_alias(r"\bsample[\s_-]*weight(?:ing|s)?\b")],
    "newton_update": [_alias(r"\bnewton[\s_-]*(?:step|update)\b"), _alias(r"\bsecond[\s_-]*order\b")],
    "exponential_loss": [_alias(r"\bexponential[\s_-]*loss\b")],
    "pseudo_residuals": [_alias(r"\bpseudo[\s_-]*(?:residuals?|targets?)\b")],
    "line_search": [_alias(r"\bline[\s_-]*search\b")],
    "huber_loss": [_alias(r"\bhuber\b")],
    "focal_loss": [_alias(r"\bfocal[\s_-]*loss\b")],
    "quantile_loss": [_alias(r"\bquantile[\s_-]*loss\b")],
    "orthogonalization": [_alias(r"\borthogonalization\b"), _alias(r"\borthogonal\b")],
    "residualization": [_alias(r"\bresidualization\b"), _alias(r"\bresidualized\b")],
    "tmle": [_alias(r"\btmle\b"), _alias(r"\btargeted[\s_-]*maximum[\s_-]*likelihood\b")],
}


EXACT_NAME_MAP: Dict[str, str] = {}
for canon, pats in ALIASES.items():
    EXACT_NAME_MAP[canon] = canon
for raw, canon in {
    "random search": "random_search",
    "random sampling": "random_sampling",
    "least-confidence": "least_confidence",
    "least confidence": "least_confidence",
    "uncertainty sampling": "uncertainty_sampling",
    "gradient boosting": "gradient_boosting",
    "xgboost-style": "xgboost",
    "one-class svm": "one_class_svm",
    "local outlier factor": "local_outlier_factor",
    "isolation forest": "isolation_forest",
    "k-means": "kmeans",
    "k-means++": "kmeans_plus_plus",
    "t-sne": "tsne",
    "temperature scaling": "temperature_scaling",
    "isotonic regression": "isotonic_regression",
    "platt scaling": "platt_scaling",
    "beta calibration": "beta_calibration",
    "group-wise temperature scaling": "group_temperature_scaling",
    "mean imputation": "mean_imputation",
    "knn imputation": "knn_imputation",
    "k-nearest neighbors imputation": "knn_imputation",
    "standard gp": "standard_gp",
    "parsimony gp": "parsimony_gp",
    "lexicase gp": "lexicase_gp",
    "genetic programming": "genetic_programming",
    "balanced packing": "balanced_packing",
    "expert replication": "expert_replication",
    "bin-packing": "bin_packing",
    "regularized evolution": "regularized_evolution",
    "nsga-ii": "nsga2",
    "nsga-iii": "nsga3",
    "moea/d": "moead",
    "cma-es": "cma_es",
    "l-shade": "lshade",
    "differential evolution": "differential_evolution",
    "de": "differential_evolution",
    "optuna cma": "optuna_cma",
    "optuna cma es": "optuna_cma",
    "optunacma": "optuna_cma",
    "s-learner": "s_learner",
    "t-learner": "t_learner",
    "x-learner": "x_learner",
    "dr-learner": "dr_learner",
    "r-learner": "r_learner",
    "confidence thresholding": "confidence_threshold",
    "confidencethresholding": "confidence_threshold",
    "conformal abstention": "conformal_abstention",
    "conformalabstention": "conformal_abstention",
    "group wise thresholding": "groupwise_thresholding",
    "groupwisethresholding": "groupwise_thresholding",
    "learned deferral": "learned_deferral",
    "learneddeferral": "learned_deferral",
    "gain": "gain",
}.items():
    EXACT_NAME_MAP[raw] = canon


@dataclass
class ModelResult:
    techniques: Set[str] = field(default_factory=set)
    jaccard: Optional[float] = None
    beyond: Optional[bool] = None
    status: str = "ok"
    notes: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task: str
    baselines: Set[str] = field(default_factory=set)
    baseline_notes: List[str] = field(default_factory=list)
    models: Dict[str, ModelResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def normalize_name(raw: str) -> str:
    value = raw.strip().lower()
    value = re.sub(r"\([^)]*\)", "", value)
    value = value.replace("++", "plusplus")
    value = value.replace("&", " and ")
    value = re.sub(r"['`*_]", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return re.sub(r"\s+", " ", value)


def canonicalize(raw: str) -> Optional[str]:
    if not raw:
        return None
    normalized = normalize_name(raw)
    if not normalized:
        return None
    if normalized.replace(" ", "_") in GENERIC_METHOD_STOPWORDS:
        return None
    if normalized in EXACT_NAME_MAP:
        return EXACT_NAME_MAP[normalized]
    underscored = normalized.replace(" ", "_")
    if underscored in EXACT_NAME_MAP:
        return EXACT_NAME_MAP[underscored]
    for canon, patterns in ALIASES.items():
        if any(pattern.search(raw) or pattern.search(normalized) for pattern in patterns):
            return canon
    # Keep only plausible method-looking names from markdown baseline lists.
    if 2 <= len(normalized) <= 40 and not re.search(
        r"\b(metric|dataset|notes|label|nodes|samples|function|primary|secondary|available|task)\b",
        normalized,
    ):
        return underscored
    return None


def scan_known_aliases(text: str) -> Set[str]:
    found: Set[str] = set()
    for canon, patterns in ALIASES.items():
        if any(pattern.search(text) for pattern in patterns):
            found.add(canon)
    return found


def baseline_relevant_text(description: str) -> str:
    lines = description.splitlines()
    captured: List[str] = []
    in_section = False
    transient = 0
    for line in lines:
        lower = line.lower()
        is_heading = bool(re.match(r"^#{1,4}\s+", line))
        if is_heading:
            in_section = bool(
                re.search(
                    r"\b(baselines?|reference implementations?|reference methods?|classic algorithms?)\b",
                    lower,
                )
            )
            transient = 0
        if in_section:
            captured.append(line)
            continue
        if re.search(
            r"\b(reference baselines?|classic(?:al)? (?:strategies|approaches|algorithms)|"
            r"standard (?:approaches|methods)|existing approaches|state-of-the-art combinations)\b",
            lower,
        ):
            captured.append(line)
            transient = 8
            continue
        if transient > 0:
            if line.strip() or captured[-1].strip():
                captured.append(line)
            transient -= 1
    return "\n".join(captured)


def extract_markdown_method_names(text: str) -> Set[str]:
    methods: Set[str] = set()
    for match in re.finditer(r"\*\*([^*\n]{2,80})\*\*", text):
        name = match.group(1).strip()
        if any(word in name.lower() for word in ("metric", "input", "output", "train", "test")):
            continue
        canon = canonicalize(name)
        if canon:
            methods.add(canon)
    for match in re.finditer(r"^\s*[-*]\s+`([^`\n]{1,60})`", text, flags=re.MULTILINE):
        canon = canonicalize(match.group(1))
        if canon:
            methods.add(canon)
    for match in re.finditer(r"^\s*[-*]\s+([^:\n]{2,80}?)(?:\s+[-—:()]|\s*$)", text, flags=re.MULTILINE):
        raw = re.sub(r"\[.*?\]\(.*?\)", "", match.group(1)).strip()
        canon = canonicalize(raw)
        if canon:
            methods.add(canon)
    table_lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    for line in table_lines:
        cells = [cell.strip(" `*") for cell in line.strip().strip("|").split("|")]
        if not cells or re.fullmatch(r"[-:\s]+", cells[0] or ""):
            continue
        if cells[0].lower() in {"name", "baseline", "method"}:
            continue
        canon = canonicalize(cells[0])
        if canon:
            methods.add(canon)
    return methods


def extract_baselines(task_description: str) -> Tuple[Set[str], List[str]]:
    notes: List[str] = []
    relevant = baseline_relevant_text(task_description)
    if not relevant.strip():
        relevant = task_description
        notes.append("no_explicit_baseline_section")
    methods = extract_markdown_method_names(relevant)
    methods |= scan_known_aliases(relevant)
    methods -= COMMON_INFRA
    if not methods:
        notes.append("baselines_unclear")
    return methods, notes


def added_code_from_log(log_text: str) -> str:
    clean = strip_ansi(log_text)
    added: List[str] = []
    in_edit = False
    for line in clean.splitlines():
        if re.search(r"Step\s+\d+\s+edit", line):
            in_edit = True
            continue
        if in_edit and re.search(r"Step\s+\d+\s+(?:test|submit|undo)", line):
            in_edit = False
        if not in_edit:
            continue
        if line.startswith("+++"):
            continue
        match = re.match(r"\+\s*\d+\s*\|\s?(.*)$", line)
        if match:
            added.append(match.group(1))
            continue
        match = re.match(r"\+\s?(.*)$", line)
        if match and not match.group(1).startswith("++"):
            added.append(match.group(1))
    return "\n".join(added)


def split_identifier(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"[_\-]+", " ", spaced)
    return spaced


def extract_import_techniques(code_text: str) -> Set[str]:
    techniques: Set[str] = set()
    for line in code_text.splitlines():
        line = line.strip()
        if line.startswith("import "):
            names = re.split(r",\s*", line[len("import ") :])
            for name in names:
                root = name.split(" as ")[0].split(".")[0].strip()
                if root in {"optuna", "xgboost", "lightgbm", "catboost", "pyod", "torch"}:
                    canon = canonicalize(root)
                    if canon:
                        techniques.add(canon)
        elif line.startswith("from "):
            match = re.match(r"from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)", line)
            if not match:
                continue
            module, imported = match.group(1), match.group(2)
            root = module.split(".")[0]
            if root in {"optuna", "xgboost", "lightgbm", "catboost", "pyod"}:
                canon = canonicalize(root)
                if canon:
                    techniques.add(canon)
            for piece in re.split(r",\s*", imported):
                name = piece.split(" as ")[0].strip()
                techniques |= scan_known_aliases(split_identifier(name))
    return techniques


def extract_identifier_techniques(code_text: str) -> Set[str]:
    techniques: Set[str] = set()
    for match in re.finditer(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", code_text):
        techniques |= scan_known_aliases(split_identifier(match.group(1)))
    return techniques


def extract_log_techniques(log_text: str) -> Tuple[Set[str], str]:
    code_text = added_code_from_log(log_text)
    techniques = set()
    techniques |= extract_import_techniques(code_text)
    techniques |= extract_identifier_techniques(code_text)
    techniques |= scan_known_aliases(code_text)
    # Keep bare scientific packages only when they name a domain-specific package.
    techniques -= COMMON_INFRA
    return techniques, code_text


def jaccard(a: Set[str], b: Set[str]) -> Optional[float]:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def detect_fallback(
    log_text: str,
    code_text: str,
    techniques: Set[str],
    baselines: Set[str],
) -> Tuple[bool, List[str]]:
    clean = strip_ansi(log_text)
    reasons: List[str] = []
    final_failure_patterns = [
        r"No metric values found",
        r"Missing final results",
        r"\[agent\] No action returned after 3 attempts",
        r"\[done\] Summary: .*'done': False",
        r'"done"\s*:\s*false',
    ]
    if any(re.search(pattern, clean, flags=re.IGNORECASE | re.DOTALL) for pattern in final_failure_patterns):
        reasons.append("final_failure_or_missing_metrics")
    if reasons and re.search(r"Traceback|\[STATUS: FAILED|COMMAND FAILED|✘ ERROR", clean):
        reasons.append("errored_during_run")
    simple_random = bool(re.search(r"\bsample_uniform\b|\brandom\s*(?:search|sampling)?\b", code_text, re.IGNORECASE))
    nonbaseline = techniques - baselines
    if techniques and not nonbaseline and simple_random:
        reasons.append("baseline_only_random_or_default")
    if not techniques and clean.strip():
        reasons.append("no_proposed_techniques_extracted")
    return bool(reasons), reasons


def analyze_model(task: str, model_name: str, log_dir: Path, baselines: Set[str]) -> ModelResult:
    path = log_dir / f"{task}.log"
    if not path.exists():
        return ModelResult(status="log_missing", notes=["log_missing"])
    try:
        log_text = path.read_text(encoding="utf-8", errors="replace")
        techniques, code_text = extract_log_techniques(log_text)
        score = jaccard(techniques, baselines)
        beyond = bool(techniques - baselines)
        is_fallback, reasons = detect_fallback(log_text, code_text, techniques, baselines)
        status = "fallback" if is_fallback else "ok"
        return ModelResult(
            techniques=techniques,
            jaccard=score,
            beyond=beyond,
            status=status,
            notes=reasons,
        )
    except Exception as exc:  # Keep the run going per task.
        return ModelResult(status="error", notes=[f"{type(exc).__name__}: {exc}"])


def sentence_transformers_note() -> str:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as exc:
        return f"sentence-transformers unavailable offline ({type(exc).__name__}: {exc}); cosine similarity not computed."
    return (
        "sentence-transformers import succeeded, but this script did not load or download a model; "
        "cosine similarity not computed to keep the run offline-only."
    )


def fmt_set(values: Set[str], empty: str = "none") -> str:
    if not values:
        return empty
    return ", ".join(sorted(values))


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def fmt_bool(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "True" if value else "False"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def model_cell(result: ModelResult) -> str:
    if result.status == "log_missing":
        return "log_missing"
    if result.status == "error":
        return "ERROR: " + "; ".join(result.notes)
    prefix = ""
    if result.status == "fallback":
        prefix = "FALLBACK (" + ", ".join(result.notes) + "): "
    elif result.techniques and result.beyond is False:
        prefix = "BASELINE_ONLY: "
    return prefix + fmt_set(result.techniques, "none")


def summarize(results: List[TaskResult]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for model in LOG_DIRS:
        available = [row.models[model] for row in results if row.models[model].status != "log_missing"]
        scores = [res.jaccard for res in available if res.jaccard is not None]
        summary[model] = {
            "available": len(available),
            "mean_jaccard": statistics.mean(scores) if scores else math.nan,
            "beyond_count": sum(1 for res in available if res.beyond),
            "fallback_count": sum(1 for res in available if res.status == "fallback"),
            "error_count": sum(1 for res in available if res.status == "error"),
        }
    return summary


def verdict(summary: Dict[str, Dict[str, object]]) -> str:
    base_mean = float(summary["BASE"]["mean_jaccard"])
    mt_mean = float(summary["OURS_methodtraj"]["mean_jaccard"])
    v4_mean = float(summary["OURS_methodv4"]["mean_jaccard"])
    base_beyond = int(summary["BASE"]["beyond_count"])
    mt_beyond = int(summary["OURS_methodtraj"]["beyond_count"])
    v4_beyond = int(summary["OURS_methodv4"]["beyond_count"])

    parts: List[str] = []
    if mt_mean < base_mean and mt_beyond >= base_beyond:
        parts.append("OURS_methodtraj is less similar to named baselines on average and is at least as often beyond-baseline as BASE.")
    else:
        parts.append("OURS_methodtraj is not plainly more novel than BASE by both metrics.")
    if v4_mean < base_mean and v4_beyond >= base_beyond:
        parts.append("OURS_methodv4 is less similar to named baselines on average and is at least as often beyond-baseline as BASE.")
    else:
        parts.append("OURS_methodv4 is not plainly more novel than BASE by both metrics.")
    if not (mt_mean < base_mean or v4_mean < base_mean):
        parts.append("In Jaccard terms, OURS is not more novel than BASE on average.")
    elif mt_mean < base_mean and v4_mean < base_mean:
        parts.append("Both OURS variants have lower mean Jaccard-to-baseline than BASE.")
    else:
        parts.append("Only one OURS variant has lower mean Jaccard-to-baseline than BASE.")
    return " ".join(parts)


def render_report(results: List[TaskResult], skipped_no_logs: int, st_note: str) -> str:
    summary = summarize(results)
    lines: List[str] = []
    lines.append("# MLS-Bench Similarity-to-Baselines Analysis")
    lines.append("")
    lines.append("## Extraction Method")
    lines.append(
        "Baselines were extracted from each `task_description.md` using markdown/regex heuristics: "
        "explicit baseline/reference sections, bold method names, bullet names, code-form method names, "
        "and first-column entries in comparison tables. Proposed techniques were extracted from model log "
        "edit additions rather than the initial prompt, using imports, added class/function/docstring text, "
        "and a hardcoded algorithm vocabulary. Generic infrastructure imports such as `numpy`, `scipy`, "
        "and `sklearn` were not counted as technique tokens unless an imported estimator or package mapped "
        "to a named algorithm."
    )
    lines.append(
        "Jaccard similarity is `|proposed techniques ∩ extracted baselines| / "
        "|proposed techniques ∪ extracted baselines|`. `beyond_baseline` is true when at least one extracted "
        "proposed technique token is absent from the extracted baseline set. FALLBACK indicates final missing "
        "metrics, unrecovered agent failure, no extractable proposal, or a simple baseline/default method."
    )
    lines.append("")
    lines.append("## Per-Task Table")
    header = [
        "task",
        "baselines_extracted",
        "OURS_methodtraj_techniques",
        "OURS_methodtraj_jaccard",
        "OURS_methodtraj_beyond?",
        "OURS_methodv4_techniques",
        "OURS_methodv4_jaccard",
        "OURS_methodv4_beyond?",
        "BASE_techniques",
        "BASE_jaccard",
        "BASE_beyond?",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in results:
        baseline_text = fmt_set(row.baselines, "baselines_unclear")
        if row.baseline_notes:
            baseline_text += " (" + ", ".join(row.baseline_notes) + ")"
        mt = row.models["OURS_methodtraj"]
        v4 = row.models["OURS_methodv4"]
        base = row.models["BASE"]
        cells = [
            row.task,
            baseline_text,
            model_cell(mt),
            fmt_float(mt.jaccard),
            fmt_bool(mt.beyond),
            model_cell(v4),
            fmt_float(v4.jaccard),
            fmt_bool(v4.beyond),
            model_cell(base),
            fmt_float(base.jaccard),
            fmt_bool(base.beyond),
        ]
        lines.append("| " + " | ".join(md_escape(cell) for cell in cells) + " |")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append(f"Evaluated tasks with at least one available log: {len(results)}.")
    lines.append(f"Tasks discovered but skipped because all three logs were missing: {skipped_no_logs}.")
    lines.append("")
    lines.append("| model | available_logs | mean_jaccard_to_baselines | beyond_baseline_count | FALLBACK_count | error_count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model, stats in summary.items():
        mean_value = float(stats["mean_jaccard"])
        mean_text = "n/a" if math.isnan(mean_value) else f"{mean_value:.3f}"
        lines.append(
            f"| {model} | {stats['available']} | {mean_text} | {stats['beyond_count']} | "
            f"{stats['fallback_count']} | {stats['error_count']} |"
        )
    lines.append("")
    lines.append("## Honest Caveats")
    lines.append(
        "This is a lexical technique-fingerprint analysis, not semantic proof of novelty. "
        "A method can mention a new token without implementing a genuinely new algorithm, and a strong "
        "implementation of a baseline can still look baseline-similar. FALLBACK rows should not be read as "
        "novel proposals even when an attempted edit contained non-baseline vocabulary."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append(verdict(summary))
    lines.append("")
    lines.append("## Sentence-Transformers Availability")
    lines.append(st_note)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    task_names = sorted(path.name for path in TASKS_DIR.iterdir() if path.is_dir())
    results: List[TaskResult] = []
    skipped_no_logs = 0
    for task in task_names:
        has_any_log = any((log_dir / f"{task}.log").exists() for log_dir in LOG_DIRS.values())
        if not has_any_log:
            skipped_no_logs += 1
            continue
        row = TaskResult(task=task)
        try:
            desc_path = TASKS_DIR / task / "task_description.md"
            if not desc_path.exists():
                row.baseline_notes.append("task_description_missing")
            else:
                desc = desc_path.read_text(encoding="utf-8", errors="replace")
                row.baselines, row.baseline_notes = extract_baselines(desc)
            for model, log_dir in LOG_DIRS.items():
                row.models[model] = analyze_model(task, model, log_dir, row.baselines)
        except Exception as exc:
            row.errors.append(f"{type(exc).__name__}: {exc}")
            for model in LOG_DIRS:
                row.models.setdefault(model, ModelResult(status="error", notes=row.errors[:]))
        results.append(row)
    print(render_report(results, skipped_no_logs, sentence_transformers_note()))


if __name__ == "__main__":
    main()
