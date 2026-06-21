## Research Question

A financial return predictor is built from hundreds of engineered factors. The target is usually a short-horizon return, return rank, or trading signal for many instruments over time. The available learner can be a gradient-boosted tree or a neural net, and the data has a low signal-to-noise ratio, market shocks, and non-stationary factor behavior. A model wrapper trains on a temporal training window, validates on later dates, and emits one score per instrument-date in a still-later test or trading period.

The design problem is how to build several related predictors as members of a sequential ensemble. Existing infrastructure can train weighted learners on chosen feature columns; the open question is how to choose the per-row weights and the feature subset for each successive member.

## Background

Financial prediction starts from the multifactor view: each instrument-date is represented by a vector of factors such as technical indicators, valuation ratios, order-book summaries, or proprietary alphas. Linear multifactor models are interpretable and expose collinearity. Gradient boosting decision trees and neural networks capture richer nonlinear patterns.

The temporal split is part of the protocol: training, validation, and test windows are chronological rather than randomly shuffled, because the future period is not drawn from exactly the same distribution as the past. Models are judged on later periods and on trading metrics, not only on average supervised loss.

Ensemble learning supplies a robustness tool. Bagging generates multiple versions of a predictor from bootstrap replicates and averages them, which Breiman introduced as a way to stabilize unstable learners. Boosting instead trains learners sequentially under changing example weights, using multiplicative weight updates to emphasize parts of the training set that the current combined rule handles poorly.

Feature selection is the other axis. A trading desk or factor library can produce hundreds or thousands of columns. Permutation-style importance asks how performance changes when one feature is disrupted, while model-specific importances such as tree gain or coefficients expose one learner's internal view. A practical wrapper works with the same matrix API used by the learner.

## Baselines

**Single LightGBM / GBDT.** A gradient-boosted tree is a strong tabular baseline: each new tree fits residual or gradient information left by the current model, and LightGBM makes this practical at large scale, using one feature set throughout.

**Bagging and random-seed ensembles.** Independent members trained on bootstrap samples or different seeds reduce variance through averaging, with row and feature diversity that is uniform or seed-driven.

**Boosting-style reweighting.** AdaBoost-like procedures maintain a distribution over examples and increase emphasis where the current learner struggles. A line of boosting analyses studies how random classification noise interacts with convex-potential boosting families.

**Self-paced and denoising reweighting.** Self-paced and curriculum-style methods use training feedback to control how much emphasis outliers receive early in training, often through a hardness schedule built from a scalar loss or hardness value.

**Static feature filters and permutation selection.** Manual filters, mutual information, tree gain, and permutation loss changes rank features before training or after fitting a model, producing a fixed subset.

## Evaluation Settings

The relevant supervised protocol uses chronological train, validation, and test windows. In qlib's public Alpha158 CSI300 workflow, the data handler runs from 2008-01-01 to 2020-08-01, with train dates 2008-01-01 through 2014-12-31, validation dates 2015-01-01 through 2016-12-31, and test/backtest dates starting 2017-01-01. The model output is a score series indexed by instrument and datetime.

Signal metrics include information coefficient (IC), rank IC, and their information ratios across days. Trading metrics include annualized return, risk-adjusted return such as Sharpe or information ratio, and maximum drawdown under a fixed portfolio rule. For high-frequency cryptocurrency settings, classification-style precision, AUC, F1, and realized return per retrieved signal are also relevant.

Comparisons isolate the wrapper from the base learner: a run keeps the base learner family and data split fixed, then compares a single model, a simple ensemble, a randomly diversified ensemble, row-reweighting variants, feature-selection variants, and the combined wrapper.

## Code Framework

The pre-existing code shape is an ordinary qlib-compatible model wrapper: prepare train and validation data, train a sequence of base learners, store the feature columns used by each member, and average their predictions at inference. The undecided parts are the two update hooks between members.

```python
import numpy as np
import pandas as pd


class SequentialFinancialEnsemble:
    def __init__(self, num_models=3, sub_weights=None, base_params=None):
        self.num_models = num_models
        self.sub_weights = sub_weights or [1] * num_models
        self.base_params = {} if base_params is None else base_params
        self.ensemble = []
        self.sub_features = []

    def fit(self, dataset):
        df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
        x_train = df_train["feature"]
        weights = pd.Series(np.ones(x_train.shape[0], dtype=float), index=x_train.index)
        features = x_train.columns
        predictions = pd.DataFrame(np.zeros((x_train.shape[0], self.num_models)), index=x_train.index)

        for k in range(self.num_models):
            self.sub_features.append(features)
            model = self._train_member(df_train, df_valid, weights, features)
            self.ensemble.append(model)
            if k + 1 == self.num_models:
                break

            member_state = self._member_training_state(model, df_train, features)
            predictions.iloc[:, k] = self._predict_member(model, df_train, features)
            ensemble_state = self._ensemble_training_state(df_train, predictions.iloc[:, : k + 1], k)
            weights = self._update_sample_weights(member_state, ensemble_state, k + 1)
            features = self._choose_feature_subset(df_train, ensemble_state)
        return self

    def _train_member(self, df_train, df_valid, weights, features):
        raise NotImplementedError

    def _member_training_state(self, model, df_train, features):
        raise NotImplementedError

    def _ensemble_training_state(self, df_train, predictions, k):
        raise NotImplementedError

    def _update_sample_weights(self, member_state, ensemble_state, k_th):
        raise NotImplementedError

    def _choose_feature_subset(self, df_train, ensemble_state):
        raise NotImplementedError

    def _predict_member(self, model, df_data, features):
        raise NotImplementedError

    def predict(self, dataset, segment="test"):
        x_test = dataset.prepare(segment, col_set="feature")
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i, model in enumerate(self.ensemble):
            pred += pd.Series(
                model.predict(x_test.loc[:, self.sub_features[i]].values),
                index=x_test.index,
            ) * self.sub_weights[i]
        return pred / np.sum(self.sub_weights)
```
