## Research Question

A financial return predictor built from hundreds of engineered factors has to work in a regime where ordinary tabular assumptions are brittle. The target is usually a short-horizon return, return rank, or trading signal for many instruments over time. The available learner can be strong, for example a gradient-boosted tree or a neural net, but the data has a low signal-to-noise ratio, hidden market shocks, and non-stationary factor behavior. A useful model wrapper must therefore improve out-of-sample stability without changing the trading interface: train on a temporal training window, validate on later dates, and emit one score per instrument-date in a still-later test or trading period.

The open design problem is how to build several related predictors when not every row and not every factor deserves the same treatment. Some samples are easy, some are difficult but informative, and some are dominated by noise. Some factors are consistently useful, some are redundant, and some look weak under the current regime but may still be useful under a later one. Existing infrastructure can train weighted learners on chosen feature columns; what remains unsettled is how to choose those row weights and feature subsets for each successive member.

## Background

Financial prediction starts from the multifactor view: each instrument-date is represented by a vector of factors such as technical indicators, valuation ratios, order-book summaries, or proprietary alphas. Linear multifactor models are interpretable and expose collinearity problems, but they underfit nonlinear interactions. Gradient boosting decision trees and neural networks can capture richer patterns, yet their capacity makes them sensitive to label noise and regime drift.

The temporal split is not a detail. Randomly shuffling stock-day samples would hide the real problem, because the future period is not drawn from exactly the same distribution as the past. A model can look accurate on the training history while relying on factor relationships that are already stale. The strongest baseline must therefore be judged on later periods and on trading metrics, not only on average supervised loss.

Ensemble learning supplies a natural robustness tool. Bagging generates multiple versions of a predictor from bootstrap replicates and averages them, which Breiman introduced as a way to stabilize unstable learners. Boosting instead trains learners sequentially under changing example weights, using multiplicative weight updates to emphasize parts of the training set that the current combined rule handles poorly. In finance, the tension is that a hard row can be either a useful boundary case or a noisy event with no repeatable signal.

Feature selection is the other pressure point. A trading desk or factor library can produce hundreds or thousands of columns, and training on all of them may increase variance or lock the learner onto redundant factors. Permutation-style importance asks how performance changes when one feature is disrupted, while model-specific importances such as tree gain or coefficients expose only one learner's internal view. Any practical wrapper has to work with the same matrix API used by the learner and avoid retraining a new model for every candidate feature.

## Baselines

**Single LightGBM / GBDT.** A gradient-boosted tree is a strong tabular baseline: each new tree fits residual or gradient information left by the current model, and LightGBM makes this practical at large scale. The limitation is that one long boosting run can spend late capacity on idiosyncratic noise, and it uses the same feature set throughout.

**Bagging and random-seed ensembles.** Independent members trained on bootstrap samples or different seeds reduce variance through averaging. The limitation is that their row and feature diversity is mostly uniform or accidental; they do not distinguish easy rows, informative hard rows, noisy rows, or regime-sensitive columns.

**Boosting-style reweighting.** AdaBoost-like procedures maintain a distribution over examples and increase emphasis where the current learner struggles. The limitation is noise sensitivity: cited boosting analyses show that random classification noise can defeat broad convex-potential boosting families, and in financial labels the highest-loss rows are often not the rows one wants to chase.

**Self-paced and denoising reweighting.** Self-paced and curriculum-style methods use training feedback to avoid putting too much emphasis on outliers too early, often through a hardness schedule. The limitation is that a single scalar loss or hardness value can still conflate a useful hard case with an unlearnable noisy one.

**Static feature filters and permutation selection.** Manual filters, mutual information, tree gain, and permutation loss changes can rank features before training or after fitting a model. The limitation is that a one-time subset becomes another fixed modeling choice; it does not by itself create diverse ensemble members under shifting regimes.

## Evaluation Settings

The relevant supervised protocol uses chronological train, validation, and test windows. In qlib's public Alpha158 CSI300 workflow, the data handler runs from 2008-01-01 to 2020-08-01, with train dates 2008-01-01 through 2014-12-31, validation dates 2015-01-01 through 2016-12-31, and test/backtest dates starting 2017-01-01. The model output is a score series indexed by instrument and datetime.

Signal metrics include information coefficient (IC), rank IC, and their information ratios across days. Trading metrics include annualized return, risk-adjusted return such as Sharpe or information ratio, and maximum drawdown under a fixed portfolio rule. For high-frequency cryptocurrency settings, classification-style precision, AUC, F1, and realized return per retrieved signal are also relevant.

Comparisons should isolate the wrapper from the base learner. A fair run keeps the base learner family and data split fixed, then compares a single model, a simple ensemble, a randomly diversified ensemble, row-reweighting variants, feature-selection variants, and the combined wrapper.

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
