Financial return prediction is a noisy, non-stationary tabular problem. A single strong learner such as LightGBM captures nonlinear factor interactions, but it has no internal way to distinguish learnable hard cases from pure noise, and it uses the same full feature set throughout training. That leaves it vulnerable to two pathologies: late trees chase unlearnable residuals, and the model remains overexposed to factors that are currently strong but may be stale in the next regime. Uniform bagging or random-seed ensembles reduce variance, yet they spread diversity blindly across rows and columns rather than directing it where it helps. Simple boosting-style reweighting is risky because the highest-loss rows in financial data are often shocks, not useful boundaries. Static feature filters either freeze one subset or rank features once and discard potentially useful dormant factors. What is needed is an ensemble wrapper that deliberately changes both the sample weights and the feature subset for each new member, using signals computed from the ensemble so far.

The method that addresses this is DoubleEnsemble. It is a sequential ensemble built around an ordinary LightGBM base learner. Members are trained one after another, and after each member the wrapper updates the row weights and selects a new feature subset before the next member is trained. The sample reweighting module combines two signals. The first is the current ensemble loss: harder rows receive higher weights through an inverse-rank transformation, preserving a boosting-like pressure on remaining error. The second signal is the previous member's per-tree loss curve. By comparing each row's rank-normalized loss at the beginning and end of that member's training, the method identifies rows whose relative error descends and are therefore likely learnable, while rows whose relative error stays flat or rises are down-weighted as noisy. These two signals are combined into a single hardness score, binned, and turned into per-bin weights with an annealed denominator. Early members use the reweighting sharply, while later members move toward uniform weights so the ensemble does not keep escalating the chase.

The feature selection module uses permutation reliance. Each feature is shuffled in turn, and the change in the current ensemble's loss is measured. The score is the mean loss increase divided by its standard deviation, so a feature does not look important merely because a few noisy samples moved. Features are binned by this score and sampled at declining but nonzero ratios from each bin. Strong features dominate the next member, yet weaker or dormant factors remain represented, which matters when the market regime shifts. All adaptation happens during training; inference is simply a weighted average of the member predictions, where each member predicts using the feature subset it was trained on.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Text, Union
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.interpret.base import FeatureInt


class DoubleEnsemble(Model, FeatureInt):
    def __init__(
        self,
        base_model="gbm",
        loss="mse",
        num_models=3,
        enable_sr=True,
        enable_fs=True,
        alpha1=1.0,
        alpha2=1.0,
        bins_sr=10,
        bins_fs=5,
        decay=0.5,
        sample_ratios=None,
        sub_weights=None,
        epochs=28,
        early_stopping_rounds=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.num_models = num_models
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        if sample_ratios is None:
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:
            sub_weights = [1] * num_models
        if len(sample_ratios) != bins_fs:
            raise ValueError("sample_ratios length must equal bins_fs")
        if len(sub_weights) != num_models:
            raise ValueError("sub_weights length must equal num_models")
        self.sample_ratios = sample_ratios
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.ensemble = []
        self.sub_features = []
        self.params = {"objective": loss}
        _bench = {
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1,
        }
        for k, v in _bench.items():
            kwargs.setdefault(k, v)
        self.params.update(kwargs)
        self.loss = loss

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset")
        x_train, y_train = df_train["feature"], df_train["label"]
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models)), index=x_train.index)

        for k in range(self.num_models):
            self.sub_features.append(features)
            model_k = self._train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)
            if k + 1 == self.num_models:
                break

            loss_curve = self._retrieve_loss_curve(model_k, df_train, features)
            pred_k = self._predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = (
                pred_sub.iloc[:, : k + 1] * self.sub_weights[: k + 1]
            ).sum(axis=1) / np.sum(self.sub_weights[: k + 1])
            loss_values = pd.Series(self._get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if self.enable_sr:
                weights = self._sample_reweight(loss_curve, loss_values, k + 1)
            if self.enable_fs:
                features = self._feature_selection(df_train, loss_values)
        return self

    def _train_submodel(self, df_train, df_valid, weights, features):
        dtrain, dvalid = self._prepare_data_gbm(df_train, df_valid, weights, features)
        callbacks = [lgb.log_evaluation(20), lgb.record_evaluation({})]
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        return lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.epochs,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

    def _prepare_data_gbm(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM does not support multi-label training")
        return (
            lgb.Dataset(x_train, label=y_train, weight=weights),
            lgb.Dataset(x_valid, label=y_valid),
        )

    def _sample_reweight(self, loss_curve, loss_values, k_th):
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)
        N, T = loss_curve.shape
        part = max(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay ** k_th * h_avg[b] + 0.1)
        return weights

    def _feature_selection(self, df_train, loss_values):
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F)})
        M = len(self.ensemble)
        x_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_tmp.loc[:, feat] = np.random.permutation(x_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_tmp.loc[:, self.sub_features[i_s]].values),
                        index=x_tmp.index,
                    )
                    / M
                )
            loss_feat = self._get_loss(y_train.values.squeeze(), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (
                np.std(loss_feat - loss_values) + 1e-7
            )
            x_tmp.loc[:, feat] = x_train.loc[:, feat].copy()
        g["g_value"].replace(np.nan, 0, inplace=True)
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)
        res_feat = []
        for i_b, b in enumerate(sorted(g["bins"].unique(), reverse=True)):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat.extend(
                np.random.choice(b_feat, size=num_feat, replace=False).tolist()
            )
        return pd.Index(set(res_feat))

    def _get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        raise ValueError("Loss not implemented")

    def _retrieve_loss_curve(self, model, df_train, features):
        if self.base_model != "gbm":
            raise ValueError("Only gbm base_model is supported")
        num_trees = model.num_trees()
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train = np.squeeze(y_train.values)
        else:
            raise ValueError("LightGBM does not support multi-label training")
        N = x_train.shape[0]
        loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
        pred_tree = np.zeros(N)
        for i_tree in range(num_trees):
            pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
            loss_curve.iloc[:, i_tree] = self._get_loss(y_train, pred_tree)
        return loss_curve

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.ensemble:
            raise ValueError("Model is not fitted")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            pred += (
                pd.Series(
                    submodel.predict(x_test.loc[:, self.sub_features[i_sub]].values),
                    index=x_test.index,
                )
                * self.sub_weights[i_sub]
            )
        return pred / np.sum(self.sub_weights)

    def _predict_sub(self, submodel, df_data, features):
        x_data = df_data["feature"].loc[:, features]
        return pd.Series(submodel.predict(x_data.values), index=x_data.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        res = []
        for model, weight in zip(self.ensemble, self.sub_weights):
            res.append(
                pd.Series(
                    model.feature_importance(*args, **kwargs), index=model.feature_name()
                )
                * weight
            )
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)
```
