LightGBM did exactly what I hoped and a bit more, and its numbers tell me where the next gain must come from. On the long `csi300` window it is the best of the three baselines on signal — IC 0.0457, rank IC 0.0569 — with the shallowest max drawdown of all, $-0.0579$ against the adaptive models' $-0.09$ to $-0.10$, and info ratio 1.35. On the far `csi300_recent` window that killed both adaptive models at info ratio $\approx 0.48$, the tree does *not* die: info ratio 1.10, annualized return 0.0807, more than double either adaptive model's. So the aggregate gmean puts LightGBM clearly on top ($\approx 0.571$ vs $\approx 0.560$–$0.561$). But its own numbers expose two soft spots. First, the near window: AdaRNN's explicit alignment beat the tree on `csi300_shifted` info ratio (1.66 vs 1.30), so there is drift-adaptation value the single tree leaves on the table where the shift is mild. Second, and more fundamental, a lone GBDT has no defense beyond its leaf penalties against this data's two pathologies — it overfits the abundant noise (late trees chase unlearnable residuals) and it uses *all* features *all* the time, so it cannot shed regime-stale factors or down-weight unlearnable samples. The move is not to abandon the tree but to make *it* drift-robust by ensembling it deliberately.

I propose DoubleEnsemble — a general ensemble framework wrapped around the winning LightGBM base learner, deliberate about the two axes a plain bagged or boosted ensemble leaves uniform: which *samples* each new member focuses on, and which *features* it may use. Members are trained sequentially; after each member, two modules driven by the current ensemble's per-sample loss set up the next.

Take the sample axis first, because that is where the noise pathology lives. A single GBDT's late trees chase residuals, and on return data those residuals are disproportionately unlearnable noise — a stock that jumped on an unforecastable headline has huge loss and nothing to learn from. The boosting reflex (weight up the high-current-loss samples) is therefore the *wrong sign* here: it leans into the noise, which is exactly why a single deep-boosted tree needs the enormous $\texttt{lambda\_l1}$/$\texttt{lambda\_l2}$ to survive. I need a per-sample importance that distinguishes hard-*but-learnable* from hard-*because-noise*, and the scalar final loss cannot, since both have high loss. The richer signal is free: a GBDT produces, for each sample, the *trajectory* of its loss as trees are added. A sample whose loss collapses in the first few low-variance trees was learned from transferable structure; a sample whose loss stays high and only falls late, when the model is memorizing, is noise. So the start-to-end shape of the loss curve separates them. The Sample Reweighting module rank-normalizes each sample's loss curve down the sample axis at each iteration, averages the first 10% of iterations into $l_{\text{start}}$ and the last 10% into $l_{\text{end}}$, and forms the curve statistic $h_2 = \mathrm{rank}(l_{\text{end}}/l_{\text{start}})$ — small ratio means the loss collapsed (well-behaved), large means it stalled. It blends this with the current-ensemble error $h_1$ (rank-normalized),

$$h = \alpha_1 h_1 + \alpha_2 h_2,$$

so it still leans toward residual error like boosting but the $h_2$ term injects the learnability discrimination that keeps it off pure noise. To stabilize, it bins samples by $h$ into $\texttt{bins\_sr}=10$ bins and assigns one weight per bin, $w = 1/(\texttt{decay}^k \cdot h_{\text{avg}} + 0.1)$ — binning regularizes the weighting, $\texttt{decay}^k$ (with $\texttt{decay}=0.5$) anneals the chase as members accumulate so later members reweight gently, and $+0.1$ floors the denominator. This focuses each member on under-served but learnable samples instead of letting heavy leaf penalties bluntly suppress everything.

Now the feature axis, where the drift pathology lives — the one the sequence models tried and failed to handle, and the one the single LGBM ignores by using every factor always. I want each successive member trained on a *different, deliberately chosen* feature subset, so the ensemble does not collapse onto one clique of factors that may be regime-specific. The wrong way is a single global importance pruning — keep the top factors once — because that bakes in the present regime and discards factors that will matter when the regime turns, the exact mistake that hurt `csi300_recent`. So the Feature Selection module measures each factor's *reliance* by permutation — shuffle its column, run the current ensemble, take the standardized loss increase $g = \mathrm{mean}(\Delta\text{loss})/\mathrm{std}(\Delta\text{loss})$ — then bins factors by $g$ into $\texttt{bins\_fs}=5$ bins and samples *from every bin* at declining ratios $\texttt{sample\_ratios}=[0.8,0.7,0.6,0.5,0.4]$: keep most of the load-bearing bin but always retain a tail of currently-weak factors, never zero. That deliberate inclusion of dormant factors gives the ensemble exposure to relations that are quiet now but revive in a shifted regime — drift-robustness built on the feature axis rather than by aligning representations. It is the move the sequence models gestured at and could not land on a thin panel, expressed in a way the strong tree control can actually use.

The two modules alternate off the current ensemble's loss: train member $k$, evaluate the ensemble-so-far on the training data to get the loss values and member $k$'s loss curve, run SR to set member $k{+}1$'s sample weights and FS to set its feature subset, train member $k{+}1$, repeat. The last member needs neither (nothing follows it). At inference the whole SR/FS apparatus is gone — `predict` just runs each member on its own stored feature subset and averages, weighted by uniform $\texttt{sub\_weights}$. So, like the adaptive models, all the adaptation is training-time; unlike them, the inference is a plain tree-ensemble average with no encoder to drift.

The base LightGBM params are deliberately *identical to the winning LGBM baseline* — $\texttt{learning\_rate}=0.2$, $\texttt{num\_leaves}=210$, $\texttt{max\_depth}=8$, $\texttt{lambda\_l1}=205.6999$, $\texttt{lambda\_l2}=580.9768$, $\texttt{colsample\_bytree}=0.8879$, $\texttt{subsample}=0.8789$ — so the finale isolates the *framework*: an ensemble of the exact tree that already won, made drift-robust by SR and FS. The DoubleEnsemble settings are $\texttt{num\_models}=3$, $\texttt{enable\_sr}=\texttt{enable\_fs}=\texttt{True}$, $\alpha_1=\alpha_2=1$, $\texttt{bins\_sr}=10$, $\texttt{bins\_fs}=5$, $\texttt{decay}=0.5$, $\texttt{sample\_ratios}=[0.8,0.7,0.6,0.5,0.4]$, $\texttt{sub\_weights}=[1,1,1]$, and crucially $\texttt{epochs}=28$ — *short* members (28 rounds, not the $\approx 1000$ a single GBDT used), because the ensemble, not boosting depth, does the variance reduction and short members resist memorizing the noise. The scaffold fit is the cleanest of the four rungs: it fills `custom_model.py` with the ensemble and *leaves the workflow at the default* `DatasetH` + Alpha158 — the same full-158-raw-factor view the winning LGBM used, no YAML edit.

The bar this must clear is LGBM's aggregate $\approx 0.571$, or the framework added nothing, and the mechanism predicts *where* the gain comes from. On the long `csi300` window I expect the ensemble to lift IC/rank IC modestly above 0.0457/0.0569 and to hold or improve the already-shallow $-0.0579$ drawdown — the SR down-weighting of noise and FS diversity should make the portfolio at least as steady. On `csi300_recent` I expect a hold-or-improve over LGBM's annualized return 0.0807 — the FS tail of dormant factors is precisely the exposure a far regime rewards. On `csi300_shifted` it is my chance to close the gap to AdaRNN, pushing the tree's 1.30 info ratio toward or past 1.66 by reweighting toward the learnable samples the mild shift makes informative. The honest risk the numbers would expose is over-thinning: with only three short members each on a sampled feature subset, the ensemble could *lose* a little IC on the long window if the per-member subsets drop too much signal — if `csi300` IC comes in *below* 0.0457, that is the tell that the FS ratios are too aggressive for a 158-factor set at this sample size. But built from the exact tree that won, made deliberate only about samples and features, the expectation is a uniform-or-better lift across all three regimes with the drawdown held — an ensemble of the strongest control, turned drift-robust on the two axes the single tree left uniform.

```python
# EDITABLE region of custom_model.py (lines 16-103) — finale: DoubleEnsemble (DEnsembleModel)
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
# DoubleEnsemble wrapped around a LightGBM
# base learner, faithful to qlib/contrib/model/double_ensemble.py (DEnsembleModel).
import lightgbm as lgb
from typing import Text, Union
from qlib.model.interpret.base import FeatureInt
from qlib.log import get_module_logger

class CustomModel(Model, FeatureInt):
    """Double Ensemble Model"""

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
        self.base_model = base_model  # "gbm" or "mlp", specifically, we use lgbm for "gbm"
        self.num_models = num_models  # the number of sub-models
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        if sample_ratios is None:  # the default values for sample_ratios
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:  # the default values for sub_weights
            sub_weights = [1] * self.num_models
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.logger = get_module_logger("DEnsembleModel")
        self.logger.info("Double Ensemble Model...")
        self.ensemble = []  # the current ensemble model, a list contains all the sub-models
        self.sub_features = []  # the features for each sub model in the form of pandas.Index
        self.params = {"objective": loss}
        # Official Alpha158 CSI300 benchmark LightGBM base params (identical to the LGBM baseline),
        # so the finale isolates the ensembling framework, not the base learner.
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
        for _k, _v in _bench.items():
            kwargs.setdefault(_k, _v)
        self.params.update(kwargs)
        self.loss = loss
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        # initialize the sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        # train sub-models
        for k in range(self.num_models):
            self.sub_features.append(features)
            self.logger.info("Training sub-model: ({}/{})".format(k + 1, self.num_models))
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break

            self.logger.info("Retrieving loss curve and loss values...")
            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            pred_k = self.predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            )
            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if self.enable_sr:
                self.logger.info("Sample re-weighting...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                self.logger.info("Feature selection...")
                features = self.feature_selection(df_train, loss_values)

    def train_submodel(self, df_train, df_valid, weights, features):
        dtrain, dvalid = self._prepare_data_gbm(df_train, df_valid, weights, features)
        evals_result = dict()

        callbacks = [lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            self.logger.info("Training with early_stopping...")

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.epochs,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
        return model

    def _prepare_data_gbm(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        the SR module of Double Ensemble
        :param loss_curve: the shape is NxT
        the loss curve for the previous sub-model, where the element (i, t) if the error on the i-th sample
        after the t-th iteration in the training of the previous sub-model.
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :param k_th: the index of the current sub-model, starting from 1
        :return: weights
        the weights for all the samples.
        """
        # normalize loss_curve and loss_values with ranking
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        # calculate l_start and l_end from loss_curve
        N, T = loss_curve.shape
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # calculate h-value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # calculate weights
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights

    def feature_selection(self, df_train, loss_values):
        """
        the FS module of Double Ensemble
        :param df_train: the shape is NxF
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :return: res_feat: in the form of pandas.Index

        """
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # shuffle specific columns and calculate g-value for each feature
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                    )
                    / M
                )
            loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # one column in train features is all-nan # if g['g_value'].isna().any()
        g["g_value"].replace(np.nan, 0, inplace=True)

        # divide features into bins_fs bins
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # randomly sample features from bins to construct the new features
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        return pd.Index(set(res_feat))

    def get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        else:
            raise ValueError("not implemented yet")

    def retrieve_loss_curve(self, model, df_train, features):
        if self.base_model == "gbm":
            num_trees = model.num_trees()
            x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
            # Lightgbm need 1D array as its label
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train = np.squeeze(y_train.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            N = x_train.shape[0]
            loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
            pred_tree = np.zeros(N, dtype=float)
            for i_tree in range(num_trees):
                pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
                loss_curve.iloc[:, i_tree] = self.get_loss(y_train, pred_tree)
        else:
            raise ValueError("not implemented yet")
        return loss_curve

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.ensemble is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )
        pred = pred / np.sum(self.sub_weights)
        return pred

    def predict_sub(self, submodel, df_data, features):
        x_data = df_data["feature"].loc[:, features]
        pred_sub = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            res.append(pd.Series(_model.feature_importance(*args, **kwargs), index=_model.feature_name()) * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)
```

No workflow YAML edit: the finale uses the default `DatasetH` + Alpha158 handler (full 158 raw
factors), the same data view as the winning LightGBM baseline.
