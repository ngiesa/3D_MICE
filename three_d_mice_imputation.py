
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ThreeDMICE:
    """
    3D-MICE hybrid imputation for variable-length time series.

    Each subject has its own (T_i, F) array:
        - T_i = time steps (rows)
        - F   = features (columns)
    Missing values: np.nan

    Example:
    mice = ThreeDMICE(n_iter=3, use_gp=True)
    mice.fit(X_train)          # X_train = list of (T_i, F)
    X_test_imp = mice.transform(X_test)
    """

    def __init__(self,
                use_gp=True,
                n_iter=10,
                mice_max_iter=10,
                n_imputations=1,
                gp_kernel=None,
                random_state=None):
        self.use_gp = use_gp
        self.n_iter = n_iter
        self.mice_max_iter = mice_max_iter
        self.n_imputations = n_imputations
        self.random_state = random_state
        self.gp_kernel = gp_kernel or (C(1.0) * RBF(1.0))
        self.train_means = []
        self.trained_ = False
        self.gp_models_ = {}

    def _init_fill(self, X, means_list = None):
        """Forward fill missing values along time for one subject."""
        # storing feature training means
        df, subject_means = pd.DataFrame(X), []
        df = df.ffill()
        for ci, col in enumerate(df.columns):
            if not means_list:
                mean_val = df[col].mean(skipna=True)
                subject_means.append(mean_val)
            else:
                mean_val = means_list[ci]
                df[col] = df[col].fillna(mean_val)
        self.train_means.append(subject_means)
        return df.values

    def _fit_gp_subject(self, subj_idx, X_subj):
        """Fit a GaussianProcess per feature for one subject."""
        T_i, F = X_subj.shape
        times = np.arange(T_i).reshape(-1, 1)
        for feat in range(F):
            y = X_subj[:, feat]
            obs = ~np.isnan(y)
            if obs.sum() < 2:
                continue
            gp = GaussianProcessRegressor(kernel=self.gp_kernel,
                                        random_state=self.random_state)
            gp.fit(times[obs], y[obs])
            self.gp_models_[(subj_idx, feat)] = gp

    def _gp_predict(self, subj_idx, feat, times):
        """Predict using subject-specific GP."""
        key = (subj_idx, feat)
        if key not in self.gp_models_:
            return np.full(len(times), np.nan)
        return self.gp_models_[key].predict(times.reshape(-1, 1))

    def fit(self, X_train):
        """Fit on variable-length subjects."""
        self.gp_models_ = {}
        X_filled = [self._init_fill(x) for x in X_train]
        F = X_filled[0].shape[1]

        # fit GPs
        if self.use_gp:
            for i, x in enumerate(X_filled):
                self._fit_gp_subject(i, x)

        # iterative imputation cycles
        for cycle in range(self.n_iter):
            for subj_idx, X_subj in enumerate(X_filled):
                T_i, F = X_subj.shape
                for t in range(T_i):
                    # construct causal design (times ≤ t)
                    block = X_subj[:t + 1, :].reshape(1, (t + 1) * F)

                    if self.use_gp:
                        gp_pred = np.array([
                            self._gp_predict(subj_idx, f, np.array([t]))
                            for f in range(F)
                        ]).reshape(1, F)
                        design = np.hstack([block, gp_pred])
                    else:
                        design = block

                    full_cols = np.hstack([design, X_subj[t, :].reshape(1, -1)])

                    # create new imputer per t (to match correct column dimension)
                    imputer = IterativeImputer(
                        estimator=BayesianRidge(),
                        max_iter=self.mice_max_iter,
                        sample_posterior=self.n_imputations > 1,
                        random_state=self.random_state
                    )
                    imputed = imputer.fit_transform(full_cols)
                    X_subj[t, :] = imputed[:, -F:]
                X_filled[subj_idx] = X_subj

        self.trained_ = True
        self.X_train_imputed_ = X_filled
        return self

    def transform(self, X_test):
        """Impute test data causally (list of (T_i, F) arrays)."""
        if not self.trained_:
            raise RuntimeError("Model not fitted yet. Call .fit(X_train).")

        print(pd.DataFrame(self.train_means))

        means_list = list(pd.DataFrame(self.train_means).mean())

        X_out = []
        for subj_idx, X_subj in enumerate(X_test):
            
            X_subj = self._init_fill(np.asarray(X_subj, float), means_list=means_list)
            T_i, F = X_subj.shape

            for t in range(T_i):
                block = X_subj[:t + 1, :].reshape(1, (t + 1) * F)

                if self.use_gp:
                    gp_pred = np.array([
                        self._gp_predict(subj_idx % len(self.X_train_imputed_), f, np.array([t]))
                        for f in range(F)
                    ]).reshape(1, F)
                    design = np.hstack([block, gp_pred])
                else:
                    design = block

                full_cols = np.hstack([design, X_subj[t, :].reshape(1, -1)])
                imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=self.mice_max_iter,
                    random_state=self.random_state
                )
                imputed = imputer.fit_transform(full_cols)
                X_subj[t, :] = imputed[:, -F:]

            X_out.append(X_subj)
        return X_out