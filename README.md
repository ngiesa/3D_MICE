# 3D-MICE for Python: Multivariate Imputation with Time Causal Structure

This repository provides a **Python implementation of the 3D-MICE algorithm** —  
a **time-aware multivariate imputation** framework that extends traditional MICE (Multiple Imputation by Chained Equations) into three dimensions:
- **Features (F)**
- **Time (T)**
- **Subjects (N)**

The algorithm performs imputation **causally** over time, ensuring that values at time `t` are only influenced by observations from **previous or equal time points** for each subject.  
This implementation also supports **variable-length time series**, where subjects can have different numbers of observations.
Also, there has not been a Python implementation yet but rather multiple implementations in R. 
---

## Reference

The 3D-MICE method was introduced in:

> **Yuan Luo, Peter Szolovits, Anand S Dighe, Jason M Baron**  
> *3D-MICE: integration of cross-sectional and longitudinal imputation for multi-analyte longitudinal clinical data*  
> *J Am Med Inform Assoc. 2017 Nov 30;25(6):645–653. doi: 10.1093/jamia/ocx133*
> (https://pmc.ncbi.nlm.nih.gov/articles/PMC7646951/)

---

## Model Fundamentals

### Background

Traditional **MICE** imputes missing values feature-by-feature, using regression models iteratively until convergence.  
However, in longitudinal or time-dependent data (e.g., patient measurements over time), simple MICE ignores **temporal dependencies** — leading to unrealistic imputations.

**3D-MICE** extends MICE by:

1. **Adding a temporal dimension** — imputation at time `t` considers all previous time steps (`≤ t`).
2. **Maintaining causal directionality** — no future information is used for imputing the present.
3. **Optionally incorporating temporal priors** via **Gaussian Processes (GPs)** to smooth predictions along time.

---

### Implementation

This version enhances the standard 3D-MICE algorithm with several modern features:

| Feature | Description |
|----------|--------------|
| **Variable-length support** | Each subject can have a different number of time steps (`T_i` varies). |
| **Causal Imputation** | Only previous observations of the same subject are used. |
| **GP Temporal Regularization** | Optionally fits Gaussian Processes per feature to enforce temporal smoothness. |
| **Train–Test separation** | Training data is used for hyperparameter optimization; test data is imputed using fitted models only. |
| **Full compatibility with `numpy` and `scikit-learn`** | Uses standard estimators like `IterativeImputer` and `BayesianRidge`. |

---

## Data Format

Each subject’s time series is represented as a 2D array:
X_i.shape == (T_i, F)


where:
- **T_i** = number of time steps (can differ across subjects)
- **F** = number of features (consistent across all subjects)

Missing entries should be `np.nan`.

### Example

| Time | HR | BP | Temp |
|------|----|----|------|
| t₀ | 72 | 110 | 36.7 |
| t₁ | NaN | 108 | 36.8 |
| t₂ | 75 | NaN | NaN |


### Code Example

```python


import numpy as np
from three_d_mice_varlen_fixed import ThreeDMICE

# Example training and test data (subjects with variable lengths)
X_train = [
    np.array([[1.0, np.nan, 3.0],
              [np.nan, 2.5, 3.2],
              [4.0, np.nan, np.nan]]),
    np.array([[np.nan, 1.2, 0.5],
              [0.9, np.nan, 0.7]])
]

X_test = [
    np.array([[1.1, np.nan, 3.1],
              [np.nan, np.nan, 3.4]]),
    np.array([[0.8, 1.0, np.nan]])
]

# Initialize the model
mice3d = ThreeDMICE(
    n_iter=3,          # outer imputation cycles
    use_gp=True,       # enable Gaussian Process smoothing
    mice_max_iter=10,  # MICE iterations per time step
    random_state=42
)

# Fit on training data
mice3d.fit(X_train)

# Impute test data causally
X_test_imputed = mice3d.transform(X_test)

for subj_i, arr in enumerate(X_test_imputed):
    print(f"Subject {subj_i} (shape {arr.shape}):")
    print(np.round(arr, 3))
