'''
Evaluation calibration diagnostics for MUSE measurement noise predictions.

Functions:
    - innovation_whiteness_test: ACF + KS normality + Ljung-Box on normalized innovations
    - nis_conditional_calibration: NIS grouped by constellation
    - gt_residual_vs_predicted: rolling empirical mean/var vs predicted mu_v/R
    - reliability_diagram: binned predicted R vs empirical variance

All functions return plain dicts/arrays suitable for JSON serialization or plotting.
No statsmodels dependency — uses scipy.stats and numpy only.
'''
import numpy as np
from scipy import stats


def normalized_innovations(err_uwb, S_star, mu_v):
    """Compute normalized innovations: e_k = (err_uwb_k - mu_v) / sqrt(S_star).

    Args:
        err_uwb: (N,) raw innovation (z - h(x))
        S_star: (N,) innovation covariance (scalar per step)
        mu_v: (N,) predicted measurement bias

    Returns:
        e_norm: (N,) normalized innovations
    """
    e_norm = (err_uwb - mu_v) / np.sqrt(np.maximum(S_star, 1e-12))
    return e_norm


def acf(x, max_lag=20):
    """Compute autocorrelation function up to max_lag.

    Args:
        x: (N,) time series (should be zero-mean or normalized)
        max_lag: maximum lag to compute

    Returns:
        lags: (max_lag+1,) array [0, 1, ..., max_lag]
        acf_vals: (max_lag+1,) autocorrelation at each lag (acf[0]=1.0)
    """
    N = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x_centered)
    if var < 1e-15:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)

    acf_vals = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k >= N:
            break
        acf_vals[k] = np.mean(x_centered[:N-k] * x_centered[k:]) / var

    return np.arange(max_lag + 1), acf_vals


def ljung_box(acf_vals, N, max_lag=20):
    """Compute Ljung-Box Q statistic.

    Q = N*(N+2) * sum_{k=1}^{L} r_k^2 / (N-k)

    Under H0 (white noise), Q ~ chi2(L).

    Args:
        acf_vals: (max_lag+1,) ACF values (acf_vals[0]=1, use [1:])
        N: number of observations
        max_lag: number of lags to include

    Returns:
        Q: Ljung-Box statistic
        p_value: p-value from chi2(max_lag) distribution
    """
    L = min(max_lag, len(acf_vals) - 1)
    r = acf_vals[1:L+1]
    ks = np.arange(1, L + 1)
    Q = N * (N + 2) * np.sum(r**2 / (N - ks))
    p_value = 1.0 - stats.chi2.cdf(Q, df=L)
    return float(Q), float(p_value)


def innovation_whiteness_test(err_uwb, S_star, mu_v, max_lag=20):
    """Full whiteness test on normalized innovations.

    Args:
        err_uwb: (N,) raw innovation
        S_star: (N,) innovation covariance
        mu_v: (N,) predicted measurement bias
        max_lag: max ACF lag

    Returns:
        dict with keys:
            e_norm: (N,) normalized innovations
            lags: (max_lag+1,)
            acf_vals: (max_lag+1,)
            confidence_bound: float (±1.96/sqrt(N))
            ks_statistic: float
            ks_pvalue: float
            ljung_box_Q: float
            ljung_box_pvalue: float
    """
    e_norm = normalized_innovations(err_uwb, S_star, mu_v)

    # Remove NaN/Inf
    mask = np.isfinite(e_norm)
    e_clean = e_norm[mask]
    N = len(e_clean)

    if N < 10:
        return {
            'e_norm': e_norm,
            'lags': np.arange(max_lag + 1),
            'acf_vals': np.zeros(max_lag + 1),
            'confidence_bound': 0.0,
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'ljung_box_Q': np.nan,
            'ljung_box_pvalue': np.nan,
        }

    lags, acf_vals = acf(e_clean, max_lag)
    Q, Q_pval = ljung_box(acf_vals, N, max_lag)
    ks_stat, ks_pval = stats.kstest(e_clean, 'norm')

    return {
        'e_norm': e_norm,
        'lags': lags,
        'acf_vals': acf_vals,
        'confidence_bound': 1.96 / np.sqrt(N),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'ljung_box_Q': float(Q),
        'ljung_box_pvalue': float(Q_pval),
    }


def nis_per_constellation(nis_values, ds_names, method_labels=None):
    """Group NIS values by constellation and compute mean NIS per group.

    Args:
        nis_values: dict {method_name: dict {ds_name: array of NIS values}}
        ds_names: list of dataset names
        method_labels: optional dict {method_name: display_label}

    Returns:
        dict with keys:
            constellations: list of constellation names
            methods: list of method names
            mean_nis: dict {method: dict {const: float}}
            chi2_bounds: (lower, upper) for 95% CI around 1.0 (dof=1)
    """
    # Group datasets by constellation
    const_map = {}
    for ds_name in ds_names:
        const = ds_name.split('-')[0]
        if const not in const_map:
            const_map[const] = []
        const_map[const].append(ds_name)

    constellations = sorted(const_map.keys())
    methods = sorted(nis_values.keys())

    mean_nis = {}
    for method in methods:
        mean_nis[method] = {}
        for const in constellations:
            all_nis = []
            for ds_name in const_map[const]:
                if ds_name in nis_values[method]:
                    vals = nis_values[method][ds_name]
                    if len(vals) > 0:
                        all_nis.extend(vals.tolist() if hasattr(vals, 'tolist') else list(vals))
            mean_nis[method][const] = float(np.mean(all_nis)) if all_nis else np.nan

    # Chi-square 95% bounds for NIS (dof=1 for scalar measurement)
    N_typical = 1000  # approximate sample size per group
    lower = stats.chi2.ppf(0.025, N_typical) / N_typical
    upper = stats.chi2.ppf(0.975, N_typical) / N_typical

    return {
        'constellations': constellations,
        'methods': methods,
        'mean_nis': mean_nis,
        'chi2_bounds': (float(lower), float(upper)),
    }


def gt_residual_vs_predicted(gt_residuals, mu_v_pred, R_pred, window=50):
    """Compare rolling empirical statistics of GT residuals vs predicted mu_v and R.

    Args:
        gt_residuals: (N,) ground truth residuals
        mu_v_pred: (N,) predicted measurement bias (per step)
        R_pred: (N,) predicted measurement variance (per step)
        window: rolling window size

    Returns:
        dict with keys:
            steps: (M,) step indices (center of each window)
            empirical_mean: (M,) rolling mean of GT residuals
            empirical_var: (M,) rolling variance of GT residuals
            predicted_mu_v: (M,) mu_v at window centers
            predicted_R: (M,) R at window centers
    """
    N = len(gt_residuals)
    if N < window:
        return {
            'steps': np.array([]),
            'empirical_mean': np.array([]),
            'empirical_var': np.array([]),
            'predicted_mu_v': np.array([]),
            'predicted_R': np.array([]),
        }

    M = N - window + 1
    steps = np.arange(window // 2, window // 2 + M)
    emp_mean = np.zeros(M)
    emp_var = np.zeros(M)

    for i in range(M):
        w = gt_residuals[i:i+window]
        emp_mean[i] = np.mean(w)
        emp_var[i] = np.var(w)

    return {
        'steps': steps,
        'empirical_mean': emp_mean,
        'empirical_var': emp_var,
        'predicted_mu_v': mu_v_pred[steps] if len(mu_v_pred) > steps[-1] else mu_v_pred[:M],
        'predicted_R': R_pred[steps] if len(R_pred) > steps[-1] else R_pred[:M],
    }


def reliability_diagram(R_pred, gt_residuals, mu_v_pred, n_bins=10):
    """Compute reliability diagram: binned predicted R vs empirical variance.

    Perfect calibration: predicted R = empirical variance of centered residuals.

    Args:
        R_pred: (N,) predicted measurement variance
        gt_residuals: (N,) ground truth residuals
        mu_v_pred: (N,) predicted measurement bias
        n_bins: number of quantile bins

    Returns:
        dict with keys:
            bin_centers: (n_bins,) mean predicted R per bin
            empirical_var: (n_bins,) mean (r - mu_v)^2 per bin
            bin_counts: (n_bins,) number of samples per bin
    """
    centered = (gt_residuals - mu_v_pred) ** 2

    # Quantile-based binning
    try:
        bin_edges = np.percentile(R_pred, np.linspace(0, 100, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([R_pred.min(), R_pred.max()])
    except Exception:
        return {
            'bin_centers': np.array([]),
            'empirical_var': np.array([]),
            'bin_counts': np.array([]),
        }

    actual_bins = len(bin_edges) - 1
    bin_centers = np.zeros(actual_bins)
    emp_var = np.zeros(actual_bins)
    bin_counts = np.zeros(actual_bins, dtype=int)

    for i in range(actual_bins):
        if i < actual_bins - 1:
            mask = (R_pred >= bin_edges[i]) & (R_pred < bin_edges[i+1])
        else:
            mask = (R_pred >= bin_edges[i]) & (R_pred <= bin_edges[i+1])

        if mask.sum() > 0:
            bin_centers[i] = np.mean(R_pred[mask])
            emp_var[i] = np.mean(centered[mask])
            bin_counts[i] = mask.sum()

    # Remove empty bins
    valid = bin_counts > 0
    return {
        'bin_centers': bin_centers[valid],
        'empirical_var': emp_var[valid],
        'bin_counts': bin_counts[valid],
    }
