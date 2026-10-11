import numpy as np
import pandas as pd
import sklearn.metrics
from numba import njit
from scipy.stats import ks_2samp
from M1_Manna import D_AVALANCHE, S_STAR, phase_array

BETA_AVALANCHE = 1 + 3 / 11

@njit(cache=True)
def get_eps(L, s_ast, s_sizes, q_sizes):
    """
    Find the alarm threshold q* that minimizes eps = nu + tau on the training stream.

    Parameters:
        L (int): system size.
        s_ast (ndarray): extreme thresholds in units of L^2.75.
        s_sizes (ndarray): avalanche sizes of the stream.
        q_sizes (ndarray): densities of the stream.

    Returns:
        tuple: (q_stars, e_stars), the threshold and the error for every element of s_ast.
    """
    q_stars = np.zeros(len(s_ast), dtype=np.float64)
    e_stars = np.zeros(len(s_ast), dtype=np.float64)
    order = np.argsort(q_sizes[:-1])[::-1]
    q_roder = q_sizes[:-1][order]
    n = len(q_roder)
    for j in range(len(s_ast)):
        s_star = s_ast[j] * L ** 2.75
        X_sorted = (s_sizes[1:] > s_star)[order]
        TP = 0
        FP = 0
        num_p = X_sorted.sum()
        num_n = n - num_p
        best = 0
        best_eps = 2.0
        for i in range(n):
            if X_sorted[i] == 1:
                TP += 1
            else:
                FP += 1
            if i == n - 1 or q_roder[i + 1] != q_roder[i]:
                eps = (num_p - TP) / num_p + FP / num_n
                if eps < best_eps:
                    best_eps = eps
                    best = i
        q_stars[j] = q_roder[best]
        e_stars[j] = best_eps
    return q_stars, e_stars


def get_test_metrics(L, s_sizes, q_sizes, s_ast, q_ast, s_extreme=0.353, window=(0.05, 0.3)):
    """
    Evaluate density as a predictor on the test stream at thresholds chosen on the training stream.

    Parameters:
        L (int): system size.
        s_sizes (ndarray): avalanche sizes of the test stream.
        q_sizes (ndarray): densities of the test stream.
        s_ast (ndarray): extreme thresholds in units of L^2.75.
        q_ast (ndarray): alarm thresholds q*, one for every element of s_ast.
        s_extreme (float): extreme threshold used for the ROC curve.
        window (tuple): range of s_ast in which the slope is fitted.

    Returns:
        tuple: (fpr, tpr, auc, delta_1, e_test). fpr, tpr and auc describe the ROC of density at s_extreme,
        delta_1 is minus the slope of e_test against s_ast (Table 3), and e_test is the error for every
        element of s_ast.
    """
    e_test = np.zeros(len(s_ast), dtype=np.float64)
    for j in range(len(s_ast)):
        X_true = s_sizes[1:] > s_ast[j] * L ** 2.75
        X_pred = q_sizes[:-1] >= q_ast[j]
        num_p = X_true.sum()
        num_n = len(X_true) - num_p
        e_test[j] = ((X_true & ~X_pred).sum() / num_p
                     + (~X_true & X_pred).sum() / num_n)

    w = (window[0] < s_ast) & (s_ast < window[1])
    slope, _ = np.polyfit(s_ast[w], e_test[w], 1)

    S_star = int(s_extreme * L ** 2.75)
    X_true = s_sizes[1:] > S_star
    fpr, tpr, _ = sklearn.metrics.roc_curve(X_true, q_sizes[:-1])
    auc = sklearn.metrics.roc_auc_score(X_true, q_sizes[:-1])
    return fpr, tpr, float(auc), float(-slope), e_test


def avalanche_pdf(s_sizes, n_bins=50):
    """
    Compute the density of avalanche sizes in logarithmic bins.

    Parameters:
        s_sizes (ndarray): avalanche sizes.
        n_bins (int): number of bin edges.

    Returns:
        tuple: (centers, pdf) for the non-empty bins.
    """
    s = s_sizes[s_sizes > 0]
    bins = np.logspace(0, np.log10(s.max()), num=n_bins)
    counts, edges = np.histogram(s, bins=bins)
    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = counts / (len(s) * widths)
    mask = pdf > 0
    return centers[mask], pdf[mask]


def exponent(centers, pdf, L, window=(10, 0.1)):
    """
    Fit the power-law exponent alpha of the avalanche-size distribution.

    Parameters:
        centers (ndarray): bin centers.
        pdf (ndarray): density in the bins.
        L (int): system size.
        window (tuple): fitting range, window[0] < s < window[1] * L^2.75.

    Returns:
        float: the exponent alpha.
    """
    m = (centers > window[0]) & (centers < window[1] * L ** 2.75)
    return float(-np.polyfit(np.log10(centers[m]), np.log10(pdf[m]), 1)[0])


def epsilon_star(feature, y, n_bins=1000):
    """
    Find the smallest error eps = nu + tau of a threshold rule on the given data (Table 3).

    Parameters:
        feature (ndarray): predictor, with an alarm when it is at or above the threshold.
        y (ndarray): labels, true for states followed by an extreme.
        n_bins (int): number of bins whose edges serve as candidate thresholds.

    Returns:
        float: the smallest error.
    """
    feature = np.asarray(feature)
    y = np.asarray(y).astype(bool)
    P = int(y.sum())
    N = int((~y).sum())
    bins = np.linspace(float(feature.min()), float(feature.max()), n_bins + 1)
    h1, _ = np.histogram(feature[y], bins=bins)
    hall, _ = np.histogram(feature, bins=bins)
    hn = hall - h1
    prefP = np.concatenate(([0], np.cumsum(h1)))
    sufN = np.concatenate((np.cumsum(hn[::-1])[::-1], [0]))
    eps = prefP / P + sufN / N
    return float(eps.min())


def baseline(L, s_ast=None):
    """
    Compute a row of Table 3 and the data of Figure 1 for one system size.

    Parameters:
        L (int): system size.
        s_ast (ndarray): extreme thresholds in units of L^2.75, by default 0.02 to 0.40 with step 0.01 and
            S_STAR.

    Returns:
        dict: T (number of steps), auc, alpha, delta1, q_star, eps_qt, eps_threshold, roc (thinned to about
        400 points) and pdf.
    """
    s_ast = np.append(np.arange(0.02, 0.401, 0.01), S_STAR) if s_ast is None else np.asarray(s_ast)
    s_tr = np.array(phase_array("train_pool", L, "s_all"))
    q_tr = np.array(phase_array("train_pool", L, "q_all"))
    q_star, _ = get_eps(L, s_ast, s_tr, q_tr)
    del s_tr, q_tr

    s_te = np.array(phase_array("test_pool", L, "s_all"))
    q_te = np.array(phase_array("test_pool", L, "q_all"))
    fpr, tpr, auc, delta_1, e_test = get_test_metrics(L, s_te, q_te, s_ast, q_star, S_STAR)
    i = int(np.flatnonzero(s_ast == S_STAR)[0])
    centers, pdf = avalanche_pdf(s_te)
    eps_qt = epsilon_star(q_te[:-1], s_te[1:] > int(S_STAR * L ** D_AVALANCHE))
    step = max(1, len(fpr) // 400)
    return dict(T=len(q_te), auc=auc, alpha=exponent(centers, pdf, L), delta1=delta_1,
                q_star=float(q_star[i]), eps_qt=eps_qt, eps_threshold=float(e_test[i]),
                roc=[fpr[::step].tolist(), tpr[::step].tolist()],
                pdf=[centers.tolist(), pdf.tolist()])


def distribution_collapse(curves, window=(10, 0.1)):
    """
    Measure how well the rescaled avalanche-size distributions overlap (Figure 1a).

    Parameters:
        curves (dict): {L: (centers, pdf)}.
        window (tuple): scaling range, s > window[0] in the smaller system and s < window[1] * L^D.

    Returns:
        dict: mean absolute difference of log10 of the rescaled densities for every pair of sizes.
    """
    D, beta = D_AVALANCHE, BETA_AVALANCHE
    log = {L: (np.log10(np.asarray(c) / L ** D), np.log10(np.asarray(p) * L ** (beta * D))) for L, (c, p) in curves.items()}
    sizes = sorted(log)
    result = {}
    for i, a in enumerate(sizes):
        for b in sizes[i + 1:]:
            x = np.linspace(np.log10(window[0] / a ** D), np.log10(window[1]), 50)
            result[f"{a}-{b}"] = float(np.abs(np.interp(x, *log[a]) - np.interp(x, *log[b])).mean())
    return result


SHAPOVAL_2022 = dict(auc=0.946, q_star=0.7098, eps=0.205, nu=0.076, tau=0.129, delta1=2.23, events=131, steps=5 * 10**7)


def comparison_with_shapoval(result, events):
    """
    Compare the baseline at L = 128 with the values of Shapoval et al. (2022).

    Parameters:
        result (dict): output of baseline(128).
        events (int): number of extremes in the test stream.

    Returns:
        dict: our value and the published one for AUC, q*, eps, delta_1 and the rate of extremes, with the
        difference in standard errors of the published sample for AUC and eps.
    """
    sh = SHAPOVAL_2022
    se_eps = np.sqrt(sh["nu"] * (1 - sh["nu"]) / sh["events"])
    se_auc = sh["auc"] * np.sqrt((1 - sh["auc"]) / ((1 + sh["auc"]) * sh["events"]))
    return {
        "AUC": dict(ours=result["auc"], shapoval=sh["auc"], difference_in_standard_errors=float(abs(result["auc"] - sh["auc"]) / se_auc)),
        "q*": dict(ours=result["q_star"], shapoval=sh["q_star"]),
        "eps": dict(ours=result["eps_threshold"], shapoval=sh["eps"],
                    difference_in_standard_errors=float(abs(result["eps_threshold"] - sh["eps"]) / se_eps)),
        "delta_1": dict(ours=result["delta1"], shapoval=sh["delta1"]),
        "extremes per 10^7 steps": dict(ours=events / result["T"] * 1e7, shapoval=sh["events"] / sh["steps"] * 1e7),
    }


def collected_examples(L):
    """
    Count the collected examples in both phases (Table 1).

    Parameters:
        L (int): system size.

    Returns:
        dict: for each phase n, pos, neg and the mean density of the last frame in each class (q_pos,
        q_neg).
    """
    w = {}
    for scope in ("train_pool", "test_pool"):
        lab = np.array(phase_array(scope, L, "labels")).astype(int)
        idx = np.array(phase_array(scope, L, "indices"))
        q = np.asarray(phase_array(scope, L, "q_all")[idx - 1], dtype=np.float64)
        w[scope] = dict(n=len(lab), pos=int(lab.sum()), neg=int((lab == 0).sum()),
                        q_pos=float(q[lab == 1].mean()), q_neg=float(q[lab == 0].mean()))
    return w


def collector_vs_stream(L):
    """
    Check the collected examples against the full stream (Section 2.2).

    Parameters:
        L (int): system size.

    Returns:
        dict: for each phase extremes_in_stream (number of extreme avalanches in the stream),
        labels_consistent (positives are exactly the states followed by an extreme), negatives_spacing
        (smallest and largest gap between negatives) and ks_D, ks_p (Kolmogorov-Smirnov test of the density
        of negatives against the stream).
    """
    threshold = int(S_STAR * L ** D_AVALANCHE)
    result = {}
    for scope in ("train_pool", "test_pool"):
        s = phase_array(scope, L, "s_all")
        q = phase_array(scope, L, "q_all")
        idx = np.array(phase_array(scope, L, "indices"))
        lab = np.array(phase_array(scope, L, "labels")).astype(int)
        extremes = sum(int((np.asarray(s[i:i + 50_000_000]) > threshold).sum()) for i in range(0, len(s), 50_000_000))
        order = np.argsort(idx)
        s_after = np.empty(len(idx), dtype=np.int64)
        s_after[order] = np.asarray(s[idx[order]])
        neg = np.sort(idx[lab == 0])
        gaps = np.diff(neg)
        q_neg = np.asarray(q[neg - 1], dtype=np.float64)
        q_stream = np.asarray(q[np.linspace(0, len(q) - 2, 200_000).astype(np.int64)], dtype=np.float64)
        ks = ks_2samp(q_neg, q_stream)
        result[scope] = dict(extremes_in_stream=extremes, labels_consistent=bool(np.all((s_after > threshold) == (lab == 1))),
                             negatives_spacing=[int(gaps.min()), int(gaps.max())],
                             ks_D=float(ks.statistic), ks_p=float(ks.pvalue))
    return result


def example_independence(scope, L, lags=(1, 2, 4, 8)):
    """
    Compute two measures of independence of the examples (Section 2.7).

    Parameters:
        scope (str): "train_pool" or "test_pool".
        L (int): system size.
        lags (tuple): lags of the autocorrelation in units of the sampling interval.

    Returns:
        dict: cv_extreme_gaps (coefficient of variation of the gaps between extremes) and
        autocorrelation_q_negatives (autocorrelation of the density of consecutive negatives at every lag).
    """
    lab = np.array(phase_array(scope, L, "labels")).astype(int)
    idx = np.array(phase_array(scope, L, "indices"))
    q_all = phase_array(scope, L, "q_all")
    d = np.diff(np.sort(idx[lab == 1])).astype(np.float64)
    qn = np.asarray(q_all[np.sort(idx[lab == 0]) - 1], dtype=np.float64)
    return dict(cv_extreme_gaps=float(d.std() / d.mean()),
                autocorrelation_q_negatives=[float(np.corrcoef(qn[:-k], qn[k:])[0, 1]) for k in lags])


def table(results, skip=()):
    """
    Show a dictionary of results as a table.

    Parameters:
        results (dict): one row per key, with nested fields joined by a dot.
        skip (tuple): fields left out.

    Returns:
        pd.DataFrame: the table, with numbers in lists rounded to four significant digits.
    """
    def flatten(d, prefix=""):
        row = {}
        for k, v in d.items():
            if k in skip:
                continue
            if isinstance(v, dict):
                row.update(flatten(v, f"{prefix}{k}."))
            elif isinstance(v, (list, tuple)):
                row[f"{prefix}{k}"] = [float(f"{x:.4g}") if isinstance(x, float) else x for x in v]
            else:
                row[f"{prefix}{k}"] = v
        return row

    return pd.DataFrame.from_dict({k: flatten(v) for k, v in results.items()}, orient="index")
