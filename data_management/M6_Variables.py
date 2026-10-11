import warnings

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from M2_Shapoval import table
from M3_Thinning_NN import (VALIDATION_FRACTION, EVAL_SEED, K_train, ThinnedDataset, conditional_auc, last_frames,
                            trim_test, device, load_model, network_scores)
from M5_Modes_R2 import PAIR, energies, microstates, modes, r2_fast, r2_from_modes


def best_threshold(y, s):
    """
    Find the alarm threshold that minimizes eps = nu + tau.

    Parameters:
        y (ndarray): labels.
        s (ndarray): scores.

    Returns:
        tuple: (threshold, eps).
    """
    o = np.argsort(s)[::-1]
    ys = np.asarray(y)[o]
    npos, nneg = ys.sum(), len(ys) - ys.sum()
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    eps = (npos - tp) / npos + fp / nneg
    i = int(np.argmin(eps))
    return float(np.asarray(s)[o][i]), float(eps[i])


def at_threshold(y, s, threshold):
    """
    Compute the errors at a given threshold, with an alarm when s >= threshold.

    Parameters:
        y (ndarray): labels.
        s (ndarray): scores.
        threshold (float): alarm threshold.

    Returns:
        dict: missed, positives, false_alarms, negatives, nu, tau and eps.
    """
    y = np.asarray(y)
    alarm = np.asarray(s) >= threshold
    missed = int(((y == 1) & ~alarm).sum())
    false_alarms = int(((y == 0) & alarm).sum())
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    return dict(missed=missed, positives=npos, false_alarms=false_alarms, negatives=nneg,
                nu=missed / npos, tau=false_alarms / nneg, eps=missed / npos + false_alarms / nneg)


def _quantiles(v, k):
    """
    Assign every value to one of k quantile groups.

    Parameters:
        v (ndarray): values.
        k (int): number of groups.

    Returns:
        ndarray: group number, 0 to k - 1, of every value.
    """
    edges = np.quantile(v, np.linspace(0, 1, k + 1))
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, k - 1)


def _row(y_ex, s_ex, y_ho, s_ho):
    """
    Compute the AUC on the test set and eps at the threshold chosen on the training set.

    Parameters:
        y_ex (ndarray): training labels.
        s_ex (ndarray): training scores.
        y_ho (ndarray): test labels.
        s_ho (ndarray): test scores.

    Returns:
        dict: auc, threshold and eps.
    """
    threshold, _ = best_threshold(y_ex, s_ex)
    return dict(auc=float(roc_auc_score(y_ho, s_ho)), threshold=threshold, eps=at_threshold(y_ho, s_ho, threshold)["eps"])


def _ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def bootstrap_holdout(y, s_q, r2_, s_qr, n_boot=2000, seed=0):
    """
    Compute AUC(q), AUC(R2), AUC(q,R2) and the gain of the model over density with a paired bootstrap.

    Parameters:
        y (ndarray): labels.
        s_q (ndarray): density.
        r2_ (ndarray): R2.
        s_qr (ndarray): scores of the model (q, R2).
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the bootstrap.

    Returns:
        dict: auc_q, auc_R2, auc_qR2, delta_auc, ci_delta_auc and n_boot.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    point = (roc_auc_score(y, s_q), roc_auc_score(y, r2_), roc_auc_score(y, s_qr))
    d = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        yb = y[b]
        d[i] = roc_auc_score(yb, s_qr[b]) - roc_auc_score(yb, s_q[b])
    return dict(auc_q=float(point[0]), auc_R2=float(point[1]), auc_qR2=float(point[2]),
                delta_auc=float(point[2] - point[0]), ci_delta_auc=_ci(d), n_boot=n_boot)


def bootstrap_eps(y, scores, thresholds, n_boot=2000, seed=0):
    """
    Bootstrap eps of several predictors at frozen thresholds on the same resampled examples.

    Parameters:
        y (ndarray): labels.
        scores (list): scores of every predictor.
        thresholds (list): alarm threshold of every predictor.
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the bootstrap.

    Returns:
        ndarray: eps, shape (n_boot, number of predictors).
    """
    y = np.asarray(y).astype(bool)
    alarm = [np.asarray(s) >= p for s, p in zip(scores, thresholds)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    E = np.empty((n_boot, len(alarm)))
    for i in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        yb = y[b]
        P, N = yb.sum(), (~yb).sum()
        for j, a in enumerate(alarm):
            ab = a[b]
            E[i, j] = (yb & ~ab).sum() / P + (~yb & ab).sum() / N
    return E


def r2(R):
    """
    Compute R2 = R_20 + R_02 from the mode energies (equation eq:R2).

    Parameters:
        R (ndarray): mode energies, shape (N, L, L).

    Returns:
        ndarray: R2 of every state.
    """
    (m1, n1), (m2, n2) = PAIR
    return R[:, m1, n1] + R[:, m2, n2]


class ModelQR2:
    """
    Logistic model of standardized q and R2 (equation eq:logit), fitted without a penalty.

    Parameters:
        q (ndarray): densities of the training set.
        r2_ (ndarray): R2 of the training set.
        y (ndarray): training labels.
    """

    def __init__(self, q, r2_, y):
        X = np.column_stack([q, r2_])
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*penalty.*")
            self.lr = LogisticRegression(C=np.inf, max_iter=10_000).fit(self._z(q, r2_), y)

    def _z(self, q, r2_):
        return (np.column_stack([q, r2_]) - self.mean) / self.std

    def score(self, q, r2_):
        """
        Compute the logit of the model.

        Parameters:
            q (ndarray): densities.
            r2_ (ndarray): R2.

        Returns:
            ndarray: one score per state.
        """
        return self.lr.decision_function(self._z(q, r2_))

    def coefficients(self):
        return dict(b0=float(self.lr.intercept_[0]), bq=float(self.lr.coef_[0, 0]),
                    bR=float(self.lr.coef_[0, 1]),
                    mean=[float(v) for v in self.mean],
                    std=[float(v) for v in self.std])


def two_variable_model(L, n_boot=2000, seed=0):
    """
    Fit the model (q, R2) on the training set and evaluate it on the test set (Section 3.6, Table 6, Figure 4).

    Parameters:
        L (int): system size.
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the bootstrap.

    Returns:
        dict: model (coefficients and standardization), auc (output of bootstrap_holdout) and eps (threshold
        and error of q and of the model, and their difference delta_eps).
    """
    H, q_ex, y_ex = microstates("train_pool", L)
    r2_ex = r2(energies(modes(H), q_ex))
    del H
    model = ModelQR2(q_ex, r2_ex, y_ex)
    s_ex_qr = model.score(q_ex, r2_ex)

    H, q_ho, y_ho = microstates("test_pool", L)
    r2_ho = r2_from_modes(modes(H), q_ho)
    del H
    s_ho_qr = model.score(q_ho, r2_ho)

    eps = {name: {kk: v for kk, v in _row(y_ex, s_ex, y_ho, s_ho).items() if kk != "auc"}
           for name, s_ex, s_ho in (("q", q_ex, q_ho), ("qR2", s_ex_qr, s_ho_qr))}
    eps["delta_eps"] = eps["qR2"]["eps"] - eps["q"]["eps"]
    return dict(model=model.coefficients(),
                auc=bootstrap_holdout(y_ho, q_ho, r2_ho, s_ho_qr, n_boot, seed), eps=eps)


def raw_scores(L, X, y, seeds, dev=None):
    """
    Compute the scores of every network trained on raw images.

    Parameters:
        L (int): system size.
        X (ndarray): frames, shape (N, 1, L, L).
        y (ndarray): labels.
        seeds (tuple): training seeds.
        dev (str): device, chosen automatically when None.

    Returns:
        ndarray: scores, shape (seeds, examples).
    """
    dev = dev or device()
    P = []
    for s in seeds:
        m = load_model(L, "raw", s, dev)
        P.append(network_scores(m, X, y, None, False, 0, raw=True, batch=256))
        del m
        torch.cuda.empty_cache()
    return np.array(P)


def qr2_vs_network(L, seeds=(0, 1, 2, 3, 4), n_boot=2000, seed=0):
    """
    Compare the model (q, R2) and density with the network on raw images (Section 3.6, Table 6, Appendix B).

    Parameters:
        L (int): system size.
        seeds (tuple): training seeds.
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the bootstrap.

    Returns:
        dict: seeds, n_boot, eps_network (mean over seeds) with its spread, eps_network_train and
        eps_network_validation (error on the two parts of the training set), ci_qR2_minus_network, and
        fraction_qR2 (part of the network's error reduction recovered by the model) with its interval.
    """
    dev = device()
    X_ex, y_ex, n_ex = last_frames("train_pool", L)
    X_ho, y_ho, n_ho = last_frames("test_pool", L)
    q_ex, q_ho = n_ex / (L * L), n_ho / (L * L)
    H_ex, H_ho = X_ex[:, 0].astype(np.float32), X_ho[:, 0].astype(np.float32)
    r2_ex, r2_ho = r2_fast(H_ex, q_ex), r2_fast(H_ho, q_ho)
    model = ModelQR2(q_ex, r2_ex, y_ex)
    qr_ex, qr_ho = model.score(q_ex, r2_ex), model.score(q_ho, r2_ho)
    del H_ex, H_ho

    P_ex = raw_scores(L, X_ex, y_ex, seeds, dev)
    P_ho = raw_scores(L, X_ho, y_ho, seeds, dev)
    thresholds = [best_threshold(y_ex, p)[0] for p in P_ex]
    eps_s = np.array([at_threshold(y_ho, p, pr)["eps"] for p, pr in zip(P_ho, thresholds)])
    parts = {"train": [], "validation": []}
    for p, s in zip(P_ex, seeds):
        split = train_test_split(np.arange(len(y_ex)), test_size=VALIDATION_FRACTION, stratify=y_ex, random_state=s)
        for name, idx in zip(parts, split):
            parts[name].append(best_threshold(y_ex[idx], p[idx])[1])

    fixed = [(q_ho, best_threshold(y_ex, q_ex)[0]), (qr_ho, best_threshold(y_ex, qr_ex)[0])]
    E = bootstrap_eps(y_ho, [s for s, _ in fixed] + list(P_ho), [p for _, p in fixed] + thresholds, n_boot, seed)
    e_q, e_qr, e_n = E[:, 0], E[:, 1], E[:, 2:].mean(axis=1)
    p_q, p_qr = (at_threshold(y_ho, s, p)["eps"] for s, p in fixed)
    p_n = float(eps_s.mean())
    return dict(seeds=list(seeds), n_boot=n_boot,
                eps_network=p_n, eps_network_spread=float(eps_s.std()),
                eps_network_train=float(np.mean(parts["train"])), eps_network_validation=float(np.mean(parts["validation"])),
                ci_qR2_minus_network=_ci(e_qr - e_n),
                fraction_qR2=(p_q - p_qr) / (p_q - p_n), ci_fraction_qR2=_ci((e_q - e_qr) / (e_q - e_n)))


def network_vs_r2(L, seeds=(0, 1, 2, 3, 4), quantiles=20):
    """
    Compare the network with R2 on thinned test images (Section 3.6).

    Parameters:
        L (int): system size.
        seeds (tuple): training seeds.
        quantiles (int): number of groups of similar R2 or network score.

    Returns:
        dict: auc_R2, spearman (rank correlation of the network score with R2), auc_network_fixed_R2_k and
        auc_R2_fixed_network_k (AUC within the groups), averaged over seeds.
    """
    K = K_train(L)
    X, y, n = last_frames("test_pool", L)
    X, y, n, _ = trim_test(K, X, y, n)
    Z = ThinnedDataset(X, y, K, shuffled=False, fixed=True, seed=EVAL_SEED)
    inputs = [xb for xb, _ in DataLoader(Z, batch_size=256, shuffle=False)]
    qp = np.full(1, K / (L * L))
    r2_ = np.concatenate([r2_from_modes(modes(xb.numpy()[:, 0].astype(np.float64)),
                                        np.repeat(qp, len(xb))) for xb in inputs])
    dev = device()
    per = []
    for s in seeds:
        m = load_model(L, "thinned", s, dev)
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(m(xb.to(dev))).cpu().numpy().ravel() for xb in inputs])
        del m
        per.append({"spearman": float(spearmanr(p, r2_).correlation),
                    f"auc_network_fixed_R2_{quantiles}": float(conditional_auc(y, p, _quantiles(r2_, quantiles))[0]),
                    f"auc_R2_fixed_network_{quantiles}": float(conditional_auc(y, r2_, _quantiles(p, quantiles))[0])})
    return dict(auc_R2=float(roc_auc_score(y, r2_)),
                **{kk: float(np.mean([d[kk] for d in per])) for kk in per[0]})
