import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from M2_Shapoval import table
from M3_Thinning_NN import (VALIDATION_FRACTION, K_train, ThinnedDataset, frames_2d, block_variance, thin, load_model)


def block_counts(L, max_nb=32):
    """
    List the numbers of blocks per side that divide L.

    Parameters:
        L (int): system size.
        max_nb (int): largest number of blocks considered.

    Returns:
        list: divisors of L from 2 to max_nb.
    """
    return [nb for nb in range(2, max_nb + 1) if L % nb == 0]


def random_level(n, L, nb):
    """
    Compute the random level E_rand, the mean block variance of n grains scattered at random (equation
    eq:hyper).

    Parameters:
        n (ndarray): number of grains.
        L (int): system size.
        nb (int): number of blocks per side.

    Returns:
        ndarray: the random level for every n.
    """
    b = L // nb
    q = np.asarray(n, dtype=np.float64) / (L * L)
    return q * (1 - q) * (L * L - b * b) / (b * b * (L * L - 1))


def excess_by_scale(V, y, K, L, n_boot=2000, seed=3):
    """
    Compute the excess Delta of the block variance over the random level and its shape (equation eq:delta).

    Parameters:
        V (dict): {nb: block variance of the thinned examples}.
        y (ndarray): labels.
        K (int): thinning level.
        L (int): system size.
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the bootstrap.

    Returns:
        dict: for every nb ratio_neg and ratio_pos (class means in units of the random level), excess, shape
        (excess divided by its maximum) and ci_shape.
    """
    nbs = sorted(V)
    M = np.column_stack([np.asarray(V[nb], dtype=np.float64) for nb in nbs])
    pos = np.asarray(y).astype(bool)
    E = np.array([random_level(np.array([K]), L, nb)[0] for nb in nbs])

    def means(w):
        wp, wn = w * pos, w * ~pos
        return wp @ M / wp.sum(), wn @ M / wn.sum()

    Vp, Vn = means(np.ones(len(pos)))
    exc = (Vp - Vn) / E
    rng = np.random.default_rng(seed)
    B = np.empty((n_boot, len(nbs)))
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos)).astype(np.float64)
        vp, vn = means(w)
        B[i] = (vp - vn) / E
    ci_shape = np.percentile(B / B.max(axis=1, keepdims=True), [2.5, 97.5], axis=0)
    return {nb: dict(ratio_neg=float(Vn[j] / E[j]), ratio_pos=float(Vp[j] / E[j]),
                     excess=float(exc[j]), shape=float(exc[j] / exc.max()),
                     ci_shape=[float(ci_shape[0, j]), float(ci_shape[1, j])])
            for j, nb in enumerate(nbs)}


def scan_block_counts(L):
    """
    List the block counts used in the scans, with a block side of at least 2 sites.

    Parameters:
        L (int): system size.

    Returns:
        list: numbers of blocks per side.
    """
    return [nb for nb in block_counts(L) if L // nb >= 2]


def excess(L, scope="test_pool", n_boot=2000, seed=0):
    """
    Compute the excess block variance across scales with bootstrap intervals (Figure 2a).

    Parameters:
        L (int): system size.
        scope (str): "train_pool" or "test_pool".
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the thinning.

    Returns:
        dict: K, scope and points, the output of excess_by_scale for every scale l/L.
    """
    K = K_train(L)
    X, y, n, _ = frames_2d(scope, L, K)
    Xp = thin(X, K, np.random.default_rng(seed))
    assert np.all(Xp.sum((1, 2)) == K), "fixed number of grains violated"
    del X
    nbs = scan_block_counts(L)
    d = excess_by_scale({nb: block_variance(Xp, nb) for nb in nbs}, y, K, L, n_boot=n_boot)
    return {"K": K, "scope": scope, "points": {f"{(L // nb) / L:.4f}": dict(l=L // nb, **d[nb]) for nb in nbs}}


def excess_fixed_n(L, scope="test_pool", n_boot=2000, seed=0):
    """
    Compute the shape of the excess within groups of identical original number of grains (Appendix A).

    Parameters:
        L (int): system size.
        scope (str): "train_pool" or "test_pool".
        n_boot (int): number of bootstrap resamples.
        seed (int): seed of the thinning.

    Returns:
        dict: K, scope and points with excess, shape and ci_shape for every scale l/L.
    """
    K = K_train(L)
    X, y, n, _ = frames_2d(scope, L, K)
    Xp = thin(X, K, np.random.default_rng(seed))
    del X
    nbs = scan_block_counts(L)
    M = np.column_stack([block_variance(Xp, nb).astype(np.float64) for nb in nbs])
    E = np.array([random_level(np.array([K]), L, nb)[0] for nb in nbs])
    del Xp
    pos = y.astype(bool)
    _, gi = np.unique(n, return_inverse=True)
    G = gi.max() + 1

    def within_groups(w):
        wp, wn = w * pos, w * ~pos
        P, Q = np.bincount(gi, wp, G), np.bincount(gi, wn, G)
        ok = (P > 0) & (Q > 0)
        wg = P[ok] * Q[ok] / (P[ok] + Q[ok])
        d = np.array([(wg * (np.bincount(gi, wp * M[:, j], G)[ok] / P[ok]
                             - np.bincount(gi, wn * M[:, j], G)[ok] / Q[ok])).sum() / wg.sum()
                      for j in range(len(nbs))])
        return d / E

    exc = within_groups(np.ones(len(y)))
    rng = np.random.default_rng(seed + 3)
    B = np.array([within_groups(np.bincount(rng.integers(0, len(y), len(y)),
                                            minlength=len(y)).astype(np.float64))
                  for _ in range(n_boot)])
    ci_shape = np.percentile(B / B.max(axis=1, keepdims=True), [2.5, 97.5], axis=0)
    return {"K": K, "scope": scope, "points": {
        f"{(L // nb) / L:.4f}": dict(l=L // nb, excess=float(exc[j]), shape=float(exc[j] / exc.max()),
                                     ci_shape=[float(ci_shape[0, j]), float(ci_shape[1, j])])
        for j, nb in enumerate(nbs)}}


def shuffle_within_blocks(X, b, rng):
    """
    Rearrange the grains at random inside every block of side b (Section 2.4).

    Parameters:
        X (ndarray): frames, shape (N, L, L).
        b (int): block side.
        rng (np.random.Generator): random generator.

    Returns:
        ndarray: frames with unchanged block densities.
    """
    N, L, _ = X.shape
    nb = L // b
    v = np.asarray(X).reshape(N, nb, b, nb, b).transpose(0, 1, 3, 2, 4).reshape(N, nb * nb, b * b).copy()
    perm = np.argsort(rng.random((N, nb * nb, b * b)), axis=-1)
    v = np.take_along_axis(v, perm, axis=-1)
    return v.reshape(N, nb, nb, b, b).transpose(0, 1, 3, 2, 4).reshape(N, L, L)


def destruction_scan(L, scope="validation", seeds=(0, 1, 2, 3, 4), seed=0):
    """
    Compute the lost fraction F of the network's advantage after structure below a scale is destroyed (equation
    eq:lost).

    Parameters:
        L (int): system size.
        scope (str): "validation" for the validation part of every seed, or "test_pool".
        seeds (tuple): training seeds.
        seed (int): seed of the thinning and of the rearrangement.

    Returns:
        dict: K, scope, seeds and points with lost_fraction for every scale l/L.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(m, Xarr, y, K):
        """
        Compute the probabilities of a model on frames already thinned to K.

        Parameters:
            m (nn.Module): network.
            Xarr (ndarray): frames, shape (N, L, L).
            y (ndarray): labels.
            K (int): thinning level.

        Returns:
            ndarray: one probability per example.
        """
        Z = ThinnedDataset(Xarr[:, None], y, K, fixed=True, seed=7)
        pr = []
        with torch.no_grad():
            for xb, _ in DataLoader(Z, batch_size=128):
                pr.append(torch.sigmoid(m(xb.to(dev))).cpu().numpy())
        return np.concatenate(pr).ravel()

    K = K_train(L)
    X, y, n, _ = frames_2d("train_pool" if scope == "validation" else scope, L, K)
    Xp = thin(X, K, np.random.default_rng(seed))
    del X
    va = ({s: train_test_split(np.arange(len(y)), test_size=VALIDATION_FRACTION, stratify=y, random_state=s)[1] for s in seeds}
          if scope == "validation" else {s: np.arange(len(y)) for s in seeds})
    models = [load_model(L, "thinned", s, dev) for s in seeds]

    def mean_auc(Xarr):
        P = [predict(m, Xarr, y, K) for m in models]
        return float(np.mean([roc_auc_score(y[va[s]], P[i][va[s]]) for i, s in enumerate(seeds)]))

    a0 = mean_auc(Xp)
    result = {"K": K, "scope": scope, "seeds": list(seeds), "points": {}}
    rng = np.random.default_rng(seed + 1)
    for nb in scan_block_counts(L):
        b = L // nb
        a = mean_auc(shuffle_within_blocks(Xp, b, rng))
        result["points"][f"{b / L:.4f}"] = dict(lost_fraction=float((a0 - a) / (a0 - 0.5)))
    del models
    torch.cuda.empty_cache()
    return result


def band(points, lo=0.25, hi=0.5):
    """
    Compute the fraction of the network's advantage in the band of scales (lo, hi].

    Parameters:
        points (dict): points of destruction_scan.
        lo (float): lower scale l/L.
        hi (float): upper scale l/L.

    Returns:
        float: increase of F between the two scales.
    """
    return points[f"{hi:.4f}"]["lost_fraction"] - points[f"{lo:.4f}"]["lost_fraction"]
