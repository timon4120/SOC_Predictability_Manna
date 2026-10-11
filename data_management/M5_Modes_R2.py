import numpy as np
from scipy.fft import dctn
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from M2_Shapoval import table
from M3_Thinning_NN import conditional_auc, block_variance, last_frames
from M4_Scale import shuffle_within_blocks


PAIR = ((2, 0), (0, 2))


def microstates(scope, L):
    """
    Load the last frames as binary fields with the density computed from the frame.

    Parameters:
        scope (str): "train_pool" or "test_pool".
        L (int): system size.

    Returns:
        tuple: (H, q, y) with fields of shape (N, L, L), densities and labels.
    """
    X, y, n = last_frames(scope, L)
    H = X[:, 0].astype(np.float64)
    del X
    values = np.unique(H)
    assert np.isin(values, (0.0, 1.0)).all(), f"a stable microstate is not binary: {values}"
    q = H.mean(axis=(1, 2))
    assert np.array_equal(np.rint(q * L * L).astype(np.int64), n), \
        "the density from the frame does not match the number of grains from the stream"
    return H, q, np.asarray(y).astype(int)


def modes(H):
    """
    Compute the coefficients a_mn in the orthonormal cosine basis (equations eq:dct and eq:basis).

    Parameters:
        H (ndarray): fields, shape (N, L, L).

    Returns:
        ndarray: coefficients, shape (N, L, L), with axis 1 for m and axis 2 for n.
    """
    return dctn(H, type=2, norm="ortho", axes=(1, 2))


def check_identities(A, q):
    """
    Check the two identities of Section 2.5: a_00 = L q and the sum of the other a_mn^2 = L^2 q (1 - q).

    Parameters:
        A (ndarray): mode coefficients, shape (N, L, L).
        q (ndarray): densities.

    Returns:
        dict: err_a00 and err_parseval, the largest absolute errors.
    """
    L = A.shape[1]
    b00 = float(np.abs(A[:, 0, 0] - L * q).max())
    energy = (A ** 2).sum(axis=(1, 2)) - A[:, 0, 0] ** 2
    bpar = float(np.abs(energy - L * L * q * (1 - q)).max())
    return dict(err_a00=b00, err_parseval=bpar)


def energies(A, q):
    """
    Compute the normalized mode energies R_mn (equation eq:Rmn), with the zero mode set to zero.

    Parameters:
        A (ndarray): mode coefficients, shape (N, L, L).
        q (ndarray): densities.

    Returns:
        ndarray: energies, shape (N, L, L).
    """
    L = A.shape[1]
    R = A ** 2
    R /= (L * L * q * (1 - q))[:, None, None]
    R[:, 0, 0] = 0.0
    return R


def auc_map(R, y, chunk=1024):
    """
    Compute the AUC of every mode energy.

    Parameters:
        R (ndarray): mode energies, shape (N, L, L).
        y (ndarray): labels.
        chunk (int): number of modes processed at once.

    Returns:
        ndarray: AUC of every mode, shape (L, L), with NaN at (0, 0).
    """
    N, L, _ = R.shape
    Rp = R.reshape(N, L * L)
    y = np.asarray(y).astype(bool)
    P, M = int(y.sum()), int((~y).sum())
    auc = np.empty(L * L)
    for j in range(0, L * L, chunk):
        r = rankdata(Rp[:, j:j + chunk], axis=0)
        auc[j:j + chunk] = (r[y].sum(axis=0) - P * (P + 1) / 2) / (P * M)
    auc = auc.reshape(L, L)
    auc[0, 0] = np.nan
    return auc


def rank_modes(auc, k=8):
    """
    List the k modes with the highest AUC.

    Parameters:
        auc (ndarray): AUC of every mode, shape (L, L).
        k (int): number of modes.

    Returns:
        list: dictionaries with m, n and auc, in descending order.
    """
    L = auc.shape[0]
    flat = np.nan_to_num(auc, nan=-np.inf).ravel()
    order = np.argsort(flat)[::-1]
    return [dict(m=int(i // L), n=int(i % L), auc=float(flat[i])) for i in order[:k]]


def mode_analysis(L, k=8):
    """
    Analyze the modes on the training set (Sections 2.5 and 3.4, Figure 3).

    Parameters:
        L (int): system size.
        k (int): number of strongest modes in the ranking.

    Returns:
        dict: identities (errors of the identities on both sets), auc_map, ranking and pair (AUC of the
        modes (2,0) and (0,2) and whether they are the two strongest).
    """
    H, q, y = microstates("train_pool", L)
    A = modes(H)
    del H
    ident_ex = check_identities(A, q)
    R = energies(A, q)
    del A
    sum_err = float(np.abs(R.sum(axis=(1, 2)) - 1).max())
    auc = auc_map(R, y)
    del R
    ranking = rank_modes(auc, k)
    pair = dict(auc_20=float(auc[2, 0]), auc_02=float(auc[0, 2]),
                in_top_two={(r["m"], r["n"]) for r in ranking[:2]} == set(PAIR))

    H, q, y = microstates("test_pool", L)
    ident_ho = check_identities(modes(H), q)
    return dict(identities=dict(train=ident_ex, test=ident_ho, err_sum_R=sum_err),
                auc_map=np.round(auc, 5), ranking=ranking, pair=pair)


def r2_from_modes(A, q):
    """
    Compute R2 from the mode coefficients.

    Parameters:
        A (ndarray): mode coefficients, shape (N, L, L).
        q (ndarray): densities.

    Returns:
        ndarray: R2 of every state.
    """
    L = A.shape[1]
    (m1, n1), (m2, n2) = PAIR
    return (A[:, m1, n1] ** 2 + A[:, m2, n2] ** 2) / (L * L * q * (1 - q))


def r2_fast(H, q):
    """
    Compute R2 directly from the fields, without the full cosine transform.

    Parameters:
        H (ndarray): fields, shape (N, L, L).
        q (ndarray): densities.

    Returns:
        ndarray: R2 of every state.
    """
    L = H.shape[-1]

    def basis(m):
        return (np.sqrt(1.0 / L) if m == 0 else np.sqrt(2.0 / L)) * np.cos(np.pi * m * (np.arange(L) + 0.5) / L)

    a = [np.einsum("nxy,x,y->n", H, basis(m), basis(n)) for m, n in PAIR]
    return (a[0] ** 2 + a[1] ** 2) / (L * L * q * (1 - q))


def contrast_direction(a_1, a_2, q, y, L):
    """
    Describe the direction of the contrast of the pair (2,0) and (0,2) (Appendix C).

    Parameters:
        a_1 (ndarray): coefficients of the first mode of the pair.
        a_2 (ndarray): coefficients of the second mode of the pair.
        q (ndarray): densities.
        y (ndarray): labels.
        L (int): system size.

    Returns:
        dict: for each class the fraction of states with a negative normalized amplitude (heavier center),
        and overload_ratio, the ratio of the class means of the signed amplitude.
    """
    y = np.asarray(y)
    norm = np.sqrt(L * L * q * (1 - q))
    s1, s2 = np.asarray(a_1) / norm, np.asarray(a_2) / norm
    result, mean = {}, {}
    for name, m in (("positives", y == 1), ("negatives", y == 0)):
        result[name] = dict(center_fraction_s20=float(np.mean(s1[m] < 0)),
                            center_fraction_s02=float(np.mean(s2[m] < 0)),
                            center_fraction_both=float(np.mean((s1[m] < 0) & (s2[m] < 0))))
        mean[name] = (float(s1[m].mean()) + float(s2[m].mean())) / 2
    result["overload_ratio"] = mean["positives"] / mean["negatives"]
    return result


def check_on_test(L):
    """
    Compare R2 with density on the test set (Section 3.5, Table 5, Appendix C).

    Parameters:
        L (int): system size.

    Returns:
        dict: rank correlations of R2 with q, auc_R2, auc_R2_fixed_n and auc_q_fixed_n (within groups of an
        identical number of grains), positives_coverage_n and direction.
    """
    H, q, y = microstates("test_pool", L)
    A = modes(H)
    del H
    r2_ = r2_from_modes(A, q)
    (m1, n1), (m2, n2) = PAIR
    direction = contrast_direction(A[:, m1, n1].copy(), A[:, m2, n2].copy(), q, y, L)
    del A
    n = np.rint(q * L * L).astype(np.int64)
    auc_n, cov_n, _ = conditional_auc(y, r2_, n)
    return dict(
        spearman_R2_q=float(spearmanr(r2_, q).statistic),
        spearman_R2_q_positives=float(spearmanr(r2_[y == 1], q[y == 1]).statistic),
        spearman_R2_q_negatives=float(spearmanr(r2_[y == 0], q[y == 0]).statistic),
        auc_R2=float(roc_auc_score(y, r2_)),
        auc_R2_fixed_n=float(auc_n), positives_coverage_n=float(cov_n),
        auc_q_fixed_n=float(conditional_auc(y, q, n)[0]),
        direction=direction)


def relation_to_scale(L, seed=0):
    """
    Relate R2 to the scale of the signal on the test set (Section 3.5, Appendix C).

    Parameters:
        L (int): system size.
        seed (int): seed of the rearrangement.

    Returns:
        dict: auc_R2, auc_V4, spearman_V4_R2 and the AUC of R2 after the grains are rearranged inside blocks
        of side L/8, L/4 and L/2.
    """
    H, q, y = microstates("test_pool", L)
    r2_ = r2_fast(H, q)
    V4 = block_variance(H, 4)
    d = dict(auc_R2=float(roc_auc_score(y, r2_)), auc_V4=float(roc_auc_score(y, V4)),
             spearman_V4_R2=float(spearmanr(V4, r2_).correlation))
    rng = np.random.default_rng(seed)
    for div in (8, 4, 2):
        Hs = np.concatenate([shuffle_within_blocks(H[i:i + 1500].astype(np.float32), L // div, rng)
                             for i in range(0, len(H), 1500)])
        assert np.array_equal(Hs.sum(axis=(1, 2)), H.sum(axis=(1, 2))), "the rearrangement changed the number of grains"
        d[f"auc_R2_in_blocks_L/{div}"] = float(roc_auc_score(y, r2_fast(Hs, q)))
        del Hs
    return dict(test=d)
