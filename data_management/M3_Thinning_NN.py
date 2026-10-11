import functools
import json
import math
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from M1_Manna import N_FRAMES, npz_memmap, phase_path, phase_array
from M2_Shapoval import table

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_repository", "models")


def thin_frame(flat, K, rng):
    """
    Thin one flattened binary frame to exactly K grains.

    Parameters:
        flat (ndarray): flattened binary frame.
        K (int): number of grains to keep.
        rng (np.random.Generator): random generator.

    Returns:
        ndarray: new float32 array with exactly K grains.
    """
    y = np.array(flat, dtype=np.float32, copy=True)
    grains = np.flatnonzero(y)
    surplus = len(grains) - K
    if surplus > 0:
        y[rng.choice(grains, surplus, replace=False)] = 0
    elif surplus < 0:
        raise ValueError(f"the frame has {len(grains)} grains, fewer than K={K}")
    return y


def thin(frames, K, rng):
    """
    Thin every frame to exactly K grains.

    Parameters:
        frames (ndarray): binary frames, shape (N, L, L).
        K (int): number of grains to keep.
        rng (np.random.Generator): random generator.

    Returns:
        ndarray: new float32 array of the same shape.
    """
    a = np.array(frames, dtype=np.float32, copy=True)
    shape = a.shape
    flat = a.reshape(-1, shape[-2] * shape[-1])
    for i in range(len(flat)):
        flat[i] = thin_frame(flat[i], K, rng)
    return flat.reshape(shape)


def shuffle(frames, rng, chunk=2000):
    """
    Scatter the grains of every frame at random over the lattice (null level).

    Parameters:
        frames (ndarray): frames, shape (N, L, L).
        rng (np.random.Generator): random generator.
        chunk (int): number of frames processed at once.

    Returns:
        ndarray: shuffled frames with the same number of grains.
    """
    a = np.asarray(frames)
    shape = a.shape
    flat = a.reshape(-1, shape[-2] * shape[-1])
    out = np.empty_like(flat)
    for i in range(0, len(flat), chunk):
        b = flat[i:i + chunk]
        out[i:i + chunk] = np.take_along_axis(b, np.argsort(rng.random(b.shape), axis=1), axis=1)
    return out.reshape(shape)


def rotate_flip(x, rng):
    """
    Apply a random element of the symmetry group D4 to the last two axes.

    Parameters:
        x (ndarray): frame or stack of frames.
        rng (np.random.Generator): random generator.

    Returns:
        ndarray: rotated and possibly reflected view of x.
    """
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    y = np.rot90(x, k, axes=(-2, -1))
    return y[..., ::-1] if flip else y


class ThinnedDataset:
    """
    Torch dataset of last frames thinned to K grains at each read and centered by K / L^2.

    Parameters:
        frames (ndarray): frames, shape (N, 1, L, L).
        labels (ndarray): labels of the examples.
        K (int): number of grains to keep.
        shuffled (bool): scatter the K grains at random (null level).
        fixed (bool): use the same thinning draw for an example at every read.
        seed (int): seed of the thinning draws.
        d4 (bool): apply a random rotation or reflection.
    """

    def __init__(self, frames, labels, K, shuffled=False, fixed=False, seed=0, d4=False):
        self.g = frames
        self.y = np.asarray(labels, dtype=np.float32)
        self.K = int(K)
        self.shuffled = shuffled
        self.fixed = fixed
        self.seed = seed
        self.d4 = d4
        self._rng = None
        L = frames.shape[-1]
        self.offset = np.float32(self.K / (L * L))

    def __len__(self):
        return len(self.g)

    def __getitem__(self, i):
        if self.fixed:
            rng = np.random.default_rng((self.seed, i))
        else:
            if self._rng is None:
                self._rng = np.random.default_rng([self.seed, 0])
            rng = self._rng
        x = np.asarray(self.g[i, -1:], dtype=np.float32)
        flat = x.reshape(1, -1)
        if self.shuffled:
            chosen = np.argpartition(rng.random(flat.shape[1]), self.K)[:self.K]
            flat[0] = 0.0
            flat[0][chosen] = 1.0
        else:
            flat[0] = thin_frame(flat[0], self.K, rng)
        x = flat.reshape(x.shape) - self.offset
        if self.d4:
            x = rotate_flip(x, rng)
        return torch.from_numpy(np.ascontiguousarray(x)), self.y[i]


def _window_grain_counts(L):
    """
    Count the grains in every frame of every training window.

    Parameters:
        L (int): system size.

    Returns:
        list: one array of grain numbers per window.
    """
    idx = np.array(phase_array("train_pool", L, "indices"))
    q_all = phase_array("train_pool", L, "q_all")
    return [np.rint(np.asarray(q_all[max(t - N_FRAMES, 0):t], dtype=np.float64) * L * L) for t in idx]


@functools.lru_cache(maxsize=None)
def K_train(L):
    """
    Get the thinning level K, the smallest number of grains in any frame of the training windows.

    Parameters:
        L (int): system size.

    Returns:
        int: the thinning level K.
    """
    return int(min(n.min() for n in _window_grain_counts(L)))


def thinning_level(L):
    """
    Compute the thinning level and the fraction of grains it removes (Table 2).

    Parameters:
        L (int): system size.

    Returns:
        dict: K, train_frames, removed_max, removed_mean, n_test and rejected_last_frame (test examples
        whose last frame has fewer than K grains).
    """
    windows = _window_grain_counts(L)
    n = np.concatenate(windows)
    K = int(n.min())
    u = 1 - K / n
    idx_te = np.array(phase_array("test_pool", L, "indices"))
    q_te = phase_array("test_pool", L, "q_all")
    n_last = np.rint(np.asarray(q_te[idx_te - 1], dtype=np.float64) * L * L).astype(np.int64)
    return dict(K=K, train_frames=len(windows) * N_FRAMES,
                removed_max=float(u.max()), removed_mean=float(u.mean()),
                n_test=len(n_last), rejected_last_frame=int((n_last < K).sum()))


def last_frames(scope, L):
    """
    Load the last frame of every example.

    Parameters:
        scope (str): "train_pool" or "test_pool".
        L (int): system size.

    Returns:
        tuple: (X, y, n) with frames of shape (N, 1, L, L), labels and grain numbers.
    """
    p = phase_path(scope, L)
    lab = np.array(npz_memmap(p, "labels")).astype(int)
    idx = np.array(npz_memmap(p, "indices"))
    q = np.array(npz_memmap(p, "q_all")[idx - 1], dtype=np.float64)
    g = npz_memmap(p, "grids")
    X = np.empty((len(lab), 1, L, L), dtype=np.int8)
    for i in range(len(lab)):
        X[i, 0] = g[i, -1]
    return X, lab, np.rint(q * L * L).astype(np.int64)


def frames_2d(scope, L, K=None):
    """
    Load the last frames as 2D arrays, optionally only those with at least K grains.

    Parameters:
        scope (str): "train_pool" or "test_pool".
        L (int): system size.
        K (int): minimum number of grains, no filtering when None.

    Returns:
        tuple: (X, y, n, rejected) with frames of shape (N, L, L), labels, grain numbers and the number of
        rejected examples.
    """
    p = phase_path(scope, L)
    lab = np.array(npz_memmap(p, "labels")).astype(int)
    idx = np.array(npz_memmap(p, "indices"))
    q = np.array(npz_memmap(p, "q_all")[idx - 1], dtype=np.float64)
    n = np.rint(q * L * L).astype(np.int64)
    ok = np.ones(len(n), dtype=bool) if K is None else (n >= K)
    g = npz_memmap(p, "grids")
    X = np.empty((int(ok.sum()), L, L), dtype=np.float32)
    for j, i in enumerate(np.flatnonzero(ok)):
        X[j] = g[int(i), -1]
    del g
    assert np.array_equal(X.sum((1, 2)).astype(np.int64), n[ok])
    return X, lab[ok], n[ok], int((~ok).sum())


def trim_test(K, X, y, n):
    """
    Drop the examples with fewer than K grains.

    Parameters:
        K (int): thinning level.
        X (ndarray): frames.
        y (ndarray): labels.
        n (ndarray): grain numbers.

    Returns:
        tuple: (X, y, n, rejected) for the remaining examples and the number of rejected ones.
    """
    ok = n >= K
    return X[ok], y[ok], n[ok], int((~ok).sum())


class RawDataset:
    """
    Torch dataset of last frames without thinning, centered by 0.71.

    Parameters:
        frames (ndarray): frames, shape (N, 1, L, L).
        labels (ndarray): labels of the examples.
        d4 (bool): apply a random rotation or reflection.
        seed (int): seed of the augmentation.
    """
    def __init__(self, frames, labels, d4=False, seed=0):
        self.g, self.y = frames, np.asarray(labels, np.float32)
        self.offset = np.float32(0.71)
        self.d4 = d4
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.g)

    def __getitem__(self, i):
        x = np.asarray(self.g[i, -1:], dtype=np.float32) - self.offset
        if self.d4:
            x = np.ascontiguousarray(rotate_flip(x, self._rng))
        return torch.from_numpy(x), self.y[i]


def block_variance(frames2d, nb):
    """
    Compute the block-density variance V_l of every frame (equation eq:Vl).

    Parameters:
        frames2d (ndarray): frames, shape (N, L, L).
        nb (int): number of blocks per side, a divisor of L.

    Returns:
        ndarray: one value per frame.
    """
    a = np.asarray(frames2d, dtype=np.float32)
    N, L, _ = a.shape
    if L % nb:
        raise ValueError(f"nb={nb} does not divide L={L}: cropping the lattice breaks the fixed number of grains")
    b = L // nb
    dens = a.reshape(N, nb, b, nb, b).mean(axis=(2, 4))
    return dens.reshape(N, -1).var(axis=1)


def conditional_auc(y, score, cell):
    """
    Compute the AUC within groups of identical cell value, weighted by the number of pairs (equation eq:auck).

    Parameters:
        y (ndarray): labels.
        score (ndarray): predictor.
        cell (ndarray): group of every example, e.g. the original number of grains.

    Returns:
        tuple: (auc, fraction of positives in groups with both classes, number of such groups).
    """
    num = den = 0.0
    pos = groups = 0
    for v in np.unique(cell):
        m = cell == v
        yy = y[m]
        p, n_ = int(yy.sum()), int((yy == 0).sum())
        if p == 0 or n_ == 0:
            continue
        num += roc_auc_score(yy, score[m]) * p * n_
        den += p * n_
        pos += p
        groups += 1
    return num / den, pos / int(np.sum(y)), groups


def blocks_for(L):
    """
    Get the number of convolutional blocks for a system size (3, 4, 5 for L = 32, 64, 128).

    Parameters:
        L (int): system size.

    Returns:
        int: number of blocks.
    """
    return 4 + round(math.log2(L / 64))


def _stack(inputs, channels):
    """
    Build a sequence of Conv-BN-ReLU blocks with stride 2.

    Parameters:
        inputs (int): number of input channels.
        channels (list): output channels of the consecutive blocks.

    Returns:
        nn.Sequential: the blocks.
    """
    layers, c_in = [], inputs
    for c in channels:
        layers += [nn.Conv2d(c_in, c, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        c_in = c
    return nn.Sequential(*layers)


def _diag_channels(n):
    """
    List the output channels of n blocks: 16, 32, 64, ..., 32.

    Parameters:
        n (int): number of blocks.

    Returns:
        list: output channels.
    """
    return [16, 32] + [64] * (n - 3) + [32]


class CNN_2D_Single_Stat(nn.Module):
    """
    Single-frame convolutional network with a statistical head (mean and standard deviation of the feature
    maps).

    Parameters:
        blocks (int): number of convolutional blocks.
    """
    def __init__(self, blocks=4):
        super().__init__()
        self.blocks = blocks
        self.features = _stack(1, _diag_channels(blocks))
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        f = self.features(x)
        stat = torch.cat([f.mean(dim=(-2, -1)), f.std(dim=(-2, -1))], dim=1)
        return self.classifier(stat).squeeze(-1)


def network_summary(L):
    """
    Describe the network architecture for a system size.

    Parameters:
        L (int): system size.

    Returns:
        dict: blocks, parameters, channels, feature_map, numbers_per_image, receptive_field and
        receptive_field_over_L.
    """
    m = CNN_2D_Single_Stat(blocks=blocks_for(L))
    convs = [w for w in m.features if isinstance(w, nn.Conv2d)]
    field, step = 1, 1
    for w in convs:
        field += (w.kernel_size[0] - 1) * step
        step *= w.stride[0]
    channels, height, width = m.features(torch.zeros(1, 1, L, L)).shape[1:]
    return dict(blocks=len(convs), parameters=sum(p.numel() for p in m.parameters()), channels=channels,
                feature_map=f"{height} x {width}", numbers_per_image=m.classifier.in_features,
                receptive_field=field, receptive_field_over_L=field / L)


VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
HISTORY_COLUMNS = ("train_loss", "validation_loss", "validation_auc")
VARIANTS = {"thinned": (True, False), "null": (True, True), "raw": (False, False)}


def weights_file(L, variant, seed, directory=None):
    return os.path.join(directory or os.path.join(MODELS_DIR, f"models_L{L}"), f"diagcnn_stat_{variant}_{seed}.pt")


def history_file(L, directory=None):
    return os.path.join(directory or os.path.join(MODELS_DIR, f"models_L{L}"), "training_history.json")


def training_history(L, directory=None):
    """
    Read the training log of the networks.

    Parameters:
        L (int): system size.
        directory (str): folder with the weights and the log, by default models_L{L} in MODELS_DIR.

    Returns:
        dict: {variant: {seed: array of shape (epochs, 3)}} with columns as in HISTORY_COLUMNS.
    """
    with open(history_file(L, directory)) as f:
        d = json.load(f)
    assert tuple(d["columns"]) == HISTORY_COLUMNS
    return {variant: {int(s): np.array(h, dtype=np.float64) for s, h in seeds.items()}
            for variant, seeds in d["history"].items()}


def _append_history(L, variant, seed, history, directory=None):
    """
    Write the history of one training to the log.

    Parameters:
        L (int): system size.
        variant (str): network variant.
        seed (int): training seed.
        history (list): one row of HISTORY_COLUMNS per epoch.
        directory (str): folder with the weights and the log, by default models_L{L} in MODELS_DIR.
    """
    path = history_file(L, directory)
    d = {"columns": list(HISTORY_COLUMNS), "history": {}}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        assert tuple(d["columns"]) == HISTORY_COLUMNS
    d["history"].setdefault(variant, {})[str(seed)] = [[float(v) for v in epoch] for epoch in history]
    with open(path, "w") as f:
        json.dump(d, f)


def set_seed(seed, deterministic=False):
    """
    Seed all random generators used in training.

    Parameters:
        seed (int): random seed.
        deterministic (bool): switch cuDNN and torch to deterministic algorithms.

    Returns:
        torch.Generator: generator for the DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def validation_part(Xtr, ytr, K, variant, seed):
    """
    Split the training set for one seed into a training and a validation part.

    Parameters:
        Xtr (ndarray): training frames, shape (N, 1, L, L).
        ytr (ndarray): training labels.
        K (int): thinning level.
        variant (str): network variant.
        seed (int): training seed.

    Returns:
        tuple: (indices of the training part, validation dataset with a fixed thinning).
    """
    thinned, shuffled = VARIANTS[variant]
    idx = np.arange(len(ytr))
    tr, va = train_test_split(idx, test_size=VALIDATION_FRACTION, stratify=ytr, random_state=seed)
    Zva = (ThinnedDataset(Xtr, ytr, K, shuffled=shuffled, fixed=True, seed=100 + seed) if thinned
           else RawDataset(Xtr, ytr))
    return tr, Subset(Zva, va)


def validation_loader(ds_va, batch_size):
    return DataLoader(ds_va, batch_size=max(batch_size, 32), shuffle=False)


def loss_and_auc(m, loader, dev):
    """
    Evaluate a model on a data loader.

    Parameters:
        m (nn.Module): model.
        loader (DataLoader): data to evaluate on.
        dev (str): device.

    Returns:
        tuple: (mean binary cross-entropy, AUC).
    """
    loss_sum = nn.BCEWithLogitsLoss(reduction="sum")
    m.eval()
    total, pr, ys = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(dev), y.to(dev)
            lo = m(X)
            total += loss_sum(lo, y).item()
            pr.append(torch.sigmoid(lo).cpu().numpy())
            ys.append(y.cpu().numpy())
    ys = np.concatenate(ys)
    return total / len(ys), float(roc_auc_score(ys, np.concatenate(pr)))


def fit(ds_tr, ds_va, model_path, epochs, patience, model_class, lr, wd, batch, gen):
    """
    Train one network with Adam and early stopping on the validation loss.

    Parameters:
        ds_tr (Dataset): training part.
        ds_va (Dataset): validation part.
        model_path (str): file for the weights of the best epoch.
        epochs (int): maximum number of epochs.
        patience (int): number of epochs without improvement before stopping.
        model_class (callable): constructor of the model.
        lr (float): learning rate.
        wd (float): weight decay.
        batch (int): batch size.
        gen (torch.Generator): generator for shuffling.

    Returns:
        tuple: (model with the weights of the best epoch, history with one row of HISTORY_COLUMNS per
        epoch).
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = model_class().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    tl = DataLoader(ds_tr, batch_size=batch, shuffle=True, generator=gen)
    vl = validation_loader(ds_va, batch)
    best_loss, no_gain, history = float("inf"), 0, []
    for e in range(epochs):
        m.train()
        total_tr, n_tr = 0.0, 0
        for X, y in tl:
            X, y = X.to(dev), y.to(dev)
            opt.zero_grad()
            l = loss_fn(m(X), y)
            l.backward()
            opt.step()
            total_tr += l.item() * len(y)
            n_tr += len(y)
        val_loss, val_auc = loss_and_auc(m, vl, dev)
        history.append((total_tr / n_tr, val_loss, val_auc))
        if val_loss < best_loss:
            best_loss, no_gain = val_loss, 0
            torch.save(m.state_dict(), model_path)
        else:
            no_gain += 1
            if no_gain >= patience:
                break
    m.load_state_dict(torch.load(model_path, weights_only=True))
    return m, history


def train_networks(L, seeds=(0, 1, 2, 3, 4), variants=("thinned", "null", "raw"), epochs=100, patience=10, lr=1e-3, wd=1e-4, batch_size=BATCH_SIZE, deterministic=False, directory=None):
    """
    Train the networks for one system size and save the weights and the training log.

    Parameters:
        L (int): system size.
        seeds (tuple): training seeds.
        variants (tuple): network variants, "thinned", "null" or "raw".
        epochs (int): maximum number of epochs.
        patience (int): number of epochs without improvement before stopping.
        lr (float): learning rate.
        wd (float): weight decay.
        batch_size (int): batch size.
        deterministic (bool): train in the deterministic mode.
        directory (str): folder with the weights and the log, by default models_L{L} in MODELS_DIR.

    Returns:
        dict: {variant: [number of epochs for every seed]}.
    """
    K = K_train(L)
    Xtr, ytr, _ = last_frames("train_pool", L)
    model_class = functools.partial(CNN_2D_Single_Stat, blocks=blocks_for(L))
    folder = directory or os.path.join(MODELS_DIR, f"models_L{L}")
    os.makedirs(folder, exist_ok=True)
    result = {}
    for name in variants:
        thinned, shuffled = VARIANTS[name]
        n_epochs = []
        for s in seeds:
            gen = set_seed(s, deterministic)
            tr, ds_va = validation_part(Xtr, ytr, K, name, s)
            Ztr = (ThinnedDataset(Xtr, ytr, K, shuffled=shuffled, seed=s, d4=True) if thinned
                   else RawDataset(Xtr, ytr, d4=True, seed=s))
            m, history = fit(Subset(Ztr, tr), ds_va, weights_file(L, name, s, folder),
                             epochs, patience, model_class, lr, wd, batch_size, gen)
            _append_history(L, name, s, history, folder)
            n_epochs.append(len(history))
            del m
            torch.cuda.empty_cache()
        result[name] = n_epochs
    return result


EVAL_SEED = 1000
SATURATION_TOLERANCE = 0.01


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(L, variant, seed, dev=None):
    """
    Load stored weights as a model in evaluation mode.

    Parameters:
        L (int): system size.
        variant (str): network variant.
        seed (int): training seed.
        dev (str): device, chosen automatically when None.

    Returns:
        CNN_2D_Single_Stat: the model.
    """
    dev = dev or device()
    m = CNN_2D_Single_Stat(blocks=blocks_for(L)).to(dev)
    m.load_state_dict(torch.load(weights_file(L, variant, seed), map_location=dev, weights_only=True))
    m.eval()
    return m


def probe(L, nb=4, null_repeats=5):
    """
    Compute the AUC of density and of the block variance on the test set (Table 4, columns without the network).

    Parameters:
        L (int): system size.
        nb (int): number of blocks per side.
        null_repeats (int): number of independent draws of the null level.

    Returns:
        dict: K, nb, n_test, rejected_test, auc_q_raw, auc_q_thinned, auc_V_raw, auc_V_thinned,
        auc_V_thinned_fixed_n (within groups of identical original number of grains), and auc_V_null with
        its spread and the individual draws.
    """
    K = K_train(L)
    X, lab, n, rejected = frames_2d("test_pool", L, K)

    Xp = thin(X, K, np.random.default_rng(0))
    assert np.all(Xp.sum((1, 2)) == K)
    nulls = []
    for j in range(null_repeats):
        Xtp = thin(shuffle(X, np.random.default_rng(100 + j)), K, np.random.default_rng(200 + j))
        assert np.all(Xtp.sum((1, 2)) == K)
        nulls.append(block_variance(Xtp, nb))

    V_thin = block_variance(Xp, nb)
    ak = np.array([roc_auc_score(lab, s) for s in nulls])
    return dict(K=K, nb=nb, n_test=len(lab), rejected_test=rejected,
                auc_q_raw=float(roc_auc_score(lab, np.asarray(n, dtype=np.float64) / (L * L))),
                auc_q_thinned=float(roc_auc_score(lab, Xp.sum((1, 2)) / (L * L))),
                auc_V_raw=float(roc_auc_score(lab, block_variance(X, nb))),
                auc_V_thinned=float(roc_auc_score(lab, V_thin)),
                auc_V_thinned_fixed_n=float(conditional_auc(lab, V_thin, n)[0]),
                auc_V_null=float(ak.mean()), auc_V_null_spread=float(ak.std()),
                auc_V_null_draws=[float(v) for v in ak])


def network_scores(model, X, y, K, shuffled, seed, raw=False, batch=64):
    """
    Compute the probabilities of a model on one fixed thinning draw.

    Parameters:
        model (nn.Module): network.
        X (ndarray): frames, shape (N, 1, L, L).
        y (ndarray): labels.
        K (int): thinning level.
        shuffled (bool): evaluate on the null level.
        seed (int): seed of the thinning draw.
        raw (bool): evaluate without thinning.
        batch (int): batch size.

    Returns:
        ndarray: one probability per example.
    """
    Z = RawDataset(X, y) if raw else ThinnedDataset(X, y, K, shuffled=shuffled, fixed=True, seed=seed)
    dev = next(model.parameters()).device
    pr = []
    with torch.no_grad():
        for xb, _ in DataLoader(Z, batch_size=batch, shuffle=False):
            pr.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
    return np.concatenate(pr).ravel()


def _summarize_seeds(y, P, n, n_boot, n_perm, seed=1):
    """
    Summarize the AUC of several seeds with a bootstrap interval and a permutation test.

    Parameters:
        y (ndarray): labels.
        P (ndarray): scores, shape (seeds, examples).
        n (ndarray): original number of grains of every example.
        n_boot (int): number of bootstrap resamples.
        n_perm (int): number of label permutations.
        seed (int): seed of the resampling.

    Returns:
        dict: auc (mean over seeds), spread (standard deviation over seeds), ci_auc, p and auc_fixed_n.
    """
    y = np.asarray(y)
    P = np.asarray(P)
    aucs = np.array([roc_auc_score(y, p) for p in P])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        boot[i] = (np.mean([roc_auc_score(y[b], p[b]) for p in P])
                   if len(np.unique(y[b])) > 1 else np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if np.mean([roc_auc_score(yp, p) for p in P]) >= aucs.mean():
            count += 1

    return dict(auc=float(aucs.mean()), spread=float(aucs.std()),
                ci_auc=[float(lo), float(hi)], p=(count + 1) / (n_perm + 1),
                auc_fixed_n=float(np.mean([conditional_auc(y, p, n)[0] for p in P])))


def network(L, seeds=(0, 1, 2, 3, 4), n_boot=2000, n_perm=2000):
    """
    Evaluate the stored networks on the test set (Table 4, network columns, and Section 3.1).

    Parameters:
        L (int): system size.
        seeds (tuple): training seeds.
        n_boot (int): number of bootstrap resamples.
        n_perm (int): number of label permutations.

    Returns:
        dict: K, rejected_test, n_test, seeds and the results for the variants thinned, null and raw.
    """
    K = K_train(L)
    Xte, yte, nte = last_frames("test_pool", L)
    Xte, yte, nte, rejected = trim_test(K, Xte, yte, nte)
    dev = device()
    r = {"K": K, "rejected_test": rejected, "n_test": len(yte), "seeds": list(seeds)}
    for variant, shuffled in (("thinned", False), ("null", True), ("raw", False)):
        P = []
        for s in seeds:
            m = load_model(L, variant, s, dev)
            P.append(network_scores(m, Xte, yte, K, shuffled, EVAL_SEED, raw=(variant == "raw")))
            del m
            torch.cuda.empty_cache()
        if variant == "raw":
            r[variant] = dict(auc=float(np.mean([roc_auc_score(yte, p) for p in P])))
        else:
            r[variant] = _summarize_seeds(yte, P, nte, n_boot, n_perm)
    return r


def learning_curves(L, seeds=(0, 1, 2, 3, 4)):
    """
    Summarize the training log and check it against the stored weights (Appendix B).

    Parameters:
        L (int): system size.
        seeds (tuple): training seeds.

    Returns:
        dict: for every variant epochs, saved_epoch, validation_auc, saturation_epoch, drift_after_save,
        weights (loss, auc and auc_train of the stored weights), validation_auc_mean, auc_train_mean,
        auc_range and log_mismatch (largest difference between the log and the recomputed values).
    """
    K = K_train(L)
    Xtr, ytr, _ = last_frames("train_pool", L)
    H = training_history(L)
    dev = device()
    r = {"seeds": list(seeds), "saturation_tolerance": SATURATION_TOLERANCE}
    for variant in VARIANTS:
        w = dict(epochs=[], saved_epoch=[], validation_auc=[], saturation_epoch=[], drift_after_save=[],
                 weights=dict(loss=[], auc=[], auc_train=[]))
        mismatch = 0.0
        for s in seeds:
            h = H[variant][s]
            e = int(np.argmin(h[:, 1]))
            auc = h[:, 2]
            w["epochs"].append(len(h))
            w["saved_epoch"].append(e + 1)
            w["validation_auc"].append(float(auc[e]))
            w["saturation_epoch"].append(int(np.argmax(auc >= auc[e] - SATURATION_TOLERANCE)) + 1)
            w["drift_after_save"].append(float(np.abs(auc[e:] - auc[e]).max()))

            tr, ds_va = validation_part(Xtr, ytr, K, variant, s)
            m = load_model(L, variant, s, dev)
            loss_w, auc_w = loss_and_auc(m, validation_loader(ds_va, BATCH_SIZE), dev)
            _, auc_tr = loss_and_auc(m, validation_loader(Subset(ds_va.dataset, tr), BATCH_SIZE), dev)
            del m
            torch.cuda.empty_cache()
            w["weights"]["loss"].append(loss_w)
            w["weights"]["auc"].append(auc_w)
            w["weights"]["auc_train"].append(auc_tr)
            mismatch = max(mismatch, abs(loss_w - h[e, 1]), abs(auc_w - auc[e]))
        w["validation_auc_mean"] = float(np.mean(w["validation_auc"]))
        w["auc_train_mean"] = float(np.mean(w["weights"]["auc_train"]))
        w["auc_range"] = [float(min(H[variant][s][:, 2].min() for s in seeds)),
                          float(max(H[variant][s][:, 2].max() for s in seeds))]
        w["log_mismatch"] = float(mismatch)
        r[variant] = w
    return r
