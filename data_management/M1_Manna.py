import os
import struct
import zipfile
import numpy as np
from numba import njit

DATA_DIR = os.environ.get("SOC_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_repository", "grids"))

WARMUP = 3.5
S_STAR = 0.353
D_AVALANCHE = 2.75
N_FRAMES = 50

@njit(cache=True)
def run_Manna(L, T, grid, s_star=0.353, n_frames=50, max_pos=2500, neg_spacing=1000, seed=-1, max_neg=0):
    """
    Simulate T steps of the Manna model and collect examples.

    Parameters:
        L (int): system size.
        T (int): number of steps.
        grid (ndarray): initial state, shape (L + 2, L + 2), with a one-site margin.
        s_star (float): extreme threshold in units of L^2.75.
        n_frames (int): number of frames stored for every example.
        max_pos (int): maximum number of positives, the states followed by an extreme avalanche.
        neg_spacing (int): minimum number of steps between two negatives.
        seed (int): random seed, not set when negative.
        max_neg (int): maximum number of negatives, the quiet states.

    Returns:
        tuple: (grid, s_all, q_all, windows, labels, indices). s_all[t] is the avalanche size at step t,
        q_all[t] the density after that step, windows[i] the n_frames states before the grain is added at
        step indices[i], and labels[i] is 1 for a positive and 0 for a negative.
    """
    if seed >= 0: np.random.seed(seed)
    dX = np.array([-1, 0, 1, 0], dtype=np.int32)
    dY = np.array([0, 1, 0, -1], dtype=np.int32)
    queue_size = max(4 * L * L, int(4 * L**2.75))
    queue = np.zeros((queue_size, 2), dtype=np.int32)
    q_size = np.sum(grid)

    LBP = int(s_star * L ** 2.75)

    buf = np.zeros((n_frames, L, L), dtype=np.int8)
    buf_idx = 0
    buf_filled = 0

    s_all = np.zeros(T, dtype=np.int32)
    q_all = np.zeros(T, dtype=np.float32)

    max_examples = max_pos + max_neg
    all_grids = np.zeros((max_examples, n_frames, L, L), dtype=np.int8)
    all_labels = np.zeros(max_examples, dtype=np.int8)
    all_indices = np.zeros(max_examples, dtype=np.int64)
    cnt_pos = 0
    cnt_neg = 0
    cnt = 0
    last_neg_t = -neg_spacing - 1

    for t in range(T):
        if t % 10_000_000 == 0: print(t, cnt_pos, cnt_neg)
        if cnt < max_examples:
            buf[buf_idx] = grid[1:-1, 1:-1]
            buf_idx = (buf_idx + 1) % n_frames
            if buf_filled < n_frames:
                buf_filled += 1

        s_size = 0
        x, y = np.random.randint(1, L+1), np.random.randint(1, L+1)
        grid[x, y] += 1
        q_size += 1
        if grid[x, y] >= 2:
            head, tail = 0, 0
            queue[tail, 0] = x
            queue[tail, 1] = y
            tail += 1
            while head < tail:
                x, y = queue[head, 0], queue[head, 1]
                head += 1
                if grid[x, y] < 2: continue
                s_size += 1
                grid[x, y] -= 2
                n = np.random.randint(16)
                n_1 = n % 4
                n_2 = n >> 2
                x1, y1 = x + dX[n_1], y + dY[n_1]
                x2, y2 = x + dX[n_2], y + dY[n_2]
                grid[x1, y1] += 1
                grid[x2, y2] += 1
                for i in range(3):
                    if i == 0: xi, yi = x1, y1
                    elif i == 1: xi, yi = x2, y2
                    else: xi, yi = x, y
                    if not (1 <= xi <= L and 1 <= yi <= L):
                        q_size -= grid[xi, yi]
                        grid[xi, yi] = 0
                    elif grid[xi, yi] >= 2:
                        queue[tail, 0] = xi
                        queue[tail, 1] = yi
                        tail += 1

        s_all[t] = s_size
        q_all[t] = q_size / (L ** 2)

        if cnt < max_examples and buf_filled >= n_frames:
            record = False
            label = np.int8(0)

            if s_size > LBP and cnt_pos < max_pos:
                record = True
                label = np.int8(1)
                cnt_pos += 1

            elif s_size <= LBP and (t - last_neg_t) >= neg_spacing:
                if cnt_neg < max_neg:
                    record = True
                    label = np.int8(0)
                    cnt_neg += 1
                    last_neg_t = t

            if record:
                for k in range(n_frames):
                    src = (buf_idx + k) % n_frames
                    all_grids[cnt, k] = buf[src]
                all_labels[cnt] = label
                all_indices[cnt] = t
                cnt += 1

    return grid, s_all, q_all, all_grids[:cnt], all_labels[:cnt], all_indices[:cnt]

def run_for(scope, L, N=500_000_000, grid=None, max_pos=2500, max_neg=10_000, s_star=0.353, n_frames=50, seed=-1):
    """
    Run the warm-up or one collection phase.

    Parameters:
        scope (str): "load" for the warm-up (3.5 L^2.75 steps from an empty lattice), "train" or "test" for
            a collection phase.
        L (int): system size.
        N (int): number of steps of a collection phase.
        grid (ndarray): state from which a collection phase starts.
        max_pos (int): maximum number of positives.
        max_neg (int): maximum number of negatives, taken every N // max_neg steps.
        s_star (float): extreme threshold in units of L^2.75.
        n_frames (int): number of frames stored for every example.
        seed (int): random seed.

    Returns:
        tuple: (grid, s_all, q_all, windows, labels, indices), as returned by run_Manna.
    """
    if scope == "load":
        max_pos, max_neg = 1, 1
        time = int(WARMUP * L**2.75)
        grid = np.zeros((L + 2, L + 2), dtype=np.int32)
    elif scope in ["train", "test"]:
        time = int(N)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    grid, s_all, q_all, all_grids, all_labels, all_indices = run_Manna(
        L=L, T=time, grid=grid, s_star=s_star, n_frames=n_frames, max_pos=max_pos,
        max_neg=max_neg, neg_spacing=max(1, N // max(max_neg, 1)), seed=seed)
    print(f"{scope.upper()} (dm) phase ({time} steps) - done!", flush=True)
    print(f"  Collected pool: {len(all_labels)} ({int(all_labels.sum())} pos, "
          f"{int((all_labels == 0).sum())} neg-candidates)")
    return grid, s_all, q_all, all_grids, all_labels, all_indices

def save_phase(scope, L, grid, s_all=None, q_all=None, grids=None, labels=None, indices=None, data_dir=None):
    path = f'{data_dir or DATA_DIR}/spatial_temp_diag_{scope}_{L}.npz'
    data = {'grid': grid}
    if s_all is not None: data['s_all'] = s_all
    if q_all is not None: data['q_all'] = q_all
    if grids is not None: data['grids'] = grids
    if labels is not None: data['labels'] = labels
    if indices is not None: data['indices'] = indices
    np.savez(path, **data)
    print(f"Saved: {path}")

def npz_memmap(path, name):
    """
    Open one array of an uncompressed .npz file as a memory map.

    Parameters:
        path (str): path to a file saved with np.savez.
        name (str): name of the array.

    Returns:
        np.memmap: read-only view of the array.
    """
    if not name.endswith(".npy"):
        name += ".npy"
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"compressed array ({info.filename}): save with np.savez, not savez_compressed")
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            nlen, elen = struct.unpack("<HH", f.read(30)[26:30])
            f.seek(info.header_offset + 30 + nlen + elen)
            ver = np.lib.format.read_magic(f)
            read_header = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read_header(f)
            offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")

def phase_path(scope, L):
    return os.path.join(DATA_DIR, f"spatial_temp_diag_{scope}_{L}.npz")

def phase_array(scope, L, name):
    """
    Open one array of a stored phase as a memory map.

    Parameters:
        scope (str): "train_pool" or "test_pool".
        L (int): system size.
        name (str): name of the array, e.g. "q_all".

    Returns:
        np.memmap: read-only view of the array.
    """
    return npz_memmap(phase_path(scope, L), name)