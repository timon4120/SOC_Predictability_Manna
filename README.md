<p align = "center" style="font-size:200%;">
<b>Beyond global density in a stochastic sandpile: the role of spatial signatures in the predictability of extremes</b></p>
<p align = "center">K.P. Orzechowski, J. Sienkiewicz & P. Fronczak</p>
<p align = "center">Overview</p>

---

This directory contains the code that produces every number, table and figure of the paper. It simulates the
Abelian Manna sandpile on lattices of size $L \in \{32, 64, 128\}$ and checks whether the arrangement of grains, and
not only their density, announces an extreme avalanche. There is one module per notebook: the module (in
`data_management/`) computes and the notebook (in `data_analysis/`) shows the results.

<br>

| notebook | module | what it computes | where in the paper |
|---|---|---|---|
| `W1_Manna` | `M1_Manna` | simulation of the Manna model, collection of examples, saving the data to `data_repository/grids/` | Sections 2.1 and 2.2 |
| `W2_Shapoval` | `M2_Shapoval` | the collected examples, their consistency with the stream and their independence, prediction from density alone on the full stream | Tables 1 and 3, Figure 1, Sections 2.1, 2.2 and 2.7 |
| `W3_Thinning_NN` | `M3_Thinning_NN` | thinning level, block variance and the network at a fixed number of grains, learning curves | Tables 2 and 4, Appendix B (Figure B.1) |
| `W4_Scale` | `M4_Scale` | excess block variance across scales, destruction scan | Section 3.3 and Appendix A, Figure 2 |
| `W5_Modes_R2` | `M5_Modes_R2` | cosine modes, the precursor R2 | Sections 3.4 and 3.5, Appendix C, Figure 3, Table 5 |
| `W6_Variables` | `M6_Variables` | the model (q, R2), its comparison with the network on raw images, the network and R2 on thinned images | Section 3.6 and Appendix B, Table 6, Figure 4 |

## Notes

- The test set is used only for the final evaluation. The thinning level, the scale, the pair of modes, the alarm
  thresholds and the model fits come from the training set, and the stopping epoch of the network from
  a validation part split off from it.

## Data and weights

- `data_repository/grids/spatial_temp_diag_{train,test}_pool_{32,64,128}.npz`: the streams and the collected
  examples (notebook 1).
- `data_repository/models/models_L{32,64,128}/`: the network weights (three variants × five seeds) and the
  training log `training_history.json`.
- `data_repository/figures/`: the figures saved by notebooks 2 to 6, each as PNG and PDF.

## Reproducibility

The simulation, the thinning, the bootstrap and the permutation tests use fixed seeds, so notebooks 2 to 6
reproduce the numbers and figures digit for digit, given the stored network weights. The weights were trained
without the deterministic mode, so by default notebook 3 reads them and does not train. `TRAIN = True`
trains the networks again. The weights then come out different, and the numbers that depend on the network
change in the later decimal places.

## Installation

The code needs Python 3.11. All commands are run from the repository root, the folder
`SOC_Predictability_Manna/` that contains `requirements.txt` and this README.

1. Open a terminal in the repository root and create the virtual environment there. The first command creates
   the folder `.venv` next to `requirements.txt`, and the second one activates it.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the packages.

   ```bash
   pip install -r requirements.txt
   ```

   The file pins PyTorch built for CUDA 12.8 and points pip to the index where this build is published.
   The build also runs on a computer without a GPU, only more slowly.

3. Open the notebooks in one of two ways. The first is JupyterLab, started in the directory with the notebooks.

   ```bash
   pip install jupyterlab
   cd data_analysis
   jupyter lab
   ```

   Alternatively, the second is VS Code with the Python and Jupyter extensions. Install the kernel package in the
   environment.

   ```bash
   pip install ipykernel
   ```

   Then open the repository root as a folder in VS Code, open a notebook from `data_analysis/`, click
   "Select Kernel" in the top right corner and choose the Python interpreter from `.venv`.

4. Optionally check that PyTorch sees the GPU. The command prints `True` when it does.

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

Notebooks 2 to 6 read the data from `data_repository/grids/` (see Data and weights) and save the figures to
`data_repository/figures/`, so both directories must exist before they are run.
