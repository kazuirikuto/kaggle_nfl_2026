# Nfl Big Data Bowl 2026 Prediction

- Kaggle: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
- Created: 2025-11-18 13:22 UTC

## Folders

- `data/raw` – original Kaggle downloads
- `data/processed` – cleaned datasets ready for modeling
- `notebooks` – exploratory & submission notebooks
- `src` – standalone scripts or reusable modules
- `models` – serialized model artifacts
- `experiments` – logs or reports from experiments
- `submissions` – CSV files submitted to Kaggle

## Notes

- Keep the submission notebook deterministic; Kaggle runs them in a clean
  environment.
- Track experiment ideas, validation results, and lessons learned here.

## Environment Setup for `notebooks/nfl-deep-gnn.ipynb`

The notebook imports `tensorflow`, `polars`, and the bundled
`kaggle_evaluation` package, so a dedicated conda environment prevents
version conflicts.

```bash
# 1) Create the Python 3.10 environment (do this once)
CONDA_SOLVER=classic CONDA_NO_PLUGINS=true conda create -y -n nfl python=3.10

# 2) Install the CPU packages pulled from conda-forge
CONDA_SOLVER=classic CONDA_NO_PLUGINS=true \
  conda install -y -n nfl -c conda-forge pandas numpy scikit-learn scipy polars

# 3) Download the offline TensorFlow wheels (PyPI is blocked here)
mkdir -p third_party
kaggle datasets download -d lonnieqin/tensorflow-2-15 \
  -p third_party/tensorflow-2-15 --unzip

# 4) Install TensorFlow 2.15.0 and its wheels from the dataset
CONDA_NO_PLUGINS=true \
  conda run -n nfl python -m pip install \
  --no-index --find-links third_party/tensorflow-2-15/tensorflow tensorflow==2.15.0

# 5) (Optional) reinstall numpy/pandas/etc. if pip downgraded them
CONDA_SOLVER=classic CONDA_NO_PLUGINS=true \
  conda install -y -n nfl pandas numpy scikit-learn scipy
```

Before running the notebook, expose the competition's `kaggle_evaluation`
module and activate the environment:

```bash
export PYTHONPATH="/home/rikuto/kaggle_note/competitions/nfl-big-data-bowl-2026-prediction/data/raw:${PYTHONPATH:-}"
conda activate nfl
```

You can now open `competitions/nfl-big-data-bowl-2026-prediction/notebooks/nfl-deep-gnn.ipynb`
in Jupyter Lab/VS Code and run the training/inference cells locally. The
`DATA_DIR` constant inside the notebook already points to the downloaded
`data/raw` folder. If you see `joblib` warnings about multiprocessing in
this environment, set `JOBLIB_MULTIPROCESSING=0` when launching Jupyter to
force serial execution.

## GPU Execution via NVIDIA TensorFlow Docker

If TensorFlow inside the conda environment cannot see the RTX 4070, run the
notebook inside NVIDIA's official TensorFlow container (CUDA/cuDNN and
drivers are pre-wired there).

1. Install Docker Desktop (with WSL integration) and the NVIDIA Container
   Toolkit so `docker run --gpus all ...` succeeds.
2. From the repo root run `bash scripts/run_tf_gpu_container.sh`. The script:
   - pulls `nvcr.io/nvidia/tensorflow:24.08-tf2-py3`
   - mounts this repo to `/workspace` inside the container
   - exports `PYTHONPATH` so the bundled `kaggle_evaluation` module resolves.
3. Inside the container install the few Python packages that are not shipped
   with the base image (run once):
   ```bash
   pip install pandas numpy scikit-learn scipy polars==0.20.26
   ```
   You can also install additional Kaggle utilities if needed.
4. Verify TensorFlow sees the GPU before opening the notebook:
   ```bash
   python - <<'PY'
   import tensorflow as tf
   print("TF", tf.__version__)
   print(tf.config.list_physical_devices("GPU"))
   PY
   ```
   You should see at least one `/physical_device:GPU:0`.

Launch Jupyter Lab or VS Code inside the container, or run the notebook via
`jupyter lab --ip=0.0.0.0 --port=8888`. Because everything lives under
`/workspace`, the notebook still references the same data and artifacts.
