#/home/domino/reny28/Projects/Visium_projects/TBIO-8057/env/gpu_env/.venv/bin/python
#!/usr/bin/env bash
# Install the CPU SPARKLE notebook kernel from a JupyterLab terminal:
#   bash setup_09d_sparkle.sh
# To use a particular Python: CT26_PYTHON=/path/to/python3 bash setup_09d_sparkle.sh
set -euo pipefail

python_bin="${CT26_PYTHON:-python3}"
env_dir="${CT26_SPARKLE_ENV:-${XDG_DATA_HOME:-$HOME/.local/share}/ct26/envs/sparkle-0.1.4}"
kernel_name="ct26-sparkle-0-1-4"
kernel_display="CT26 SPARKLE 0.1.4 (CPU)"
unset PYTHONPATH PYTHONHOME

# Direct dependencies used by the 09D notebook are pinned for repeatable installs.
# The stambient 0.1.4 release matches WangShuai-3/SPARKLE at
# 01138e19c78b9d29cdabd180aa3e248131bc8a1c.
packages=(
    "stambient==0.1.4" "numpy==2.3.5" "scipy==1.16.3"
    "pandas==2.3.3" "anndata==0.12.11" "zarr==3.1.3" "h5py==3.15.1"
    "pyarrow==22.0.0" "shapely==2.1.2" "geopandas==1.1.2"
    "matplotlib==3.10.8" "ipython==9.6.0" "ipykernel==6.30.1"
)

if [[ ! -f "$env_dir/pyvenv.cfg" ]]; then
    "$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Select Python 3.11 or later with CT26_PYTHON."'
    if [[ -e "$env_dir" ]]; then
        echo "A non-venv directory already exists at $env_dir. Set CT26_SPARKLE_ENV to a new directory." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$env_dir")"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$python_bin" "$env_dir"
    else
        "$python_bin" -m venv "$env_dir"
    fi
fi
env_python="$env_dir/bin/python"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$env_python" "${packages[@]}"
else
    "$env_python" -m pip install "${packages[@]}"
fi

# Exercise the actual fitter on a 2-µm grid with signal inside and outside four masks.
"$env_python" - <<'PY'
from importlib.metadata import version
import numpy as np
from scipy import sparse
from stambient import SPARKLE

rng = np.random.default_rng(17)
axis = 2.0 * np.arange(24)
xx, yy = np.meshgrid(axis, axis)
coords = np.column_stack([xx.ravel(), yy.ravel()])
centres = np.array([(10, 10), (34, 10), (10, 34), (34, 34)])
cell_ids = np.array([7, 12, 29, 44])
labels = np.full(len(coords), -1, dtype=int)
distances = np.linalg.norm(coords[:, None] - centres[None, :], axis=2)
for i, cell_id in enumerate(cell_ids):
    labels[distances[:, i] <= 7] = cell_id
rate = 0.03 + 3 * np.exp(-distances.T / 7)
counts = sparse.csr_matrix(rng.poisson(rate))
model = SPARKLE(bin_size=6.0, max_radius=45, lambda_grid=[4, 8, 16],
                n_lambda_genes=4, use_gpu=False, verbose=False)
corrected, diagnostics = model.fit_transform(counts, coords, labels)
raw = np.column_stack([np.asarray(counts[:, labels == i].sum(axis=1)).ravel() for i in cell_ids])
assert corrected.shape == raw.shape == (4, 4)
assert diagnostics["n_empty_bins"] > 0 and diagnostics["n_cells"] == 4
assert diagnostics["lambda_estimated"] in (4, 8, 16)
assert np.isfinite(corrected).all() and np.all(corrected >= 0)
assert np.all(corrected <= raw + 1e-7)
assert diagnostics["n_genes_corrected"] > 0 and (corrected < raw - 1e-7).any()
print(f"SPARKLE {version('stambient')} CPU smoke test passed on {len(coords)} capture squares.")
PY
"$env_python" -m ipykernel install --user --name "$kernel_name" --display-name "$kernel_display"
echo "In JupyterLab, choose the kernel: $kernel_display"
