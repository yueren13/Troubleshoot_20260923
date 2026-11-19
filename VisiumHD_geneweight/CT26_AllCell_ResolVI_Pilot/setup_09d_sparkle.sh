#!/usr/bin/env bash
# Install the CPU SPARKLE notebook kernel from a JupyterLab terminal:
#   bash setup_09d_sparkle.sh
# To reuse direct dependency versions from an existing uv environment:
#   bash setup_09d_sparkle.sh --from-python /home/domino/reny28/Projects/Visium_projects/TBIO-8057/env/gpu_env/.venv/bin/python
# To use a particular Python for the pinned environment: CT26_PYTHON=/path/to/python3 bash setup_09d_sparkle.sh
set -euo pipefail

fail() { echo "$*" >&2; exit 1; }

spatial=false
if (( $# )); then
    [[ "$#" == 2 && "$1" == --from-python ]] || fail "Usage: bash setup_09d_sparkle.sh [--from-python /path/to/.venv/bin/python]"
    [[ -f "$2" && -x "$2" ]] || fail "Cannot execute source Python at $2. Check the mounted uv project path."
    spatial=true
fi
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

# An environment built from the spatial project gets its own kernel and destination.
if "$spatial"; then
    python_bin="$2"
    env_dir="${CT26_SPARKLE_ENV:-${XDG_DATA_HOME:-$HOME/.local/share}/ct26/envs/sparkle-0.1.4-spatial}"
    kernel_name="ct26-sparkle-0-1-4-spatial"
    kernel_display="CT26 SPARKLE 0.1.4 (spatial CPU)"
    marker="$env_dir/.ct26-09d-spatial-source"
    unset UV_TARGET UV_PREFIX PIP_TARGET PIP_PREFIX PIP_USER
    fallback="For the tested pins, run: env -u CT26_SPARKLE_ENV bash setup_09d_sparkle.sh"
    # sys.prefix identifies the environment. The python executables in two separate
    # uv environments commonly resolve to the very same underlying binary.
    source_prefix="$("$python_bin" -I -B - "$env_dir" <<'PYCODE'
from pathlib import Path
import sys
source = Path(sys.prefix).resolve()
child = Path(sys.argv[1]).expanduser().resolve()
if child == source or source in child.parents or child in source.parents:
    sys.exit(f"Choose a separate sibling destination. Source is {source}, destination is {child}.")
print(source)
PYCODE
)"
    if [[ -e "$env_dir" || -L "$env_dir" ]]; then
        [[ -f "$marker" && -f "$env_dir/pyvenv.cfg" && "$(cat -- "$marker")" == "$source_prefix" ]] ||
            fail "Destination $env_dir already exists and was not created by 09D for $source_prefix. Set CT26_SPARKLE_ENV to a new directory."
    fi
    echo "Spatial source: $source_prefix ($("$python_bin" -I -B -V))"
    echo "Independent 09D environment: $env_dir"
    # Only these notebook dependencies are requested from the source. Normal
    # dependency resolution in the child supplies their transitive requirements.
    requirements="$("$python_bin" -I -B - "${packages[@]}" <<'PYCODE'
from importlib import metadata
import sys
for pin in sys.argv[1:]:
    name, tested = pin.split("==")
    version, origin = tested, "09D pin"
    if name != "stambient":
        try:
            version = metadata.version(name)
            origin = "spatial source"
        except metadata.PackageNotFoundError:
            pass
    if not version or any(char.isspace() for char in version):
        sys.exit(f"Invalid installed version metadata for {name}")
    requirement = f"{name}=={version}"
    print(f"  {requirement} ({origin})", file=sys.stderr)
    print(requirement)
PYCODE
)"
    mapfile -t packages <<< "$requirements"
fi

if [[ ! -f "$env_dir/pyvenv.cfg" ]]; then
    if ! "$spatial"; then
        "$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Select Python 3.11 or later with CT26_PYTHON."'
    fi
    [[ ! -e "$env_dir" ]] || fail "A non-venv directory already exists at $env_dir. Set CT26_SPARKLE_ENV to a new directory."
    mkdir -p "$(dirname "$env_dir")"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$python_bin" "$env_dir"
    else
        "$python_bin" -m venv "$env_dir"
    fi
    if "$spatial"; then
        printf '%s\n' "$source_prefix" > "$marker"
    fi
fi
env_python="$env_dir/bin/python"
if "$spatial"; then
    # Verify the destination before allowing the installer to touch it, including on reruns.
    fingerprint='import json, os, sys; print(json.dumps([sys.implementation.name, sys.version, os.path.realpath(sys._base_executable)]))'
    source_build="$("$python_bin" -I -B -c "$fingerprint")"
    child_build="$("$env_python" -I -B -c "$fingerprint")"
    [[ "$source_build" == "$child_build" ]] ||
        fail "Source and destination Python builds differ. Set CT26_SPARKLE_ENV to a new directory."
    "$env_python" -I -B - "$env_dir" "$source_prefix" <<'PYCODE'
from pathlib import Path
import sys, sysconfig
child = Path(sys.argv[1]).expanduser().resolve()
source = Path(sys.argv[2]).resolve()
if child != Path(sys.prefix).resolve() or child == Path(sys.base_prefix).resolve():
    sys.exit(f"Destination Python did not start in its own virtual environment at {child}.")
for key in ("purelib", "platlib", "scripts"):
    target = Path(sysconfig.get_path(key)).resolve()
    if child not in target.parents or target == source or source in target.parents:
        sys.exit(f"Unexpected destination for {key}: {target}")
PYCODE
fi

if command -v uv >/dev/null 2>&1; then
    installer=(uv pip install --python "$env_python")
else
    installer=("$env_python" -m pip install)
fi
if ! "${installer[@]}" "${packages[@]}"; then
    if "$spatial"; then echo "The source versions could not be installed together in the child. $fallback" >&2; fi
    exit 1
fi

if "$spatial"; then
    if ! "$env_python" -I - <<'PYCODE'
from importlib import import_module
for module in ("numpy", "scipy", "pandas", "anndata", "h5py", "pyarrow.parquet",
               "zarr", "geopandas", "matplotlib.pyplot", "IPython", "ipykernel", "stambient"):
    import_module(module)
from anndata.io import read_elem
from shapely import STRtree, from_wkb, points
PYCODE
    then
        fail "The child could not import the 09D scientific stack. $fallback"
    fi
fi

# Exercise the actual fitter on a 2-µm grid with signal inside and outside four masks.
if ! "$env_python" - <<'PY'
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
then
    if "$spatial"; then echo "SPARKLE's CPU fit failed in the child environment. $fallback" >&2; fi
    exit 1
fi
"$env_python" -m ipykernel install --user --name "$kernel_name" --display-name "$kernel_display"
echo "In JupyterLab, choose the kernel: $kernel_display"
