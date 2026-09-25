# RAPIDS environment

Create a SEPARATE environment following the official RAPIDS / RAPIDS-singlecell installation instructions for your installed NVIDIA driver and supported CUDA package build. Do not pip-install a random CUDA variant into the established TensorFlow or spatial environment.

Required capabilities/packages include rapids-singlecell, compatible cuML/cuGraph/CuPy, dask-cuda, distributed, anndata, scanpy, pandas, scipy, pyarrow, h5py, matplotlib, pillow and psutil. The Dask Leiden route requires the `use_dask` parameter in `rapids_singlecell.tl.leiden`. The multi-GPU neighbor route requires an installed version supporting the selected `mg_` algorithm. API signatures are checked at runtime, but dependency compatibility and GPU execution still require target testing.

This is NOT a solver lock, and no successful RAPIDS environment installation is claimed for the delivery environment. Set execution.python_rapids to the full path of the validated environment interpreter. Preserve a resolved lockfile after a small run passes.
