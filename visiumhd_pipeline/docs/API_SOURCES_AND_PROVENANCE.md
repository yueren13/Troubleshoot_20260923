# API sources and implementation provenance

Checked against official/primary documentation on September 24, 2026. These are design references, not evidence that the corresponding hardware paths were executed here.

- AWS accelerated instance specifications: https://aws.amazon.com/ec2/instance-types/accelerated-computing/
- scVI multi-GPU use case and early-stopping limitation: https://docs.scvi-tools.org/en/stable/user_guide/use_case/multi_gpu_training.html
- scVI model API (train/load/latent): https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html
- PyTorch DDP: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- RAPIDS-singlecell Leiden, including experimental use_dask: https://rapids-singlecell.readthedocs.io/en/stable/api/generated/rapids_singlecell.tl.leiden.html
- RAPIDS-singlecell neighbor algorithms, including mg_ivfflat/mg_ivfpq: https://rapids-singlecell.readthedocs.io/en/stable/api/generated/rapids_singlecell.pp.neighbors.html
- RAPIDS-singlecell UMAP: https://rapids-singlecell.readthedocs.io/en/stable/api/generated/rapids_singlecell.tl.umap.html
- RAPIDS multi-GPU preprocessing/neighbor example (not the exact full workflow implemented here): https://rapids-singlecell.readthedocs.io/en/stable/notebooks/07_multi_gpu.html
- Dask-CUDA LocalCUDACluster configuration: https://docs.rapids.ai/api/dask-cuda/stable/_modules/dask_cuda/local_cuda_cluster/
- Proseg documentation/source and CPU CLI: https://github.com/dcjones/proseg
- SpatialData models (tables, shapes/point radii): https://spatialdata.scverse.org/en/stable/api/models.html
- AnnData element I/O: https://anndata.readthedocs.io/en/stable/generated/anndata.io.read_elem.html
- AWS S3 object consistency: https://aws.amazon.com/s3/consistency/
- Requested microViz palette: https://david-barnett.github.io/microViz/reference/distinct_palette.html

## Relationship to the previous delivery

The mounted `VisiumHD_Reusable_Pipeline_20260923.zip` was the code source for v2. Scientific engines and the palette were retained, while CSV control, standalone adapters, storage publication and compute launchers were added. Historical v1 logs are retained in the original delivery archive, not republished here; they are not v2 scientific-runtime validation.

`docs/SOURCE_AUDIT.md` is the historical v1 review of the original repository. It states that the large Step01–03 notebook source and two helper modules were not fully available to that review. This revision does not overturn that limitation: fresh H&E normalization/QC remains the explicit baseline supplied in v1, while imported existing qc_class annotations preserve accepted QC decisions. Original user repository files were not modified.

Current implementation limitations include mandatory explicit sample-level input rows; local scratch staging rather than direct remote HDF5; selected counts rather than copying all corrected layers into a count-first product; no boundaries reconstructed from centroids; no recovery of omitted raw bins or molecule positions; non-streamed scVI host input; and unexecuted target GPU/DDP/S3 paths. See the validation summary.
