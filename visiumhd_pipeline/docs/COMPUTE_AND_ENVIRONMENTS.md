# Compute and environment strategy

## What v1 already did, and what v2 adds

V1 had separate CPU/GPU stages and CPU Proseg parallel controls. Its delivered scVI function used one GPU, and clustering used CPU Scanpy. It did not provide a unified per-sample CSV, standalone cell-only start, script-launched scVI DDP, or RAPIDS/Dask Leiden backend.

V2 adds isolated worker launchers, fixed resource budgets, per-row scratch routing and optional current multi-GPU APIs. Implemented support is not the same as a benchmarked speedup; GPU/DDP/RAPIDS and real Proseg runs have not been executed in the delivery environment.

## g5.24xlarge working model

AWS documents four A10G GPUs with 24 GB each, 96 vCPUs and 384 GiB system memory. Those GPUs do not form one 96-GB CUDA device. Also distinguish CPU threads/vCPUs from physical cores.

| Stage | v2 execution | What it does not claim |
|---|---|---|
| Import/QC/assembly | CPU processes; sample workers configurable, I/O threads bounded | Every preprocessing function is not rewritten into a distributed Dask expression workflow |
| StarDist | One fresh TensorFlow process per active sample, one assigned GPU; tiled/big-image inference | A single whole slide is not distributed across four GPUs |
| Proseg | CPU binary with explicit nthreads; independent samples may run concurrently | No GPU Proseg acceleration |
| scVI | Single GPU or torchrun/Lightning DDP with one process per assigned GPU | No pooled VRAM; no fully streamed training; no automatic biological validation |
| Neighbors | CPU Scanpy, single-GPU RAPIDS, or explicit RAPIDS mg_ivfflat/mg_ivfpq | All installed versions do not support every algorithm |
| Leiden | CPU, single GPU, or RAPIDS use_dask=True multi-GPU | Dask is not guaranteed faster; documented API is experimental |
| UMAP fitting | CPU or single-GPU RAPIDS; one fit shared by ten resolutions | Multi-GPU UMAP training is not implemented |

## CPU and host-RAM budgets

The g5 example reserves eight CPU threads and 40 GiB free RAM. Proseg requests four jobs × 20 threads = 80, within the 88-thread budget. StarDist requests four jobs × eight threads. CPU data-processing jobs start with one worker × eight threads because table/image working memory can be large.

The actual worker count is the minimum of requested workers, CPU budget, available assigned GPUs where relevant, and available RAM divided by the **supplied measured per-worker RAM estimate**. An unknown estimate forces a serial pilot; it is not interpreted as zero memory. Existing workloads reduce available RAM. Leave additional headroom for the notebook, OS cache and other processes.

Use `worker_*.resources.json` to inspect elapsed time and sampled summed process-tree RSS. Use a conservative estimate from large representative samples; the code does not automatically generalize one small pilot to every slide. GPU peak VRAM is not measured by this host-RAM sampler.

Thread environment variables are set before workers start. A slot token assigns one GPU exclusively to each StarDist process within a launch. Different phases are barriers, not simultaneous TensorFlow/scVI/RAPIDS jobs. The launcher cannot prevent an unrelated external process from using a GPU; inspect nvidia-smi and do not run multiple GPU phases/notebooks concurrently on the same IDs.

## scVI DDP specifics

`execution.scvi.mode="ddp"`, `devices=4` launches a script with `torch.distributed.run --standalone --nproc_per_node=4 --module vhd.compute.scvi_worker`. This avoids notebook CUDA-fork behavior. The configured CUDA_VISIBLE_DEVICES limits the job to your assigned physical devices.

Each rank reads a copy of the sparse shared-HVG matrix. Host memory is estimated for the number of ranks, not just one AnnData object. Each GPU holds a model replica. This implementation is data-parallel, not model-sharded. `scvi.batch_size` is per rank; nominal effective batch size grows with world size. Changing world size can change optimization behavior and is not guaranteed to reproduce a single-GPU model exactly.

The current scVI multi-GPU guide states that early stopping is unsupported. This implementation uses fixed epochs, train_size=1.0, validation_size=0, early_stopping=False for DDP. **It has no held-out validation curve in that mode.** The ordinary single-GPU route retains the v1 held-out split and early stopping. Compare biological diagnostics and train trajectories rather than equating a fixed-epoch DDP run with a validated hyperparameter choice.

Rank zero saves the model and training history. A separate single-GPU process performs latent extraction after every training rank has exited, avoiding multiple writers and collective-inference problems. The preparation/input and output fingerprints protect against reusing a different model accidentally. Partial checkpoints are not an automatic training-resume implementation; use a deliberate new run after inspecting failures.

## RAPIDS and Dask specifics

`clustering.backend` is `scanpy`, `rapids`, or `rapids_dask`. The standard g5 profile uses `rapids`; the opt-in profile sets `rapids_dask` plus `mg_ivfflat` neighbors. Settings expose the neighbor algorithm and options explicitly. Random seed parameter differences (`rng` versus `random_state`) are checked at runtime. Lack of `use_dask` support fails with an explanation instead of silently falling back to CPU.

The documented Leiden API recommends single GPU below ten million cells. Benchmark it first for your sample sizes. Dask adds scheduling, transfer, communication and memory overhead; 24-GB A10G devices are not the same hardware as published eight-H100 demonstrations.

A LocalCUDACluster is created for Leiden with one worker process per assigned GPU, two CPU threads per worker by default, per-worker host/device limits, and `local_directory=<integration scratch>/runtime/dask_spill`. Dask worker reservations are closed before single-GPU UMAP fitting. The implementation does not enable UCX/NVLink assumptions or a giant default RMM pool on A10G.

Dask spill applies to data managed by the Dask/CUDA stack. A cuGraph or cuML kernel can still require a graph/intermediate to fit in device memory. Configuring a spill directory is not a guarantee of out-of-core behavior for all steps. The graph and scVI latent coordinates are handled by RAPIDS; the full all-gene expression matrix is not sent to GPU for clustering. HVG preparation and count import remain the CPU/sample-at-a-time implementation, not the separate RAPIDS Dask preprocessing tutorial.

Exactly one UMAP is computed. Leiden 0.1, 0.2, ..., 1.0 labels are written as leiden_0p1 ... leiden_1p0; cluster palettes, size tables, sample fractions and 300-DPI figures follow. Numeric labels at different resolutions do not inherently represent the same biological population. GPU/CPU implementations may produce different partitions even with matching seeds.

## Environments

Use explicit `python_spatial`, `python_stardist`, `python_scvi`, `python_rapids` executable paths in settings. The notebook can remain in a small controller environment. Keep the proven legacy spatial import environment intact; use new/cloned environments for this workflow. Install GPU packages using vendor-supported builds matching the driver/CUDA stack; the env files are dependency guidance, **not verified lockfiles**.

OpenSlide needs its native library as well as openslide-python. RGB TIFF fallback and SpatialData Zarr compatibility are checked with target-environment smoke tests. Current API presence does not prove dependency-solver compatibility. Capture `environment_report()` output and a real lockfile after tests pass on your instance.

No background service, scheduler daemon or remote EC2 execution is created. Notebook calls synchronously launch local subprocesses. Logs, configuration snapshots, resource reports and errors are saved under configured approved scratch.
