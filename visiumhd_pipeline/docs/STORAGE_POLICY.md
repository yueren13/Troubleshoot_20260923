# Storage policy and publication

## Paths are part of the scientific workflow, not an afterthought

The CSV separates object, plot and table/log destinations because those categories may have different performance, retention and governance requirements. The code treats `s3://...` as remote object storage and local mounted paths as POSIX storage. It does not infer whether a mount is EBS, instance NVMe, NFS or another service from its name. The user's term “EB2” is not hard-coded as a storage type.

Approved scratch must be authorized to hold **the data themselves**: expression matrices, images, masks, trained models, output products and remote source staging copies. A path approved only for non-sensitive logs is not sufficient. Do not enable remote staging when policy prohibits copying those source data locally. There is no direct-S3 HDF5 processing fallback.

## Active work versus final publication

All compute writes go to the row's local tmp_dir subtree; large active stores are not repeatedly edited through an S3 mount. Framework cache and temp paths are also set before the scientific process starts. scVI trainer output is routed to the integration runtime directory. A Proseg worker uses local mask, input and output paths.

For local source paths, inputs are read in place, without copying whole directories unnecessarily. For an S3 source, explicit opt-in stages the file/prefix to approved scratch. A source listing includes file sizes, ETags, version IDs and modification tokens; files are size-checked after transfer. ETags are not treated as MD5 checksums. Remote staging validation is not a full SHA256 verification of remote source content.

For a standalone AnnData cell source, raw Space Ranger data are not required. Leave unrelated raw/image columns blank. Supplying an optional Space Ranger directory allows calibration recovery and may stage that supplied directory; use source_mpp explicitly to avoid such staging when appropriate.

Local source changes are tracked using file sizes/mtimes and recursive directory inventory fingerprints. These are not cryptographic proof of unchanged input bytes. Immutable source data and versioned manifests remain preferable. Publication computes actual SHA256 hashes of payload files.

## Policy settings

- `enforce_allowlists=true` requires declared row/integration roots to be under their approved category roots. Prefix checks respect directory boundaries; `/approved_bad` is not inside `/approved`.
- `approved_tmp_roots`, `approved_object_roots`, `approved_plot_roots`, `approved_data_roots` must be filled with real approved roots. Do not widen them merely to bypass an error.
- `allow_remote_staging=false` and `allow_remote_publication=false` are independent default denials.
- `min_free_tmp_gib` and `max_remote_stage_gib` guard staging space, not the total peak disk use of every later scientific algorithm. Budget room for staging H5AD, SpatialData, images, Proseg outputs and publication copies.
- `verify_remote_sha256=false` means remote published objects are verified by size after transfer; their source SHA256 is saved in the publication manifest. Setting true reads remote content back and hashes it, adding I/O and possible cost. Local publications always receive SHA256 readback.

S3 authentication uses the standard fsspec/s3fs environment/IAM credential chain. No keys belong in a notebook, CSV or settings JSON. Bucket policies, encryption/KMS, access control, regional residency, lifecycle and retention configuration remain the organization's responsibility. This code is **not a compliance certification or security boundary replacing IAM**.

## Commit protocol

The destination is versioned by dataset/sample/run_id/category. All payload files are copied first, then `_MANIFEST.json`, then `_SUCCESS.json`. S3 provides per-object consistency, not a multi-object directory transaction. Downstream readers must require the last marker for that category. Categories are committed independently; the local PUBLICATION_RECEIPT lists all successful destinations. There is no single atomic commit spanning object, plot and data roots.

A matching existing publication is validated and reused. A different or incomplete publication is not overwritten or deleted. A failure can leave a partial prefix; inspect it under local governance, then choose a new run_id or deliberately remediate. The pipeline never treats the presence of `sample.zarr/` alone as success.

Complete scVI annotation writeback **before** final per-sample publication. Changing an already-published product requires a new run_id. Re-executing import or diagnostics can change logs and metadata even when expression outputs are reused; immutable publication protects the previous record rather than silently replacing it.

## Scratch lifecycle

The code never automatically deletes source or temporary data. Verify successful publication and a downstream read before cleanup. Local instance-store scratch should be treated as non-durable; durability comes from a verified publication to an approved durable destination, not a successful computation alone. Resume receipts, masks and models reside under tmp_dir until published or separately retained by policy.

For the four-dataset migration, pilot one row first and verify its three publication destinations. Then enable only the additional samples appropriate for that run. Never point an output root at a source directory.
