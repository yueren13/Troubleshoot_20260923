# Execution backends

launch.py owns resource budgets, subprocess isolation, GPU slots and logs. worker.py runs sample actions; integration_worker.py runs preparation/clustering/diagnostics/writeback; scvi_worker.py implements script/DDP training and separate inference; clustering.py selects CPU, RAPIDS or optional Dask Leiden; probe.py reports environment/hardware. GPU paths need target-environment testing.
