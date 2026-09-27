# Control layer

manifest.py defines the fixed 41-column CSV, entry-stage plans, per-sample settings and storage allowlists. runner.py dispatches only applicable actions and handles separate final publication. storage.py handles explicit local/S3 staging, integrity checks and immutable completion markers. No biological input is modified.
