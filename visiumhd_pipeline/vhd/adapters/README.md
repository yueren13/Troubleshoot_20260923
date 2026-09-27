# Input and storage adapters

anndata_input.py reads an explicitly named expression matrix and metadata from standalone AnnData or a selected SpatialData table. bins.py canonicalizes existing bins and reuses accepted QC; cells.py handles optional polygons and cell-only inputs; assemble.py writes bins-only, cells-only or combined SpatialData. Missing cell boundaries are not reconstructed.
