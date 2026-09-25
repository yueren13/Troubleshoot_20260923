
#========================================== Custom functions ============================================
import os
import os.path
import re
import math
import gc
import glob
import sys
import numpy as np
import scipy.sparse as sp
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import xarray as xr
import anndata as ad
import shapely
import xml.etree.ElementTree as ET
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from typing import Iterable, Mapping, Optional

try:
    # shapely >= 2
    from shapely.validation import make_valid
except Exception:
    # fallback for shapely 1.x
    make_valid = lambda g: g.buffer(0)

import spatialdata as sd 
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation, set_transformation


## to remove given string from a list of strings -------------------------------------------------------
def remove_char(lst, char):
    return [re.sub(char, '', s) for s in lst]
    
## function to edit the table object in the orginal zarr file -------------------------------------------
# importantly, the pattern below is used to avoid the risk of losing the original table is this cell is killed while rewriting the original one
# this approach works for this notebook, but it other contexts, like with network storages or multithread environment, it may not prevent data
# corruption. As a user, it is important to take this factors into consideration and implement a strategy to minimize the risk of data loss.

def persist_changes(
    sdata: sd.SpatialData,
    update_elements=None, #Iterable[str]
    new_elements=None, #Iterable[str]
    update_transformations: bool = True,
    consolidate_metadata: bool = True,
) -> None:
    """
    Write only the selected elements of a SpatialData object back to its
    associated Zarr store. Does NOT touch other elements (e.g., large images).

    Parameters
    ----------
    sdata : SpatialData
        The object you’ve modified in-memory (must already be associated to a Zarr store).
    elements : iterable of str
        Names of elements to persist, e.g. ["square_002um", "annotation_labels_down_3class"].
        Works for tables, labels, shapes, points, images.
    update_transformations : bool
        If True, also update the transformations/metadata on-disk.
        Useful if you added/changed transforms for shapes/labels.
    consolidate_metadata : bool
        If True, re-write consolidated Zarr metadata for faster reads.

    Notes
    -----
    - This uses SpatialData’s incremental I/O: `write_element`, `write_transformations`,
      `write_metadata`, `write_consolidated_metadata`. See the docs.  # noqa
    """
    # Sanity: check this object is already tied to a Zarr store
    try:
        _ = sdata.elements_paths_on_disk()  # raises if no associated store
    except Exception as e:
        raise RuntimeError(
            "This SpatialData object is not associated with a Zarr store. "
            "Write it once with `sdata.write(<existing_store_path>)`, "
            "then call persist_changes()."
        ) from e

    # Write only the requested elements
    if update_elements is None:
        report_type0 = "No element will be updated."
        print(report_type0)
    else:
        for name in update_elements:
            if name not in sdata:
                raise KeyError(f"Element '{name}' not found in SpatialData.")
                #generate datatype to test
                #datatype_1 = isinstance(sdata[name], geopandas.GeoDataFrame)#test if shape is on disk
                #datatype_2 = isinstance(sdata[name], ad.AnnData)#test if table is on disk
                #datatype_3 = isinstance(sdata[name], xr.DataArray)#test if image is on disk
                #datatype_4 = isinstance(sdata[name], dd.DataFrame)#test if points is on disk
            report_type1 = "Now updating " + name + " on disk."
            #if datatype_1 or datatype_2 or datatype_3 or datatype_4: #if the data is already on disk
            element_name = "backup_" + name
            print(report_type1)
            # copy the table to a backup table
            sdata[element_name] = sdata[name]
            sdata.write_element(element_name)# <- incremental write for just this element
            # rewrite the original one
            sdata.delete_element_from_disk(name)
            sdata.write_element(name)
            # remove the backup copy
            sdata.delete_element_from_disk(element_name)
            del sdata.tables[element_name]
    #for new data that is not on disk
    if new_elements is None:
        report_type3 = "No new element will be recorededd."
        print(report_type3)
    else:
        for name in new_elements:
            if name not in sdata:
                raise KeyError(f"Element '{name}' not found in SpatialData.")
            report_type2 = "Now creating new data record for " + name + " on disk."
            print(report_type2)
            sdata.write_element(name)
            
    # Optional: update transforms/metadata (lightweight)
    if update_transformations:
        sdata.write_transformations()
        sdata.write_metadata()
        if consolidate_metadata:
            sdata.write_consolidated_metadata()

def replace_table(sdata, table_name = "table"):
    # copy the table to a backup table
    sdata["backup_table"] = sdata[table_name]
    sdata.write_element("backup_table")

    # rewrite the original one
    sdata.delete_element_from_disk(table_name)
    sdata.write_element(table_name)

    # remove the backup copy
    sdata.delete_element_from_disk("backup_table")
    del sdata.tables["backup_table"]
    
##convert dataframe to dictionary
def df_to_dict(df, key_col, val_col):
    result = {}
    for index, row in df.iterrows():
        key = row[key_col]
        value = row[val_col]

        if key in result:
            # If the key already exists, append the value to the list
            if isinstance(result[key], list):
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value

    return result
    
##add a specific phrase to every element in a list
def add_prefix_to_list_elements(my_list, prefix):
  """Adds a prefix string to the beginning of each element in a list.

  Args:
    my_list: The list of strings to modify.
    prefix: The string to add to the beginning of each element.

  Returns:
    A new list with the prefix added to each element.
  """
  return [prefix + element for element in my_list]

## monocle 3 normalization in python ##
def monocle3_log_normalize(
    adata,
    size_factors=None,              # array-like of length n_cells, or name of obs column
    size_factors_obs_key=None,      # alternative: pass the obs column name here
    layer_in=None,                  # None -> use adata.X; otherwise name of input layer with raw counts
    layer_out="monocle3_log10",     # output layer name
    use_area_from_obs=None,         # e.g. "cell_area_um2" to build sf from area
    scale_to_median=True,           # scale s.f. so median(sf)=1 (Monocle-style stability)
    pseudocount=1.0,                # must be 1.0 for sparse to match Monocle behavior
    set_X=True                      # optionally make this the working matrix
):
    # 1) Get counts
    X = adata.layers[layer_in] if layer_in is not None else adata.X
    if X is None:
        raise ValueError("No input matrix found.")
    n_cells = adata.n_obs

    # 2) Build size factors
    if use_area_from_obs is not None:
        sf = np.asarray(adata.obs[use_area_from_obs], dtype=float)
    elif size_factors is not None:
        sf = np.asarray(size_factors, dtype=float)
    elif size_factors_obs_key is not None:
        sf = np.asarray(adata.obs[size_factors_obs_key], dtype=float)
    else:
        # Fallback: Monocle’s default “mean-geometric-mean-total”
        # sf = total_counts / exp(mean(log(total_counts)))
        totals = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        gmean = np.exp(np.mean(np.log(totals[totals > 0])))
        sf = totals / gmean

    # Clean + optional scaling
    sf = np.where(np.isfinite(sf) & (sf > 0), sf, 1.0)
    if scale_to_median:
        sf = sf / np.median(sf)

    # 3) Divide counts by size factors (row-wise in AnnData: cells x genes)
    if sp.issparse(X):
        Xnorm = X.tocsr(copy=True)
        Xnorm = Xnorm.multiply(1.0 / sf[:, None])
        # 4) Log10( … + 1 ) in place on nonzeros (zeros stay zero, which equals log10(1))
        if pseudocount != 1.0:
            raise ValueError("Monocle’s sparse path requires pseudocount=1.")
        Xnorm.data = np.log10(Xnorm.data + 1.0)
    else:
        Xf = X.astype(np.float64, copy=True)
        Xnorm = np.log10((Xf / sf[:, None]) + pseudocount)

    # 5) Store
    adata.layers[layer_out] = Xnorm
    if set_X:
        adata.X = adata.layers[layer_out]
    # Also stash the size factors for reference
    adata.obs["monocle3_size_factor"] = sf
    return adata
#
def read_imagescope_annotation_xml(
    path,
    um_per_px_x=None,
    um_per_px_y=None,
    dissolve_by_annotation=True
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    path : str
        Path to .annotation XML.
    um_per_px_x, um_per_px_y : float or None
        If provided, convert pixel coordinates to micrometers.
    dissolve_by_annotation : bool
        If True, union positive regions per Annotation Name, then subtract
        union of NegativeROA regions to produce a single geometry per annotation.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for annot in root.findall(".//Annotation"):
        name = annot.get("Name", "unnamed")
        visible = annot.get("Visible", "True")
        color = annot.get("LineColor")  # integer RGB; keep for metadata

        pos_polys, neg_polys = [], []
        for region in annot.findall("./Regions/Region"):
            neg = str(region.get("NegativeROA", "0")).lower() in {"1", "true", "t", "yes"}
            vertices = region.find("./Vertices")
            coords = []
            for v in vertices.findall("./V"):
                x = float(v.get("X")); y = float(v.get("Y"))
                if (um_per_px_x is not None) and (um_per_px_y is not None):
                    x *= um_per_px_x
                    y *= um_per_px_y
                coords.append((x, y))
            if len(coords) >= 3:
                # close polygon if needed
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)  # fix self-intersections
                (neg_polys if neg else pos_polys).append(poly)

        if dissolve_by_annotation:
            geom = unary_union(pos_polys) if pos_polys else None
            if (geom is not None) and neg_polys:
                geom = geom.difference(unary_union(neg_polys))
            if (geom is not None) and (not geom.is_empty):
                rows.append(dict(name=name, visible=visible, linecolor=color, geometry=geom))
        else:
            # keep each region separately
            for i, p in enumerate(pos_polys):
                rows.append(dict(name=name, visible=visible, linecolor=color, neg=False, region=i, geometry=p))
            for i, p in enumerate(neg_polys):
                rows.append(dict(name=name, visible=visible, linecolor=color, neg=True, region=i, geometry=p))

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    # Force 2D (defensive)
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g.is_empty else Polygon(np.array(g.exterior.coords)[:, :2]) if g.geom_type=="Polygon" else g)
    return gdf
#

def _clean_polygonal(g):
    """Return a polygonal geometry (Polygon or MultiPolygon) or None."""
    if g is None or g.is_empty:
        return None
    g = make_valid(g)             # fix self-intersections / bow-ties
    if g.is_empty:
        return None
    t = g.geom_type
    if t == "Polygon":
        # drop degenerate holes with <4 vertices
        holes = [r.coords[:] for r in g.interiors if len(r.coords) >= 4]
        if len(g.exterior.coords) < 4:
            return None
        return Polygon(g.exterior.coords[:], holes)
    if t == "MultiPolygon":
        parts = []
        for p in g.geoms:
            pc = _clean_polygonal(p)
            if pc is None: 
                continue
            if pc.geom_type == "Polygon":
                parts.append(pc)
            elif pc.geom_type == "MultiPolygon":
                parts.extend(list(pc.geoms))
        if not parts:
            return None
        return MultiPolygon(parts) if len(parts) > 1 else parts[0]
    if t == "GeometryCollection":
        polys = []
        for gg in g.geoms:
            pg = _clean_polygonal(gg)
            if pg is None:
                continue
            if pg.geom_type == "Polygon":
                polys.append(pg)
            elif pg.geom_type == "MultiPolygon":
                polys.extend(list(pg.geoms))
        if not polys:
            return None
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    # Lines/points etc. are not plottable as filled shapes; drop them
    return None

def make_shapes_plot_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_clean_polygonal)
    gdf = gdf[gdf["geometry"].notnull()].reset_index(drop=True)
    # Separate MultiPolygons to Polygons to avoid MultiPolygon patch path
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf
###
flatten = lambda *n: (e for a in n
    for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
###
def assign_annotations_fast(
    sdata,
    annotation_names=["Tissue", "Anthropotic", "Necrosis"],
    annot_shapes_key="annotations_raw",
    table_key="square_002um",
    visium_name="Visium_area",                 # if not present, we fall back to the grid bounds
    target_coordinate_system="global",
    store_as="bool",                           # "bool" or "category"
):
    """
    Speedups:
      • crop to Visium area once
      • union polygons per class
      • use clip=False (membership only)
    """
    # --- 0) grab the annotations GeoDataFrame
    ann_gdf = gpd.GeoDataFrame(sdata.shapes[annot_shapes_key]).copy()

    # --- 1) build the Visium region polygon
    vis_rows = ann_gdf.loc[ann_gdf["name"].astype(str) == visium_name, "geometry"]
    if len(vis_rows):
        vis_poly = unary_union(vis_rows.values)
    else:
        # fallback: tight bbox over the squares shapes element
        grid_key = next(k for k in sdata.shapes.keys() if "square_002um" in k)
        grid_gdf = gpd.GeoDataFrame(sdata.shapes[grid_key])
        minx, miny, maxx, maxy = grid_gdf.total_bounds
        vis_poly = box(minx, miny, maxx, maxy)

    # --- 2) crop once to Visium area (fast membership, no clipping)
    sdata_vis = sd.polygon_query(
        sdata, polygon=vis_poly, target_coordinate_system=target_coordinate_system, clip=False
    )

    # We'll write into the original table; indices are the same
    adata = sdata.tables[table_key]

    # initialize destination columns
    if store_as == "bool":
        for a in annotation_names:
            col = f"is_{a}"
            if col not in adata.obs:
                adata.obs[col] = False
    elif store_as == "category":
        for a in annotation_names:
            if a not in adata.obs:
                adata.obs[a] = pd.Categorical.from_codes(
                    np.zeros(adata.n_obs, dtype=np.int8), categories=["unassigned", a]
                )
    else:
        raise ValueError("store_as must be 'bool' or 'category'.")

    # --- 3) union one polygon per annotation and query on the pre-cropped sdata
    for a in annotation_names:
        polys = ann_gdf.loc[ann_gdf["name"].astype(str) == a, "geometry"]
        if polys.empty:
            continue
        poly_union = unary_union(polys.values)

        # restrict the query area to Visium to avoid scanning off-slide
        roi = poly_union.intersection(vis_poly)
        if roi.is_empty:
            continue

        sub = sd.polygon_query(
            sdata_vis, polygon=roi, target_coordinate_system=target_coordinate_system, clip=False
        )
        idx = sub[table_key].obs.index  # rows of *your* main table that fall in this class

        if store_as == "bool":
            adata.obs.loc[idx, f"is_{a}"] = True
        else:
            adata.obs.loc[idx, a] = a

    # quick sanity counts
    if store_as == "bool":
        print({f"is_{a}": int(adata.obs[f"is_{a}"].sum()) for a in annotation_names})
    else:
        from collections import Counter
        print({a: Counter(adata.obs[a]) for a in annotation_names})

    return sdata  # modified in place

###
#def read_visium_hd_segmented(outs_path: str, sample_id: str) -> TableModel:
#    outs_path = Path(outs_path)
#    bin_size = "segmented_outputs"
#    path_bin = outs_path / bin_size
#    path_bin_spatial = path_bin / VisiumHDKeys.SPATIAL

    # Load gene expression
#    counts_file = "raw_feature_cell_matrix.h5"
#    adata = sc.read_10x_h5(path_bin / counts_file, gex_only=False)
#    adata.var_names_make_unique()

    # Load scalefactors and images
#    with open(path_bin_spatial / VisiumHDKeys.SCALEFACTORS_FILE) as f:
#        scalefactors = json.load(f)

#    hires_img = np.array(Image.open(path_bin_spatial / "tissue_hires_image.png"))
#    lowres_img = np.array(Image.open(path_bin_spatial / "tissue_lowres_image.png"))

#    library_id = "visium_hd_segmentation"
#    adata.uns["spatial"] = {
#        library_id: {
#            "images": {
#                "hires": hires_img,
#                "lowres": lowres_img,
#            },
#            "scalefactors": scalefactors,
#            "metadata": {
#                "source_image_path": "tissue_hires_image.png"
#            }
#        }
#    }

    # Assign region info
#    shapes_name = "cell_segmentation"
#    adata.obs[VisiumHDKeys.INSTANCE_KEY] = np.arange(len(adata))
#    adata.obs[VisiumHDKeys.REGION_KEY] = shapes_name
#    adata.obs[VisiumHDKeys.REGION_KEY] = adata.obs[VisiumHDKeys.REGION_KEY].astype("category")

    # Load cell shapes
#    shapes = spatialdata_io.geojson(outs_path / "segmented_outputs" / "cell_segmentations.geojson", coordinate_system=sample_id)

    # Match shape order to adata
#    centroids = shapes.geometry.centroid
#    adata.obsm["spatial"] = np.vstack([centroids.x.values, centroids.y.values]).T

    # Estimate spot diameter
#    areas = shapes.geometry.area
#    mean_area = np.mean(areas)
#    diameter_pixels = 2 * np.sqrt(mean_area / np.pi)
#    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = diameter_pixels

#    return TableModel.parse(
#        adata=adata,
#        region=shapes_name,
#        region_key=str(VisiumHDKeys.REGION_KEY),
#        instance_key=str(VisiumHDKeys.INSTANCE_KEY),
#    )
