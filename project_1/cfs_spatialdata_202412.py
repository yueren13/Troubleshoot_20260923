###=======================================================================================================###
###                                                                                                       ###            
###                                    Custom functions for spatialdata                                   ###
###                                                                                                       ###
###=======================================================================================================###

##======================================first load all required packages===================================##

from __future__ import annotations
import os
import os.path
import copy
import re
import math
import gc
import glob
import sys
import uuid
import numpy as np
import scipy.sparse as sp
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import xarray as xr
import anndata as ad
import scanpy as sc
import shapely
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from typing import Iterable, Mapping, Optional

try:
    # shapely >= 2
    from shapely.validation import make_valid
except Exception:
    # fallback for shapely 1.x
    make_valid = lambda g: g.buffer(0)

import matplotlib.pyplot as plt
import spatialdata as sd 
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation, set_transformation, remove_transformation

##=========================================================================================================##

##==================================== general quality of life helpers ====================================##

# -------------- to remove given string from a list of strings -------------- #

def remove_char(lst, char):
    return [re.sub(char, '', s) for s in lst]

##

## convert dataframe to dictionary

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

## --------------  --------------  --------------  --------------  --------------##
    
# -------------- add a specific phrase to every element in a list  -------------- #

def add_prefix_to_list_elements(my_list, prefix):
  """Adds a prefix string to the beginning of each element in a list.

  Args:
    my_list: The list of strings to modify.
    prefix: The string to add to the beginning of each element.

  Returns:
    A new list with the prefix added to each element.
  """
  return [prefix + element for element in my_list]

##=========================================================================================================##

##====================================== spatialdata object helpers =======================================##
    
## function to edit the table object in the orginal zarr file ##
# importantly, the pattern below is used to avoid the risk of losing the original table is this cell is killed while rewriting the original one
# this approach works for this notebook, but it other contexts, like with network storages or multithread environment, it may not prevent data
# corruption. As a user, it is important to take this factors into consideration and implement a strategy to minimize the risk of data loss.

# ============================================================
# Updated persist_changes()
# Designed for SpatialData 0.7.2
# ============================================================


def persist_changes(
    sdata: sd.SpatialData,
    update_elements: Iterable[str] | None = None,
    new_elements: Iterable[str] | None = None,
    transformation_elements: Iterable[str] | None = None,
    update_transformations: bool = True,
    consolidate_metadata: bool = True,
    auto_transform_non_tables: bool = True,
    allow_non_table_rewrite: bool = False,
    keep_backup_on_failure: bool = True,
) -> pd.DataFrame:
    """
    Persist selected changes from an in-memory SpatialData object to its
    associated Zarr store.

    The function distinguishes between two fundamentally different operations:

    1. Full element rewrite
       Used for data that actually changed, such as:
           tables["table"].obs
           tables["table"].obsm

       These elements are supplied through `update_elements`.

    2. Transformation-only update
       Used when image/label/points pixel data did not change, but their
       SpatialData transformations changed.

       These elements are supplied through `transformation_elements`.

    Backward-compatible automatic behavior
    ---------------------------------------
    When `transformation_elements=None` and `auto_transform_non_tables=True`:

        - tables listed in update_elements are fully rewritten;
        - images, labels, points, and shapes listed in update_elements are
          treated as transformation-only updates.

    This means the following old-style call remains supported:

        persist_changes(
            sdata,
            update_elements=[
                "table",
                "1_image",
                "1_labels",
                "1_points",
            ],
        )

    The explicit style is nevertheless recommended:

        persist_changes(
            sdata,
            update_elements=["table"],
            transformation_elements=[
                "1_image",
                "1_labels",
                "1_points",
            ],
        )

    Parameters
    ----------
    sdata
        Backed SpatialData object already associated with a Zarr store.

    update_elements
        Existing elements whose underlying data need to be rewritten.

        By default, only tables are permitted to undergo a full rewrite.
        Non-table elements are routed to transformation-only persistence when
        `auto_transform_non_tables=True`.

    new_elements
        Newly created elements that are not already recorded on disk.

    transformation_elements
        Existing elements whose transformation metadata should be persisted
        without rewriting their underlying image/label/point/shape data.

    update_transformations
        Whether transformation metadata should be written.

    consolidate_metadata
        Whether consolidated Zarr metadata should be regenerated at the end.

    auto_transform_non_tables
        When transformation_elements is not explicitly supplied, automatically
        treat non-table update_elements as transformation-only updates.

    allow_non_table_rewrite
        Permit a full rewrite of images, labels, points, or shapes.

        This defaults to False because raster rewrites can be large and, in the
        current environment, may encounter Dask/OME-Zarr chunk compatibility
        problems.

    keep_backup_on_failure
        Retain the temporary backup element on disk if an element replacement
        fails.

    Returns
    -------
    pandas.DataFrame
        Operation report.
    """

    # ------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------

    def _as_list(values):
        if values is None:
            return []

        if isinstance(values, str):
            return [values]

        return list(values)

    def _deduplicate(values):
        return list(dict.fromkeys(values))

    def _locate_element(name):
        """
        Return element type, container, and element.
        """
        containers = (
            ("images", sdata.images),
            ("labels", sdata.labels),
            ("points", sdata.points),
            ("shapes", sdata.shapes),
            ("tables", sdata.tables),
        )

        for element_type, container in containers:
            if name in container:
                return element_type, container, container[name]

        suggestions = []

        # Helpful corrections for common singular/plural mistakes.
        aliases = {
            name.replace("_label", "_labels"),
            name.replace("_point", "_points"),
            f"{name}s",
        }

        for alias in aliases:
            for element_type, container in containers:
                if alias in container:
                    suggestions.append(alias)

        message = (
            f"Element {name!r} was not found in the SpatialData object.\n"
            f"Available image examples: {list(sdata.images.keys())[:5]}\n"
            f"Available label examples: {list(sdata.labels.keys())[:5]}\n"
            f"Available point examples: {list(sdata.points.keys())[:5]}\n"
            f"Available shapes: {list(sdata.shapes.keys())}\n"
            f"Available tables: {list(sdata.tables.keys())}"
        )

        if suggestions:
            message += f"\nPossible intended name(s): {sorted(set(suggestions))}"

        raise KeyError(message)

    def _copy_for_backup(element):
        """
        Make an independent-enough in-memory copy for the replacement workflow.
        """
        try:
            return element.copy()
        except Exception:
            return copy.deepcopy(element)

    def _write_one_transformation(name):
        """
        Persist only the transformations of one element.
        """
        try:
            sdata.write_transformations(element_name=name)
        except TypeError:
            # Defensive fallback for API variations.
            sdata.write_transformations(name)

    # ------------------------------------------------------------
    # Confirm that the object is backed
    # ------------------------------------------------------------

    try:
        _ = sdata.elements_paths_on_disk()
    except Exception as error:
        raise RuntimeError(
            "This SpatialData object is not associated with a writable Zarr "
            "store. Load it with spatialdata.read_zarr(...) or write it once "
            "before calling persist_changes()."
        ) from error

    print(f"[store] {sdata.path}")

    update_elements = _deduplicate(_as_list(update_elements))
    new_elements = _deduplicate(_as_list(new_elements))

    transformation_elements_was_explicit = (
        transformation_elements is not None
    )
    transformation_elements = _deduplicate(
        _as_list(transformation_elements)
    )

    # ------------------------------------------------------------
    # Decide which elements require data rewrite versus transforms
    # ------------------------------------------------------------

    data_update_elements = []
    transform_update_elements = list(transformation_elements)

    if transformation_elements_was_explicit:
        # Explicit mode:
        # update_elements means actual element data replacement.
        data_update_elements = list(update_elements)

    elif auto_transform_non_tables:
        # Backward-compatible mode:
        # tables are rewritten; all other existing element types are treated
        # as transformation-only.
        for name in update_elements:
            element_type, _, _ = _locate_element(name)

            if element_type == "tables":
                data_update_elements.append(name)
            else:
                transform_update_elements.append(name)

    else:
        # Legacy/full rewrite behavior.
        data_update_elements = list(update_elements)

    data_update_elements = _deduplicate(data_update_elements)
    transform_update_elements = _deduplicate(
        transform_update_elements
    )

    if transform_update_elements and not update_transformations:
        raise ValueError(
            "Transformation elements were supplied, but "
            "update_transformations=False."
        )

    print("\n[persist plan]")
    print("  full data rewrites:", data_update_elements)
    print("  transformation-only:", transform_update_elements)
    print("  new elements:", new_elements)

    report_rows = []

    # ------------------------------------------------------------
    # Rewrite existing elements whose actual data changed
    # ------------------------------------------------------------

    for name in data_update_elements:
        element_type, container, element = _locate_element(name)

        if element_type != "tables" and not allow_non_table_rewrite:
            raise ValueError(
                f"{name!r} is a {element_type[:-1]} element. A full rewrite "
                "of non-table elements is disabled by default.\n"
                "If only its transformation changed, put it in "
                "`transformation_elements` instead.\n"
                "If the underlying element data genuinely changed, pass "
                "`allow_non_table_rewrite=True`, understanding that large "
                "raster rewrites may encounter chunking limitations."
            )

        safe_name = "".join(
            character if character.isalnum() or character in "_-."
            else "_"
            for character in str(name)
            )
        
        backup_name = (
            f"backup_{safe_name}_{uuid.uuid4().hex[:8]}"
            )

        print(
            f"\n[data rewrite] {element_type}/{name}"
        )
        print(f"[backup] temporary element: {backup_name}")

        updated_element = _copy_for_backup(element)
        backup_element = _copy_for_backup(element)

        backup_written = False

        # Place the backup in the correct container—not always tables.
        container[backup_name] = backup_element

        try:
            # 1. Write a recovery copy under a new name.
            sdata.write_element(backup_name)
            backup_written = True
            print(f"[backup] written: {backup_name}")

            # 2. Delete the old on-disk element.
            sdata.delete_element_from_disk(name)
            print(f"[data rewrite] old on-disk element removed: {name}")

            # 3. Restore the updated in-memory element under its original name.
            container[name] = updated_element

            # 4. Write the updated element.
            sdata.write_element(name)
            print(f"[data rewrite] updated element written: {name}")

        except Exception as error:
            report_rows.append(
                {
                    "operation": "data_rewrite",
                    "element_type": element_type,
                    "element": name,
                    "status": "failed",
                    "backup_element": (
                        backup_name if backup_written else None
                    ),
                    "error": type(error).__name__,
                    "error_message": str(error),
                }
            )

            print(
                f"\n[ERROR] failed to replace {element_type}/{name}: "
                f"{type(error).__name__}: {error}"
            )

            if backup_written and keep_backup_on_failure:
                print(
                    f"[recovery] backup {backup_name!r} has been left "
                    "on disk intentionally."
                )
            else:
                # Best-effort cleanup.
                try:
                    if backup_written:
                        sdata.delete_element_from_disk(backup_name)
                except Exception:
                    pass

                if backup_name in container:
                    del container[backup_name]

            raise

        else:
            # Confirmed success: remove the temporary backup.
            try:
                sdata.delete_element_from_disk(backup_name)
            except Exception as cleanup_error:
                print(
                    f"[warn] updated element was written successfully, but "
                    f"the temporary backup could not be removed from disk: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

            if backup_name in container:
                del container[backup_name]

            report_rows.append(
                {
                    "operation": "data_rewrite",
                    "element_type": element_type,
                    "element": name,
                    "status": "ok",
                    "backup_element": backup_name,
                    "error": None,
                    "error_message": None,
                }
            )

    # ------------------------------------------------------------
    # Write newly created elements
    # ------------------------------------------------------------

    for name in new_elements:
        element_type, _, _ = _locate_element(name)

        print(f"\n[new element] writing {element_type}/{name}")

        try:
            sdata.write_element(name)

        except Exception as error:
            report_rows.append(
                {
                    "operation": "new_element",
                    "element_type": element_type,
                    "element": name,
                    "status": "failed",
                    "backup_element": None,
                    "error": type(error).__name__,
                    "error_message": str(error),
                }
            )
            raise

        else:
            report_rows.append(
                {
                    "operation": "new_element",
                    "element_type": element_type,
                    "element": name,
                    "status": "ok",
                    "backup_element": None,
                    "error": None,
                    "error_message": None,
                }
            )

    # ------------------------------------------------------------
    # Persist transformation metadata only
    # ------------------------------------------------------------

    if update_transformations:
        for name in transform_update_elements:
            element_type, _, _ = _locate_element(name)

            if element_type == "tables":
                print(
                    f"[warn] {name!r} is a table; skipping transformation-only "
                    "write because tables are not spatial elements."
                )
                continue

            print(
                f"\n[transform] writing {element_type}/{name}"
            )

            try:
                _write_one_transformation(name)

            except Exception as error:
                report_rows.append(
                    {
                        "operation": "transformation",
                        "element_type": element_type,
                        "element": name,
                        "status": "failed",
                        "backup_element": None,
                        "error": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                raise

            else:
                report_rows.append(
                    {
                        "operation": "transformation",
                        "element_type": element_type,
                        "element": name,
                        "status": "ok",
                        "backup_element": None,
                        "error": None,
                        "error_message": None,
                    }
                )

    # ------------------------------------------------------------
    # Root and consolidated metadata
    # ------------------------------------------------------------

    print("\n[metadata] writing SpatialData metadata")
    sdata.write_metadata()

    if consolidate_metadata:
        print("[metadata] consolidating Zarr metadata")
        sdata.write_consolidated_metadata()

    print("\n[done] selected changes persisted successfully.")

    return pd.DataFrame(report_rows)
###

## update only table part of a spatialdata object

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

###

## import image polygon based annotations 

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

###

## helper function for main function to store categorical annotation in .shapes slot safely 

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

###

## main function to store categorical annotation in .shapes slot safely 

def make_shapes_plot_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_clean_polygonal)
    gdf = gdf[gdf["geometry"].notnull()].reset_index(drop=True)
    # Separate MultiPolygons to Polygons to avoid MultiPolygon patch path
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf

flatten = lambda *n: (e for a in n
    for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

###


## quickly assign polygon annotations from the .shapes slot to the corresponding Visium HD object 

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

# --- function to align the global coordinates to full-resolution image --- #

def _infer_sample_id_from_sdata(sdata):
    """Infer sample id from a '*_hires_image' key."""
    hires_key = next(k for k in sdata.images.keys() if k.endswith("_hires_image"))
    return hires_key.replace("_hires_image", "")


def _infer_fullres_cs(sdata, sample_id=None):
    """
    In Visium HD imports, the full resolution CS is usually named as the sample_id,
    and downscaled CS names look like '<sample_id>_downscaled_hires/lowres'.
    """
    if sample_id is None:
        sample_id = _infer_sample_id_from_sdata(sdata)

    # Most robust: prefer exact sample_id if it exists as a coordinate system
    if sample_id in sdata.coordinate_systems:
        return sample_id

    # Fallback: choose the largest-extent CS for the hires image
    hires_key = f"{sample_id}_hires_image"
    if hires_key not in sdata.images:
        hires_key = next(k for k in sdata.images.keys() if k.endswith("_hires_image"))

    cs_all = list(get_transformation(sdata.images[hires_key], get_all=True).keys())
    sizes = {}
    for cs in cs_all:
        try:
            ext = sd.get_extent(sdata.images[hires_key], coordinate_system=cs)
            w = float(ext["x"][1] - ext["x"][0])
            h = float(ext["y"][1] - ext["y"][0])
            sizes[cs] = w * h
        except Exception:
            pass
    if not sizes:
        raise ValueError("Could not infer fullres CS (no usable extents).")

    return max(sizes, key=sizes.get)


def _copy_cs_to_global(element, src_cs, overwrite=True, verbose=False):
    """
    Copy element's transform from src_cs -> 'global'.
    Returns True if copied, False if src_cs not present.
    """
    tr_all = get_transformation(element, get_all=True)
    if src_cs not in tr_all:
        return False

    if overwrite:
        try:
            remove_transformation(element, to_coordinate_system="global")
        except Exception:
            pass

    set_transformation(element, tr_all[src_cs], to_coordinate_system="global")
    if verbose:
        print(f"  ✓ copied {src_cs} -> global")
    return True

def make_global_fullres(
    sdata,
    sample_id=None,
    *,
    overwrite=True,
    include_images=True,
    include_shapes=True,
    include_labels=True,
    only_these_images=None,   # optional list of image keys to process
    only_these_shapes=None,   # optional list of shape keys to process
    only_these_labels=None,   # optional list of label keys to process
    verbose=True,
):
    """
    Promote the full-resolution coordinate system (fullres) to be the 'global' coordinate system
    by copying transforms full_cs -> global for images/shapes/labels.

    This makes plotting + downstream quantifications consistent with Space Ranger fullres.
    """
    if sample_id is None:
        sample_id = _infer_sample_id_from_sdata(sdata)

    full_cs = _infer_fullres_cs(sdata, sample_id=sample_id)

    if verbose:
        print(f"[make_global_fullres] sample_id={sample_id}")
        print(f"[make_global_fullres] full_cs={full_cs}  (will become 'global')")

    # ---- images ----
    if include_images:
        img_keys = list(sdata.images.keys()) if only_these_images is None else list(only_these_images)
        if verbose:
            print(f"  Images: {len(img_keys)} keys")

        for k in img_keys:
            if k not in sdata.images:
                continue
            ok = _copy_cs_to_global(sdata.images[k], full_cs, overwrite=overwrite, verbose=False)
            if verbose:
                print(f"    {'✓' if ok else '·'} {k}  ({'copied' if ok else 'no full_cs transform'})")

    # ---- shapes ----
    if include_shapes:
        shape_keys = list(sdata.shapes.keys()) if only_these_shapes is None else list(only_these_shapes)
        if verbose:
            print(f"  Shapes: {len(shape_keys)} keys")

        for k in shape_keys:
            if k not in sdata.shapes:
                continue
            ok = _copy_cs_to_global(sdata.shapes[k], full_cs, overwrite=overwrite, verbose=False)
            if verbose:
                print(f"    {'✓' if ok else '·'} {k}  ({'copied' if ok else 'no full_cs transform'})")

    # ---- labels ----
    if include_labels and hasattr(sdata, "labels"):
        label_keys = list(sdata.labels.keys()) if only_these_labels is None else list(only_these_labels)
        if verbose:
            print(f"  Labels: {len(label_keys)} keys")

        for k in label_keys:
            if k not in sdata.labels:
                continue
            ok = _copy_cs_to_global(sdata.labels[k], full_cs, overwrite=overwrite, verbose=False)
            if verbose:
                print(f"    {'✓' if ok else '·'} {k}  ({'copied' if ok else 'no full_cs transform'})")

    return {"sample_id": sample_id, "full_cs": full_cs}

###

# ---- verify all the elements are correctly aligned to the global coordinates ---- #

def verify_global_extents(sdata, sample_id=None, keys=None):
    if sample_id is None:
        sample_id = _infer_sample_id_from_sdata(sdata)

    def _extent_of(kind, key):
        if kind == "image":
            el = sdata.images[key]
        elif kind == "shape":
            el = sdata.shapes[key]
        elif kind == "label":
            el = sdata.labels[key]
        else:
            raise ValueError(kind)
        ext = sd.get_extent(el, coordinate_system="global")
        w = float(ext["x"][1] - ext["x"][0])
        h = float(ext["y"][1] - ext["y"][0])
        return w, h, ext

    if keys is None:
        keys = {
            "image": [f"{sample_id}_hires_image", f"{sample_id}_nhires_image"],
            "shape": [f"{sample_id}_square_002um", "annotations_clean", "annotations_raw"],
        }

    print(f"[verify_global_extents] sample={sample_id}")
    for kind, klist in keys.items():
        for k in klist:
            if kind == "image" and k not in sdata.images:
                continue
            if kind == "shape" and k not in sdata.shapes:
                continue
            if kind == "label" and (not hasattr(sdata, "labels") or k not in sdata.labels):
                continue
            w, h, _ = _extent_of(kind, k)
            print(f"  {kind:5s}  {k:35s}  global extent (w,h)=({w:.2f}, {h:.2f})")

###

# --- 

import numpy as np
import pandas as pd


def find_deprecated_features(adata):
    """
    Return a NumPy boolean mask over adata.var identifying features whose
    ID or name starts with 'DEPRECATED_'.

    Checks var_names and common feature-metadata columns. The return type
    is always a one-dimensional NumPy boolean array.
    """

    candidates = [
        "gene_ids",
        "gene_id",
        "feature_ids",
        "feature_id",
        "id",
        "feature_name",
        "gene",
        "name",
        "gene_name",
        "gene_symbol",
        "symbol",
    ]

    def starts_with_deprecated(values):
        # pandas StringDtype handles missing values cleanly.
        text = pd.Series(
            np.asarray(values, dtype=object),
            dtype="string",
        ).fillna("")

        return (
            text.str.startswith("DEPRECATED_")
            .to_numpy(dtype=bool)
        )

    mask = starts_with_deprecated(adata.var_names)

    for column in candidates:
        if column in adata.var.columns:
            mask |= starts_with_deprecated(
                adata.var[column].to_numpy()
            )

    if mask.shape != (adata.n_vars,):
        raise ValueError(
            "Deprecated-feature mask has the wrong shape: "
            f"{mask.shape}; expected ({adata.n_vars},)"
        )

    return mask

def add_deprecated_qc(adata, deprecated_mask=None, key_prefix="deprecated"):
    """
    Adds per-cell QC metrics:
      obs[f"{key_prefix}_counts"], obs[f"{key_prefix}_pct"]
    Requires obs['total_counts'] (creates it if missing).
    """
    if deprecated_mask is None:
        deprecated_mask = find_deprecated_features(adata)
    # total counts if absent
    if "total_counts" not in adata.obs:
        if sp.issparse(adata.X):
            adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            adata.obs["total_counts"] = adata.X.sum(axis=1)
    # counts from deprecated features
    if sp.issparse(adata.X):
        dep_counts = np.asarray(adata[:, deprecated_mask].X.sum(axis=1)).ravel()
    else:
        dep_counts = adata[:, deprecated_mask].X.sum(axis=1)
    adata.obs[f"{key_prefix}_counts"] = dep_counts
    # percent (guard against div-by-zero)
    tc = adata.obs["total_counts"].replace(0, np.nan)
    adata.obs[f"{key_prefix}_pct"] = 100.0 * adata.obs[f"{key_prefix}_counts"] / tc
    adata.obs[f"{key_prefix}_pct"] = adata.obs[f"{key_prefix}_pct"].fillna(0.0)

def drop_deprecated_features(adata, deprecated_mask=None, drop_from_layers=True):
    """
    Removes deprecated features from adata.var, adata.X, and (optionally) all layers.
    Returns the number dropped.
    """
    if deprecated_mask is None:
        deprecated_mask = find_deprecated_features(adata)
    n_drop = int(deprecated_mask.sum())
    if n_drop == 0:
        return 0
    keep = ~deprecated_mask
    # X
    adata._inplace_subset_var(keep)
    # layers
    if drop_from_layers and hasattr(adata, "layers") and len(adata.layers) > 0:
        for k in list(adata.layers.keys()):
            mat = adata.layers[k]
            if sp.issparse(mat):
                adata.layers[k] = mat[:, keep]
            else:
                adata.layers[k] = np.asarray(mat)[:, keep]
    return n_drop

###

# --- 


def relabel_deprecated_feature_types(
    adata,
    type_col: str = "feature_types",
    deprecated_label: str = "Deprecated_Gene_Expression",
    default_when_missing: str = "Gene_Expression",
) -> int:
    """
    Relabel features whose ID or name starts with ``DEPRECATED_``.

    The deprecated-feature definition is provided by
    ``_deprecated_mask_from_var(adata.var)``.

    Effects
    -------
    - Creates or updates ``adata.var["is_deprecated"]`` as a Boolean column.
    - Creates ``adata.var[type_col]`` when it is absent.
    - Preserves categorical dtype when ``type_col`` is categorical.
    - Adds ``deprecated_label`` as a category when needed.
    - Returns the number of deprecated features relabeled.
    """

    var = adata.var

    # --------------------------------------------------------------
    # 1. Identify deprecated features
    # --------------------------------------------------------------
    mask = np.asarray(
        _deprecated_mask_from_var(var),
        dtype=bool,
    ).reshape(-1)

    if mask.shape != (adata.n_vars,):
        raise ValueError(
            "Deprecated-feature mask has the wrong shape: "
            f"{mask.shape}; expected ({adata.n_vars},)."
        )

    n_dep = int(mask.sum())

    # A standard NumPy/Pandas Boolean column.
    var["is_deprecated"] = mask

    if n_dep == 0:
        print(
            "[info] No features whose ID or name begins with "
            "'DEPRECATED_' were found. Nothing was relabeled."
        )
        return 0

    # --------------------------------------------------------------
    # 2. Ensure the feature-type column exists
    # --------------------------------------------------------------
    if type_col not in var.columns:
        var[type_col] = default_when_missing
        print(
            f"[info] Created adata.var[{type_col!r}] and initialized "
            f"all features as {default_when_missing!r}."
        )

    before_counts = (
        var[type_col]
        .value_counts(dropna=False)
        .rename("before")
    )

    # --------------------------------------------------------------
    # 3. Relabel deprecated features
    # --------------------------------------------------------------
    if isinstance(
        var[type_col].dtype,
        pd.CategoricalDtype,
    ):
        if (
            deprecated_label
            not in var[type_col].cat.categories
        ):
            var[type_col] = (
                var[type_col]
                .cat.add_categories([deprecated_label])
            )

        var.loc[mask, type_col] = deprecated_label

        # Safe but optional. Remove this line if you want to preserve
        # all previously declared categories, including unused ones.
        var[type_col] = (
            var[type_col]
            .cat.remove_unused_categories()
        )

    else:
        # Convert to object first when the underlying dtype is a fixed-width
        # string dtype or another dtype that may reject a new string value.
        if not (
            pd.api.types.is_object_dtype(
                var[type_col].dtype
            )
            or pd.api.types.is_string_dtype(
                var[type_col].dtype
            )
        ):
            var[type_col] = (
                var[type_col]
                .astype("object")
            )

        var.loc[mask, type_col] = deprecated_label

    # --------------------------------------------------------------
    # 4. Report before/after counts
    # --------------------------------------------------------------
    after_counts = (
        var[type_col]
        .value_counts(dropna=False)
        .rename("after")
    )

    summary = (
        pd.concat(
            [before_counts, after_counts],
            axis=1,
        )
        .fillna(0)
        .astype(np.int64)
    )

    print(
        f"[ok] Relabeled {n_dep:,} DEPRECATED_ features "
        f"as {deprecated_label!r} in adata.var[{type_col!r}]."
    )
    print("\nCounts by feature type:")
    print(summary)

    return n_dep

###

# --- 

def _deprecated_mask_from_var(var: pd.DataFrame) -> np.ndarray:
    """Find features whose ID/name starts with 'DEPRECATED_' (robust across common columns)."""
    mask = var.index.astype(str).str.startswith("DEPRECATED_")
    for col in ["gene_ids", "feature_ids", "id", "feature_name", "gene_name", "gene_symbol"]:
        if col in var:
            mask |= var[col].astype(str).str.startswith("DEPRECATED_")
    if "is_deprecated" in var:
        mask |= var["is_deprecated"].astype(bool).values
    return mask.values





def _deprecated_mask_from_var(
    var: pd.DataFrame,
) -> np.ndarray:
    """
    Return a one-dimensional NumPy boolean mask identifying features whose
    ID or name starts with 'DEPRECATED_'.

    Checks var.index, which corresponds to adata.var_names, and common
    feature-metadata columns. The return type is always np.ndarray.
    """

    candidates = [
        "gene_ids",
        "gene_id",
        "feature_ids",
        "feature_id",
        "id",
        "feature_name",
        "gene",
        "name",
        "gene_name",
        "gene_symbol",
        "symbol",
    ]

    def starts_with_deprecated(values) -> np.ndarray:
        text = pd.Series(
            np.asarray(values, dtype=object),
            dtype="string",
        ).fillna("")

        return (
            text.str.startswith("DEPRECATED_")
            .to_numpy(dtype=bool)
        )

    # adata.var.index is adata.var_names
    mask = starts_with_deprecated(var.index)

    for column in candidates:
        if column in var.columns:
            mask |= starts_with_deprecated(
                var[column].to_numpy()
            )

    if mask.shape != (len(var),):
        raise ValueError(
            "Deprecated-feature mask has the wrong shape: "
            f"{mask.shape}; expected ({len(var)},)"
        )

    return mask


def _matrix_row_sums(matrix) -> np.ndarray:
    """
    Return row sums as a flat NumPy float64 array for sparse or dense matrices.
    """
    totals = matrix.sum(axis=1)

    if hasattr(totals, "A1"):
        totals = totals.A1
    else:
        totals = np.asarray(totals).reshape(-1)

    return np.asarray(totals, dtype=np.float64)


def qc_without_deprecated(
    adata,
    *,
    counts_layer: str | None = None,
    percent_top: tuple[int, ...] | None = (
        10,
        50,
        100,
        200,
        500,
    ),
    obs_prefix: str = "nodep_",
    var_prefix: str = "nodep_",
    log1p: bool = False,
    inplace: bool = True,
):
    """
    Calculate Scanpy QC metrics after excluding features whose ID/name begins
    with 'DEPRECATED_'.

    Parameters
    ----------
    adata
        AnnData object.

    counts_layer
        Raw-count layer to use. If None, adata.X is used. An invalid layer
        name raises an error rather than silently falling back to adata.X.

    percent_top
        Feature-expression ranks used for cumulative count percentages.
        Values greater than the number of retained features are omitted.

    obs_prefix, var_prefix
        Prefixes applied to newly calculated QC columns.

    log1p
        Whether Scanpy should also calculate log1p-transformed QC annotations.
        False is generally simpler for raw-count QC.

    inplace
        If True, attach results to adata and return None.
        If False, return the calculated DataFrames without modifying adata.

    Returns
    -------
    None or dict
        If inplace=False:

        {
            "obs": observation-level QC DataFrame,
            "var": retained-feature QC DataFrame,
            "deprecated_mask": full feature-level boolean mask,
        }
    """

    # --------------------------------------------------------------
    # Validate and standardize the deprecated-feature mask
    # --------------------------------------------------------------
    dep_mask = np.asarray(
        _deprecated_mask_from_var(adata.var),
        dtype=bool,
    ).reshape(-1)

    if dep_mask.shape != (adata.n_vars,):
        raise ValueError(
            "Deprecated-feature mask has the wrong shape: "
            f"{dep_mask.shape}; expected ({adata.n_vars},)"
        )

    keep_mask = ~dep_mask
    n_dep = int(dep_mask.sum())
    n_keep = int(keep_mask.sum())

    if n_keep == 0:
        raise ValueError(
            "All features were classified as DEPRECATED_; "
            "QC cannot be calculated on an empty feature set."
        )

    # --------------------------------------------------------------
    # Validate the raw-count source
    # --------------------------------------------------------------
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"Requested counts layer {counts_layer!r} was not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )

        count_matrix = adata.layers[counts_layer]
        count_source = f"layers[{counts_layer!r}]"
    else:
        count_matrix = adata.X
        count_source = "X"

    if count_matrix.shape != adata.shape:
        raise ValueError(
            f"Count matrix from {count_source} has shape "
            f"{count_matrix.shape}; expected {adata.shape}."
        )

    all_total_counts = _matrix_row_sums(
        count_matrix
    )

    # --------------------------------------------------------------
    # Keep only valid percent_top ranks
    # --------------------------------------------------------------
    if percent_top is None:
        valid_percent_top = None
        omitted_percent_top = []
    else:
        requested = tuple(
            dict.fromkeys(int(value) for value in percent_top)
        )

        valid_percent_top = [
            value
            for value in requested
            if 1 <= value <= n_keep
        ]
        omitted_percent_top = [
            value
            for value in requested
            if value < 1 or value > n_keep
        ]

        if not valid_percent_top:
            valid_percent_top = None

    if omitted_percent_top:
        print(
            "[info] Omitted percent_top values outside the "
            f"retained-feature range 1–{n_keep}: "
            f"{omitted_percent_top}"
        )

    # --------------------------------------------------------------
    # Calculate QC on the nondeprecated feature view
    # --------------------------------------------------------------
    adata_view = adata[:, keep_mask]

    obs_qc, var_qc = sc.pp.calculate_qc_metrics(
        adata_view,
        percent_top=valid_percent_top,
        layer=counts_layer,
        inplace=False,
        log1p=log1p,
    )

    obs_qc = obs_qc.add_prefix(obs_prefix)
    var_qc = var_qc.add_prefix(var_prefix)

    # --------------------------------------------------------------
    # Calculate deprecated-count contribution from the same matrix
    # --------------------------------------------------------------
    nodep_total_column = (
        f"{obs_prefix}total_counts"
    )

    if nodep_total_column not in obs_qc.columns:
        raise KeyError(
            f"Expected Scanpy QC column "
            f"{nodep_total_column!r} was not generated."
        )

    nodep_total_counts = obs_qc[
        nodep_total_column
    ].to_numpy(dtype=np.float64)

    deprecated_counts = (
        all_total_counts - nodep_total_counts
    )

    # Permit tiny floating-point differences but reject meaningful
    # negative values, which indicate a source/alignment mismatch.
    tolerance = max(
        1e-8,
        float(np.nanmax(all_total_counts))
        * 1e-12,
    )

    minimum_difference = float(
        np.nanmin(deprecated_counts)
    )

    if minimum_difference < -tolerance:
        raise ValueError(
            "Nondeprecated counts exceed total counts for at least one "
            "observation. This usually indicates that total and nodep QC "
            "were calculated from different matrices or layers. "
            f"Minimum difference: {minimum_difference}"
        )

    deprecated_counts = np.clip(
        deprecated_counts,
        a_min=0.0,
        a_max=None,
    )

    deprecated_pct = np.divide(
        100.0 * deprecated_counts,
        all_total_counts,
        out=np.zeros_like(
            deprecated_counts,
            dtype=np.float64,
        ),
        where=all_total_counts > 0,
    )

    obs_qc["deprecated_counts"] = (
        deprecated_counts
    )
    obs_qc["deprecated_pct"] = deprecated_pct

    # --------------------------------------------------------------
    # Non-mutating return
    # --------------------------------------------------------------
    if not inplace:
        print(
            "[ok] QC calculated without modifying the object: "
            f"excluded {n_dep:,} DEPRECATED_ features; "
            f"retained {n_keep:,}; count source={count_source}."
        )

        return {
            "obs": obs_qc,
            "var": var_qc,
            "deprecated_mask": dep_mask,
        }

    # --------------------------------------------------------------
    # Attach observation-level results by position
    # --------------------------------------------------------------
    if not obs_qc.index.equals(adata.obs_names):
        obs_qc = obs_qc.reindex(
            adata.obs_names
        )

        if obs_qc.isna().all(axis=None):
            raise ValueError(
                "Observation-level QC results could not be aligned "
                "to adata.obs_names."
            )

    for column in obs_qc.columns:
        adata.obs[column] = (
            obs_qc[column].to_numpy()
        )

    # Add unfiltered totals only when absent. The deprecated-count
    # calculation above never depends on an existing total_counts column.
    if "total_counts" not in adata.obs.columns:
        adata.obs["total_counts"] = (
            all_total_counts
        )

    # --------------------------------------------------------------
    # Attach variable-level results
    #
    # Deprecated-feature rows are deliberately set to NaN. Initializing
    # the whole column prevents stale values if the function is rerun.
    # --------------------------------------------------------------
    for column in var_qc.columns:
        full_column = np.full(
            adata.n_vars,
            np.nan,
            dtype=np.float64,
        )

        full_column[keep_mask] = (
            var_qc[column]
            .to_numpy(dtype=np.float64)
        )

        adata.var[column] = full_column

    # --------------------------------------------------------------
    # Store provenance
    # --------------------------------------------------------------
    adata.uns["qc_without_deprecated"] = {
        "n_deprecated_features": n_dep,
        "n_retained_features": n_keep,
        "counts_source": count_source,
        "counts_layer": counts_layer,
        "obs_prefix": obs_prefix,
        "var_prefix": var_prefix,
        "percent_top_requested": (
            None
            if percent_top is None
            else [int(x) for x in percent_top]
        ),
        "percent_top_used": (
            None
            if valid_percent_top is None
            else [int(x) for x in valid_percent_top]
        ),
        "log1p": bool(log1p),
    }

    print(
        "[ok] QC computed excluding "
        f"{n_dep:,} DEPRECATED_ features; "
        f"retained {n_keep:,}; "
        f"count source={count_source}."
    )

    return None


###

##+++++++ function under development to load cell segmented data from 10x +++++++##

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


##===================================== normalization related section ====================================##


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

