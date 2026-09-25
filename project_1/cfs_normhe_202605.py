###=======================================================================================================###
###                                                                                                       ###            
###                              Custom functions for easy H&E normalization                              ###
###                                                                                                       ###
###=======================================================================================================###

##======================================first load all required packages===================================##
import numpy as np
import pandas as pd
import scipy.sparse as so
import spatialdata as sd
import matplotlib.pyplot as plt
from scipy.stats import t as tdist, rankdata, hypergeom
from statsmodels.stats.multitest import multipletests
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
import gc
from typing import Optional, Tuple, Union
import json

from spatialdata.transformations import (
    get_transformation,
    set_transformation,
    remove_transformation,
)

##=========================================================================================================##

# ---------- read visium json file ---------- #

def read_json_to_dict(file_path):
    """
    Reads a JSON file and converts its content into a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A Python dictionary representing the JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



# ---------- helpers ---------- #

def _as_hwc(img_cyx: np.ndarray) -> np.ndarray:
    """
    Convert SpatialData-style image from (c, y, x) to (y, x, c).
    Keeps only first 3 channels if RGBA.
    """
    if img_cyx.ndim != 3 or img_cyx.shape[0] not in (3, 4):
        raise ValueError(
            f"Expected image with shape (c, y, x) and 3 or 4 channels. "
            f"Got shape {img_cyx.shape}."
        )
    return np.moveaxis(img_cyx[:3], 0, -1)


def _as_cyx(img_hwc: np.ndarray) -> np.ndarray:
    """
    Convert image from (y, x, c) to (c, y, x).
    """
    if img_hwc.ndim != 3 or img_hwc.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected image with shape (y, x, c) and 3 or 4 channels. "
            f"Got shape {img_hwc.shape}."
        )
    return np.moveaxis(img_hwc[..., :3], -1, 0)


def _auto_Io(img: np.ndarray) -> float:
    """
    Pick a robust transmitted-light / white intensity for OD computation.
    Works for uint8, uint16, and float images.
    """
    if np.issubdtype(img.dtype, np.integer):
        Io = float(np.percentile(img, 99.9))
        if Io < 10:
            maxv = float(np.iinfo(img.dtype).max)
            Io = 240.0 if maxv <= 255 else maxv
    else:
        m = float(np.nanmax(img))
        if m <= 1.5:
            Io = 1.0
        else:
            Io = float(np.percentile(img, 99.9))
            if Io < 1.0:
                Io = m
    return Io


def _batch_lstsq(HE: np.ndarray, Y: np.ndarray, batch: int = 100_000) -> np.ndarray:
    """
    Solve C = argmin ||HE*C - Y|| in batches.

    HE: (3, 2)
    Y:  (3, N)
    returns C: (2, N)
    """
    Cs = []
    for i in range(0, Y.shape[1], batch):
        C_i, _, _, _ = np.linalg.lstsq(HE, Y[:, i:i + batch], rcond=None)
        Cs.append(C_i)
    return np.concatenate(Cs, axis=1)

# ---------- main normalizer ---------- #

def macenko_normalize_visiumhd(
    img,
    mask=None,                         # (h,w) bool; True = use for basis estimation
    Io="auto",
    alpha=1.0,
    beta=0.15,
    sample_max=200_000,
    lstsq_batch=200_000,
    return_HE_H_E=False,

    # ---- new options ----
    stain_mode="both",                 # "both", "hematoxylin", "eosin", "none"
    basis_mode="auto",                 # "auto" or "reference"
    output_basis="hybrid",             # "hybrid", "reference", "sample"
    concentration_percentile=99,
    clip_concentrations=False,
    random_state=0,
    
    # ---- concentration output options ----
    return_concentrations=False,
    concentration_output="normalized", # "normalized", "raw", or "both"
):
    """
    Macenko H&E normalization adapted for Visium HD / SpatialData images.

    Parameters
    ----------
    stain_mode
        Which stain concentration(s) to normalize:
        - "both": normalize hematoxylin and eosin. This is the classic Macenko behavior.
        - "hematoxylin": normalize only the H concentration; keep E closer to the sample.
        - "eosin": normalize only the E concentration; keep H closer to the sample.
        - "none": no concentration normalization; useful for diagnostic reconstruction.

    basis_mode
        - "auto": estimate stain vectors from the image using Macenko.
        - "reference": use HERef directly for deconvolution. This can be more stable when
          one stain is weak or poorly separated.

    output_basis
        - "hybrid": use reference vector only for the stain(s) being normalized, and use
          sample-estimated vector for the stain(s) not being normalized.
        - "reference": reconstruct all stains using HERef.
        - "sample": reconstruct all stains using the sample-estimated HE matrix.

    Returns
    -------
    Inorm_cyx : uint8, shape (c,y,x)

    If return_HE_H_E=True:
        Inorm_cyx, HE, H_img, E_img
    """

    # ----------------------------
    # Reference stain matrix
    # ----------------------------
    HERef = np.array(
        [[0.5626, 0.2159],
         [0.7201, 0.8012],
         [0.4062, 0.5581]],
        dtype=np.float64
    )

    maxCRef = np.array([1.9705, 1.0308], dtype=np.float64)

    # ----------------------------
    # Validate modes
    # ----------------------------
    stain_mode = str(stain_mode).lower()
    aliases = {
        "h": "hematoxylin",
        "he": "both",
        "h&e": "both",
        "e": "eosin",
        "no": "none",
        "off": "none",
    }
    stain_mode = aliases.get(stain_mode, stain_mode)

    if stain_mode not in {"both", "hematoxylin", "eosin", "none"}:
        raise ValueError("stain_mode must be one of: 'both', 'hematoxylin', 'eosin', 'none'.")

    basis_mode = str(basis_mode).lower()
    if basis_mode not in {"auto", "reference"}:
        raise ValueError("basis_mode must be 'auto' or 'reference'.")

    output_basis = str(output_basis).lower()
    if output_basis not in {"hybrid", "reference", "sample"}:
        raise ValueError("output_basis must be 'hybrid', 'reference', or 'sample'.")

    normalize_h = stain_mode in {"both", "hematoxylin"}
    normalize_e = stain_mode in {"both", "eosin"}

    # ----------------------------
    # Accept xarray or numpy
    # ----------------------------
    try:
        import xarray as xr
        if isinstance(img, xr.DataArray):
            arr = img.values
        else:
            arr = img
    except Exception:
        arr = img

    # Orient to HWC
    if arr.shape[0] in (3, 4):      # (c,y,x)
        I = _as_hwc(arr)
    elif arr.shape[-1] in (3, 4):   # (h,w,c)
        I = arr[..., :3]
    else:
        raise ValueError("Provide an RGB image as (c,y,x) or (h,w,c).")

    h, w, _ = I.shape
    I = I.astype(np.float64, copy=False)

    # Auto Io
    Io_val = _auto_Io(I) if (isinstance(Io, str) and Io == "auto") else float(Io)

    # Optical density
    OD = -np.log((I + 1.0) / Io_val).astype(np.float32, copy=False)
    OD_flat = OD.reshape(-1, 3)

    # ----------------------------
    # Choose pixels for basis estimation
    # ----------------------------
    if mask is not None:
        if mask.shape != (h, w):
            raise ValueError("mask must have shape (h, w).")
        sel = mask.ravel().astype(bool)
    else:
        sel = np.ones((h * w,), dtype=bool)

    # Remove near-transparent / near-white pixels
    # This keeps rows where at least one channel has meaningful OD.
    ODsel = OD_flat[sel]
    good_sel = ~(ODsel < beta).all(axis=1)
    ODhat = ODsel[good_sel]

    if ODhat.shape[0] < 100:
        raise ValueError(
            f"Too few valid stained pixels after OD filtering: {ODhat.shape[0]}. "
            "Try lowering beta, providing a tissue mask, or using basis_mode='reference'."
        )

    # Subsample for stain basis estimation
    rng = np.random.default_rng(random_state)
    if ODhat.shape[0] > sample_max:
        idx = rng.choice(ODhat.shape[0], sample_max, replace=False)
        ODsub = ODhat[idx]
    else:
        ODsub = ODhat

    # ----------------------------
    # Estimate or set HE matrix
    # ----------------------------
    if basis_mode == "reference":
        HE = HERef.copy()
    else:
        # Macenko eigenvector estimation
        w_eig, v_eig = np.linalg.eigh(np.cov(ODsub.T))
        order = np.argsort(w_eig)
        v_eig = v_eig[:, order]

        That = ODsub @ v_eig[:, 1:3]
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = v_eig[:, 1:3] @ np.array([np.cos(minPhi), np.sin(minPhi)])
        vMax = v_eig[:, 1:3] @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

        HE = np.stack([vMin, vMax], axis=1)

        # Heuristic: hematoxylin first, eosin second
        if HE[0, 0] < HE[0, 1]:
            HE = HE[:, ::-1]

        # Normalize columns defensively
        HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)

    # ----------------------------
    # Solve stain concentrations: OD = HE * C
    # ----------------------------
    Y = OD_flat.T  # (3, N)
    gc.collect()
    C = _batch_lstsq(HE, Y, batch=lstsq_batch)  # (2, N)
    C = C.astype(np.float32, copy=False)
    gc.collect()

    if clip_concentrations:
        C[C < 0] = 0
        
    # Store raw concentration maps before selective normalization
    H_conc_raw = C[0, :].reshape(h, w).astype(np.float32, copy=False)
    E_conc_raw = C[1, :].reshape(h, w).astype(np.float32, copy=False)
    
    # Use tissue / valid pixels for concentration percentile scaling
    scale_mask = np.zeros(h * w, dtype=bool)
    scale_mask[np.where(sel)[0][good_sel]] = True

    if scale_mask.sum() < 100:
        C_for_scale = C
    else:
        C_for_scale = C[:, scale_mask]

    maxC = np.array(
        [
            np.percentile(C_for_scale[0, :], concentration_percentile),
            np.percentile(C_for_scale[1, :], concentration_percentile),
        ],
        dtype=np.float64,
    )

    # ----------------------------
    # Scale concentrations selectively
    # ----------------------------
    scale = np.ones(2, dtype=np.float64)

    if normalize_h:
        scale[0] = maxC[0] / maxCRef[0] if maxC[0] > 0 else 1.0
    if normalize_e:
        scale[1] = maxC[1] / maxCRef[1] if maxC[1] > 0 else 1.0

    C2 = C.copy()
    C2[0, :] = C2[0, :] / scale[0]
    C2[1, :] = C2[1, :] / scale[1]

    # Store normalized/selectively scaled concentration maps
    H_conc = C2[0, :].reshape(h, w).astype(np.float32)
    E_conc = C2[1, :].reshape(h, w).astype(np.float32)

    # ----------------------------
    # Choose output stain basis
    # ----------------------------
    if output_basis == "reference":
        HE_out = HERef.copy()

    elif output_basis == "sample":
        HE_out = HE.copy()

    else:
        # hybrid: reference only for normalized stains
        HE_out = HE.copy()
        if normalize_h:
            HE_out[:, 0] = HERef[:, 0]
        if normalize_e:
            HE_out[:, 1] = HERef[:, 1]

    # ----------------------------
    # Reconstruct normalized image
    # ----------------------------
    Inorm = (Io_val * np.exp(-(HE_out @ C2))).T.reshape(h, w, 3)
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)

    # Diagnostic single-stain reconstructions
    H_img = (Io_val * np.exp(-(HE_out[:, [0]] @ C2[[0], :]))).T.reshape(h, w, 3)
    H_img = np.clip(H_img, 0, 255).astype(np.uint8)

    E_img = (Io_val * np.exp(-(HE_out[:, [1]] @ C2[[1], :]))).T.reshape(h, w, 3)
    E_img = np.clip(E_img, 0, 255).astype(np.uint8)

    Inorm_cyx = _as_cyx(Inorm)

    if return_HE_H_E:
        return Inorm_cyx, HE.astype(np.float64), H_img, E_img
    if return_concentrations:
        if return_HE_H_E:
            return Inorm_cyx, HE.astype(np.float64), H_img, E_img, H_conc, E_conc
        return Inorm_cyx, H_conc, E_conc

    return Inorm_cyx

### ------------------------------------------------------------------------ ###

# ---------- convert image to uint 8 format ---------- #

def robust_rescale_to_uint8(x, q_low=1, q_high=99):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x, [q_low, q_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (255 * y).astype(np.uint8)

# ---- extract hematoxylin and eosin channels from the normalized image container object ---- #

def extract_he_from_imagecontainer(
    ic,
    layer="hires",
    h_channel=0,
    e_channel=1,
    compute=True,
):
    """
    Extract hematoxylin/eosin channels from a Squidpy ImageContainer.

    Assumes:
        channel 0 = hematoxylin
        channel 1 = eosin

    Returns
    -------
    H : np.ndarray, shape (y, x)
    E : np.ndarray, shape (y, x)
    """
    da = ic[layer]  # xarray.DataArray

    # Force lazy/dask-backed data into memory for preview
    if compute and hasattr(da, "compute"):
        da = da.compute()

    # Identify channel dimension
    channel_dim = None
    for cand in ["channels", "channel", "c"]:
        if cand in da.dims:
            channel_dim = cand
            break

    if channel_dim is None:
        arr = np.asarray(da.values)

        # Common case: (y, x, channels)
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            return arr[..., h_channel], arr[..., e_channel]

        # Common singleton case: (y, x, 1, channels)
        if arr.ndim == 4 and arr.shape[-1] >= 2:
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                return arr[..., h_channel], arr[..., e_channel]

        raise ValueError(f"Could not infer channel dimension. Data shape: {arr.shape}, dims: {da.dims}")

    # Select H and E channels as DataArrays
    H_da = da.isel({channel_dim: h_channel})
    E_da = da.isel({channel_dim: e_channel})

    # Drop singleton non-spatial dimensions such as z/scale
    for d in list(H_da.dims):
        if d not in ("y", "x"):
            if H_da.sizes[d] == 1:
                H_da = H_da.isel({d: 0})
                E_da = E_da.isel({d: 0})
            else:
                raise ValueError(
                    f"Unexpected non-singleton dimension {d!r} with size {H_da.sizes[d]}."
                )

    H = np.asarray(H_da.transpose("y", "x").values)
    E = np.asarray(E_da.transpose("y", "x").values)

    return H, E
###

# ---- ploter function to generate single channel (H and E) plots ---- #

def _robust_limits(x, q_low=1, q_high=99, mask=None):
    x = np.asarray(x, dtype=np.float32)

    if mask is not None:
        vals = x[np.asarray(mask).astype(bool)]
    else:
        vals = x[np.isfinite(x)]

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, 1

    lo, hi = np.percentile(vals, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if hi <= lo:
            hi = lo + 1

    return lo, hi

import matplotlib.pyplot as plt

def preview_he_concentration_maps(
    H_conc,
    E_conc,
    *,
    original_cyx=None,
    q_low=1,
    q_high=99,
    downsample=1,
    mask=None,
    stain_dark=True,
    figsize=(16, 5),
):
    """
    Preview Hematoxylin and Eosin concentration maps.

    Parameters
    ----------
    H_conc, E_conc
        2D concentration maps, shape (y, x).
    original_cyx
        Optional original RGB image, shape (c, y, x), shown in first panel.
    downsample
        Plot every Nth pixel for faster preview.
    stain_dark
        If True, stronger stain appears darker using gray_r.
    """
    cmap = "gray_r" if stain_dark else "gray"

    H_show = H_conc[::downsample, ::downsample]
    E_show = E_conc[::downsample, ::downsample]

    H_vmin, H_vmax = _robust_limits(H_conc, q_low=q_low, q_high=q_high, mask=mask)
    E_vmin, E_vmax = _robust_limits(E_conc, q_low=q_low, q_high=q_high, mask=mask)

    ncols = 3 if original_cyx is not None else 2
    fig, axs = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)

    if ncols == 2:
        axH, axE = axs
    else:
        orig = np.moveaxis(original_cyx, 0, -1)
        orig = orig[::downsample, ::downsample, :]

        axs[0].imshow(orig)
        axs[0].set_title("Original RGB")
        axs[0].axis("off")

        axH, axE = axs[1], axs[2]

    imH = axH.imshow(H_show, cmap=cmap, vmin=H_vmin, vmax=H_vmax)
    axH.set_title("Hematoxylin concentration")
    axH.axis("off")
    plt.colorbar(imH, ax=axH, fraction=0.046, pad=0.04)

    imE = axE.imshow(E_show, cmap=cmap, vmin=E_vmin, vmax=E_vmax)
    axE.set_title("Eosin concentration")
    axE.axis("off")
    plt.colorbar(imE, ax=axE, fraction=0.046, pad=0.04)

    plt.show()

##=========================================================================================================##
