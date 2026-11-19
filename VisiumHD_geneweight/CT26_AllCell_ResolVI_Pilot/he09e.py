"""Review registered H&E beside native 2-µm QC and out-of-mask RNA in 09B core crops."""
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

MARKERS = ("Lyz2", "C1qa", "C1qb")
CLASSES = {
    "tissue-high_transcript-high": ("#115b9d", "High H&E, high RNA"),
    "tissue-high_transcript-low": ("#66bcd2", "High H&E, low RNA"),
    "tissue-low_transcript-high": ("#d15400", "Low H&E, high RNA"),
    "tissue-low_transcript-low": ("#80798f", "Low H&E, low RNA"),
}
GHOST = "tissue-low_transcript-high"


def _inside(xy, center, width):
    return np.max(np.abs(xy - center), axis=1) <= width / 2


def _regions(tile, qc, macrophages, requested, width):
    meta = tile["meta"]
    center = np.asarray(meta["center_um"], dtype=float)
    radius = float(meta["inner_radius_um"])
    xy = tile["spots"].loc[tile["spots"].in_core, ["x_um","y_um"]].to_numpy()
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Overlay width must be positive in microns.")
    if requested is not None:
        centers = np.asarray(requested, dtype=float).reshape(-1, 2)
        if not len(centers) or not np.isfinite(centers).all():
            raise ValueError("Supply finite (Proseg x, Proseg y) crop centers in µm.")
        if (np.linalg.norm(centers-center, axis=1) > radius).any():
            raise ValueError("Requested crop center lies outside the original evaluation core.")
        return [(p, "Selected region") for p in centers], width

    # Smaller fixtures/tiles still permit two distinct views when enough tissue exists.
    width = min(width, 0.7 * radius)
    step = max(width / 2, radius / 80)
    offsets = np.arange(-int(radius / step), int(radius / step) + 1) * step
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    delta = np.column_stack([dx.ravel(), dy.ravel()])
    candidates = center + delta[np.linalg.norm(np.abs(delta) + width / 2, axis=1) <= radius]
    n_squares = cKDTree(xy).query_ball_point(candidates, width/2, p=np.inf, return_length=True)
    n_mac = (cKDTree(macrophages).query_ball_point(candidates, width/2, p=np.inf, return_length=True)
             if len(macrophages) else np.zeros(len(candidates), dtype=int))
    valid = np.flatnonzero((n_squares > 0) & (n_squares >= 0.8 * n_squares.max()))
    if not len(valid):
        raise ValueError("No auto-selected crop intersects the observed in-tissue square centers.")
    # Select on previously assigned cell labels and capture coverage; no marker RNA is consulted.
    rich = valid[np.lexsort((valid, -n_squares[valid], -n_mac[valid]))[0]]
    names = [(candidates[rich], "Macrophage-rich region" if n_mac[rich] else "High-coverage region")]
    separate = valid[np.max(np.abs(candidates[valid] - candidates[rich]), axis=1) >= width]
    if len(separate):
        k = separate[np.lexsort((separate, np.abs(n_squares[separate] - n_squares[rich]), n_mac[separate]))[0]]
        names.append((candidates[k], "Lower-macrophage region" if n_mac[k] < n_mac[rich] else "Comparison region"))
    # Also show a different ghost-enriched location if the pre-existing QC class
    # occurs outside masks in this region. Marker abundances never select a crop.
    spots = tile["spots"]
    ghost = np.asarray(spots.in_core) & (spots.cell_index.to_numpy() < 0) & (qc == GHOST)
    if ghost.any():
        ghost_xy = spots.loc[ghost,["x_um","y_um"]].to_numpy()
        n_ghost = cKDTree(ghost_xy).query_ball_point(candidates,width/2,p=np.inf,return_length=True)
        chosen = np.array([c for c,_ in names])
        separated = np.min(np.max(np.abs(candidates[:,None,:] - chosen[None,:,:]), axis=2),axis=1) >= width
        possible = np.flatnonzero(separated & (n_ghost > 0))
        if len(possible):
            k = possible[np.lexsort((possible, -n_squares[possible], -n_ghost[possible]))[0]]
            names.append((candidates[k], "Low-H&E, high-RNA region"))
    return names, width


def _sources(meta, manual):
    paths = meta["inputs"]
    preflight_path = Path(paths["preflight_json"])
    pre = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
    if pre.get("sample", meta["sample"]) != meta["sample"]:
        raise ValueError("The image preflight and native squares name different samples.")
    square = Path(paths["square_dir"]).expanduser().resolve()
    outs = square.parent.parent
    record = pre.get("image_path")
    if manual:
        natives = [Path(manual).expanduser()]
    elif record:
        natives = [Path(record), outs.parent.parent / "images" / Path(record).name]
    else:
        natives = []
    for path in dict.fromkeys(natives):
        yield ("native", path, pre)
    spatial_dirs = [Path(paths["scalefactors"]).parent, square / "spatial",
                    outs / "spatial", square.parent / "square_008um" / "spatial"]
    for folder in dict.fromkeys(spatial_dirs):
        yield ("hires", folder / "tissue_hires_image.png", pre)


@contextmanager
def _open_source(kind, path, pre):
    if not path.is_file():
        raise FileNotFoundError(path)
    if kind == "hires":
        scale_file = path.with_name("scalefactors_json.json")
        scale = float(json.loads(scale_file.read_text())["tissue_hires_scalef"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid tissue_hires_scalef in {scale_file}")
        affine = np.array([[scale,0.,0.],[0.,scale,0.]])
        with Image.open(path) as im:
            if im.width * im.height > 64_000_000:
                raise ValueError("Space Ranger hires raster exceeds 64 MP. Use native OpenSlide region access.")
            def crop(box):
                return im.crop(box).convert("RGB")
            yield crop, im.size, affine, {"kind":"Space Ranger registered hires PNG", "scale_file":str(scale_file),
                                           "tissue_hires_scalef":scale}
    else:
        import openslide
        affine = np.asarray(pre["tenx_to_image_affine"],dtype=float).reshape(2,3)
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:,:2])) < 1e-10:
            raise ValueError("The recorded 10x full-resolution to slide affine is invalid.")
        try:
            slide = openslide.OpenSlide(str(path))
        except openslide.OpenSlideError as exc:
            raise OSError(f"OpenSlide could not read registered native image {path}: {exc}") from exc
        with slide:
            size = slide.dimensions
            info = pre["image_info"]
            if tuple(size) != (int(info["width"]), int(info["height"])):
                raise ValueError("Native image dimensions differ from the registered Step 00 slide.")
            def crop(box):
                x, y, r, b = box
                try:
                    return slide.read_region((x,y),0,(r-x,b-y)).convert("RGB")
                except openslide.OpenSlideError as exc:
                    raise OSError(f"OpenSlide could not read crop {box}: {exc}") from exc
            yield crop, size, affine, {"kind":"Step 00 registered native slide",
                                       "preflight_image_path":pre.get("image_path"), "pyramid_level":0}


def _aligned_crop(reader, size, affine, center, width, mpp):
    # Output columns follow 10x x=image column=Proseg y; rows follow 10x y=Proseg x.
    side = int(np.clip(np.ceil(width / mpp), 192, 960))
    spacing = width / side
    # PIL affine transform is evaluated at output pixel centers (j+1/2, i+1/2).
    # 10x and OpenSlide pixel coordinates identify pixel centers; translate by +1/2 for PIL.
    start = (center[::-1] - width/2) / mpp
    linear = affine[:,:2] * (spacing / mpp)
    origin = affine[:,:2] @ start + affine[:,2] + 0.5
    edge = np.array([[0,0],[0,side],[side,0],[side,side]])
    corners = edge @ linear.T + origin
    lo = np.floor(corners.min(axis=0)).astype(int)-2
    hi = np.ceil(corners.max(axis=0)).astype(int)+2
    lo = np.maximum(lo,0)
    hi = np.minimum(hi,size)
    if (hi <= lo).any():
        raise ValueError("The requested crop is entirely outside the registered H&E image.")
    if np.prod((hi-lo).astype(float)) > 16_000_000:
        raise ValueError("This crop exceeds 16 MP at the source resolution. Reduce the crop width.")
    box = tuple(map(int, [*lo,*hi]))
    coeff = np.column_stack([linear, origin - lo]).ravel()
    with reader(box) as patch:
        result = patch.transform((side,side), Image.Transform.AFFINE, coeff,
                                 Image.Resampling.BILINEAR, fillcolor="white")
    return result, {"source_box_xyxy_pixels":list(box), "raster_side_px":side,
                    "partly_outside_image":bool((corners.min(axis=0) < 0).any() or
                                                 (corners.max(axis=0) > size).any())}


def _draw(output, region, center, width, raster, tile, qc, macrophages):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spots = tile["spots"]
    xy = spots[["x_um","y_um"]].to_numpy()
    sel = np.flatnonzero(_inside(xy,center,width))
    if not len(sel):
        raise ValueError("Crop has no observed native square centers.")
    # x coordinates increase to the right; y coordinates increase down the image.
    yy, xx = (xy[sel]-center).T
    outside = spots.cell_index.to_numpy()[sel] < 0
    q = qc[sel]
    mac = macrophages[_inside(macrophages, center, width)]
    my, mx = (mac-center).T
    ghost = outside & (q == GHOST)
    stats = {"native_squares":len(sel), "out_of_mask_squares":int(outside.sum()),
             "low_he_high_rna_out_of_mask_squares":int(ghost.sum()), "previous_macrophage_monocyte_centroids":len(mac),
             "qc_counts":{str(k):int(v) for k,v in zip(*np.unique(q, return_counts=True))}, "genes":{}}

    fig, axes = plt.subplots(2,3,figsize=(13.8,9.8))
    fig.subplots_adjust(top=.887,bottom=.10,left=.024,right=.983,hspace=.15,wspace=.08)
    half = width/2
    axes = axes.ravel()
    for i,ax in enumerate(axes):
        ax.imshow(raster, extent=(-half,half,half,-half), origin="upper", interpolation="nearest")
        ax.set(xlim=(-half,half),ylim=(half,-half),aspect="equal")
        ax.set_axis_off()
        if i in (2,3,4,5) and len(mac):
            ax.scatter(mx,my,s=32,facecolors="none",edgecolors="#2d50aa",linewidths=.75,zorder=5)
    axes[0].set_title("Registered H&E")
    length = max(n for n in [width/6, 5, 10, 20, 25, 50, 100, 200] if n <= width/3)
    if length == width/6:
        length = float(f"{length:.1g}")
    x0,y0 = -half+.06*width,half-.065*width
    axes[0].plot([x0,x0+length],[y0,y0],color="white",lw=5,solid_capstyle="butt")
    axes[0].plot([x0,x0+length],[y0,y0],color="#1c1c1c",lw=2.2,solid_capstyle="butt")
    axes[0].text(x0,y0-.025*width,f"{length:g} µm",fontsize=9,bbox=dict(facecolor="white",alpha=.87,linewidth=0,pad=2))
    axes[1].set_title("Four QC classes · all native centers")
    # Marker diameter approximates native square pitch; each point represents its center only.
    dot = float(np.clip((3.4*72*1.55/width)**2, 1, 28))
    class_colors = {k:v[0] for k,v in CLASSES.items()}
    for k in np.unique(q):
        use = q == k
        axes[1].scatter(xx[use],yy[use],s=dot,c=class_colors.get(k,"#454545"),linewidths=0,alpha=.80)
    axes[2].set_title(f"Low H&E, high RNA outside masks · {ghost.sum():,}")
    axes[2].scatter(xx[ghost],yy[ghost],s=dot*1.3,c=CLASSES[GHOST][0],linewidths=0,alpha=.88)
    symbols = np.asarray(tile["genes"]["gene_symbol"].astype(str))
    marker_colors = np.array(["#fdbc39","#ec681d","#ae144b","#4c125b"])
    for ax,gene in zip(axes[3:],MARKERS):
        ids = np.flatnonzero(symbols == gene)
        if not len(ids):
            ax.set_title(f"{gene} · unassayed")
            stats["genes"][gene] = {"assayed":False}
            continue
        # Sum feature IDs with the same gene symbol before comparing marker abundances.
        v = np.asarray(tile["counts"][ids][:,sel].sum(axis=0)).ravel()
        show = outside & (v>0)
        for mask, marker in [(show & ~ghost, "o"), (show & ghost, "s")]:
            colors = marker_colors[np.digitize(v[mask],[2,3,5])]
            ax.scatter(xx[mask],yy[mask],s=dot*2,c=colors,marker=marker,edgecolors="#452a27",linewidths=.22)
        ax.set_title(f"{gene} outside masks · {int(v[show].sum()):,} UMIs")
        stats["genes"][gene] = {"assayed":True,"out_of_mask_UMIs":int(v[show].sum()),"positive_squares":int(show.sum()),
                             "low_he_high_rna_out_of_mask_UMIs":int(v[ghost].sum())}

    def point(color,label,open_circle=False):
        return Line2D([],[],linestyle="none",marker="o",markersize=7,markerfacecolor="none" if open_circle else color,
                      markeredgecolor=color,label=label)
    qc_keys = [point(c,label) for c,label in CLASSES.values()]
    if not set(np.unique(q)).issubset(CLASSES):
        qc_keys.append(point("#454545","Other / unavailable QC"))
    fig.legend(handles=qc_keys,loc="lower center",bbox_to_anchor=(.5,.049),ncol=len(qc_keys),frameon=False,fontsize=9)
    marker_keys = [point(c,label) for c,label in zip(marker_colors,["1 UMI","2 UMIs","3–4 UMIs","≥5 UMIs"])]
    marker_keys.append(point("#2d50aa","Macrophage/monocyte centroid",True))
    fig.legend(handles=marker_keys,loc="lower center",bbox_to_anchor=(.5,.018),ncol=5,frameon=False,fontsize=9)
    fig.text(.5,.011,"Squares identify low-H&E/high-RNA marker sites. Filled circles mark other out-of-mask marker sites.",
             ha="center",fontsize=9)
    fig.suptitle(f"{tile['meta']['sample']}   |   {region}",fontsize=16,fontweight="semibold",y=.975)
    fig.text(.5,.932,f"{width:g}-µm field   ·   Proseg center ({center[0]:.1f}, {center[1]:.1f}) µm"
             f"   ·   {len(mac)} previously labeled macrophage/monocyte centroids",ha="center",fontsize=10)
    try:
        fig.savefig(output,dpi=170,bbox_inches="tight",facecolor="white")
    finally:
        plt.close(fig)
    return stats


def write_he_overlays(tile, qc, macrophages_xy, output_dir, *, image_path=None, centers_um=None, width_um=160.0):
    """Save morphology, four QC classes and three raw out-of-mask marker maps on identical crops.

    `image_path` identifies a relocated copy of the Step 00 registered native slide.
    `centers_um` uses Proseg order (image row in µm, image column in µm).
    """
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    qc = np.asarray(qc,dtype=str)
    macrophages = np.asarray(macrophages_xy,dtype=float).reshape(-1,2)
    if qc.shape != (len(tile["spots"]),) or not np.isfinite(macrophages).all():
        raise ValueError("Supply aligned native QC classes and finite macrophage centroids.")
    regions, width = _regions(tile,qc,macrophages,centers_um,float(width_um))
    meta = tile["meta"]
    mpp = float(meta["microns_per_pixel"])
    receipt = {"state":"unavailable","sample":meta["sample"],"source_fullres_um_per_px":mpp,
               "selection":("user centers" if centers_um is not None else
                            "Most labeled macrophage/monocyte centroids among core crops with at least 80% of maximum in-tissue square coverage. The comparison uses the eligible nonoverlapping crop with the fewest centroids. Ties favor comparable coverage. A third nonoverlapping crop maximizes the number of out-of-mask, low-H&E/high-RNA squares when any remain."),
               "coordinates":"Proseg x follows image rows and Proseg y follows image columns in µm. The display increases rightward along 10x columns and downward along 10x rows.",
               "width_um":width,"regions":[],"attempts":[]}
    for kind,path,pre in _sources(meta,image_path):
        try:
            with _open_source(kind,path,pre) as (reader,size,affine,details):
                rasters = [_aligned_crop(reader,size,affine,c,width,mpp) for c,_ in regions]
        except (ImportError, OSError, ValueError, KeyError, RuntimeError, Image.DecompressionBombError) as exc:
            receipt["attempts"].append({"kind":kind,"path":str(path),"reason":str(exc) or type(exc).__name__})
            continue
        receipt["image"] = dict(details,path=str(path),source_image_size_xy=list(size),
                                fullres_xy_to_source_xy=affine.ravel().tolist())
        for k,((center,name),(raster,reg)) in enumerate(zip(regions,rasters),1):
            fname = f"he_overlay_{k:02d}.png"
            try:
                counts = _draw(out/fname,name,center,width,raster,tile,qc,macrophages)
            finally:
                raster.close()
            receipt["regions"].append(dict(reg,field=name,center_um=center.tolist(),figure=fname,**counts))
        receipt["state"]="complete"
        break
    if receipt["state"] != "complete":
        receipt["recovery"] = ("Mount the registered Space Ranger hires PNG together with its scalefactors JSON, "
                               "or mount the Step 00 SVS and use a JupyterLab terminal to install its native reader using "
                               f"uv pip install --python {shlex.quote(sys.executable)} openslide-python openslide-bin. "
                               "For a relocated SVS set image_path to its current path.")
    (out/"he_overlay_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
