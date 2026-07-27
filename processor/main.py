"""
OME-Zarr pyramid processor.

Reads OME-TIFF files from INPUT_DIR and writes multi-resolution OME-Zarr
pyramids to OUTPUT_DIR. Handles 2D and 3D (Z-stack) inputs. For 3D inputs,
Z downsampling is chosen from PhysicalSize metadata by default: isotropic
volumes get (2,2,2) downsampling, anisotropic ones keep Z at native res.

Uses pyvips for streaming/tiled processing so peak memory stays low
regardless of image size.
"""

import os
import re
import sys
import logging
import shutil

import numpy as np
import pyvips
pyvips.cache_set_max(0)
logging.getLogger("pyvips").setLevel(logging.WARNING)
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("processor-ome-zarr")

FORMAT_TO_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


def get_config():
    return {
        "input_dir": os.environ.get("INPUT_DIR", ""),
        "output_dir": os.environ.get("OUTPUT_DIR", ""),
        "initial_downsample": int(os.environ.get("INITIAL_DOWNSAMPLE", "2")),
        "tile_size": int(os.environ.get("TILE_SIZE", "256")),
        "compression": os.environ.get("COMPRESSION", "zstd"),
        "compression_level": int(os.environ.get("COMPRESSION_LEVEL", "5")),
        "max_levels": int(os.environ.get("MAX_LEVELS", "0")),
        "min_dimension": int(os.environ.get("MIN_DIMENSION", "256")),
        "channel_chunking": os.environ.get("CHANNEL_CHUNKING", "auto"),
        "z_downsample": os.environ.get("Z_DOWNSAMPLE", "auto"),
        "z_chunk": int(os.environ.get("Z_CHUNK", "16")),
        "rgb_mode": os.environ.get("RGB_MODE", "combined"),
    }


def validate_dirs(config):
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    if not input_dir or not os.path.isdir(input_dir):
        log.error("INPUT_DIR '%s' does not exist or is not set", input_dir)
        return False

    if not output_dir:
        log.error("OUTPUT_DIR is not set")
        return False

    os.makedirs(output_dir, exist_ok=True)
    return True


def find_tiff_files(input_dir):
    extensions = (".ome.tiff", ".ome.tif", ".tiff", ".tif")
    files = []
    for f in sorted(os.listdir(input_dir)):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(input_dir, f))
    return files


def _parse_physical_sizes(ome_xml):
    """Return ((pz, py, px), found) in micrometers; found is a per-axis dict
    indicating whether the value came from OME metadata (True) or was
    defaulted to 1.0 because the attribute was absent (False).
    """
    def grab(key):
        m = re.search(rf'{key}="([\d.]+)"', ome_xml or "")
        if m:
            return float(m.group(1)), True
        return 1.0, False
    pz, fz = grab("PhysicalSizeZ")
    py, fy = grab("PhysicalSizeY")
    px, fx = grab("PhysicalSizeX")
    return (pz, py, px), {"z": fz, "y": fy, "x": fx}


def choose_z_downsample(mode, physical_sizes):
    pz, py, px = physical_sizes
    if mode == "isotropic":
        return True
    if mode == "none":
        return False
    xy = min(py, px)
    return xy > 0 and pz / xy <= 2.0


def open_image(filepath):
    """Open a TIFF lazily with pyvips, extracting metadata only.

    Returns a dict with image info — no pixel data is loaded into memory.
    """
    img = pyvips.Image.new_from_file(filepath, access="sequential")

    width = img.width
    height = img.height
    bands = img.bands
    n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
    interp = img.interpretation

    is_rgb = interp in ("srgb", "rgb", "scrgb")

    ome_xml = None
    if img.get_typeof("image-description"):
        ome_xml = img.get("image-description")

    physical, found = _parse_physical_sizes(ome_xml)

    missing = [ax for ax in ("z", "y", "x") if not found[ax]]
    if missing:
        log.warning(
            "  OME PhysicalSize missing for axes %s; defaulted to 1.0 µm. "
            "Parsed values (z,y,x) µm: (%s, %s, %s)",
            missing, physical[0], physical[1], physical[2],
        )
    else:
        log.info(
            "  OME PhysicalSize (z,y,x) µm from metadata: (%s, %s, %s)",
            physical[0], physical[1], physical[2],
        )

    vips_format = img.format
    dtype = FORMAT_TO_DTYPE.get(vips_format, np.uint8)

    return {
        "filepath": filepath,
        "width": width,
        "height": height,
        "bands": bands,
        "n_pages": n_pages,
        "is_rgb": is_rgb,
        "physical": physical,
        "dtype": dtype,
        "vips_format": vips_format,
    }


def compute_pyramid_plan(image_info, config, z_downsample):
    """Compute shapes, shrink factors, and coordinate scales for all pyramid levels.

    Pure math — no pixel data touched.
    """
    initial_ds = config["initial_downsample"]
    min_dim = config["min_dimension"]
    max_levels = config["max_levels"]
    tile_size = config["tile_size"]
    z_chunk_cfg = config["z_chunk"]
    chunking = config["channel_chunking"]

    width = image_info["width"]
    height = image_info["height"]
    bands = image_info["bands"]
    n_pages = image_info["n_pages"]
    is_rgb = image_info["is_rgb"]
    is_3d = n_pages > 1
    pz, py, px = image_info["physical"]

    if chunking == "auto":
        per_channel = not is_rgb
    else:
        per_channel = chunking == "per-channel"

    c_chunk = 1 if per_channel else bands

    xy_unit = min(py, px) if min(py, px) > 0 else 1.0
    z_ratio = pz / xy_unit

    levels = []
    level = 0
    while True:
        xy_shrink = initial_ds * (2 ** level)
        lw = max(1, width // xy_shrink)
        lh = max(1, height // xy_shrink)

        if level > 0 and lw <= min_dim and lh <= min_dim:
            break
        if max_levels > 0 and level >= max_levels:
            break

        if z_downsample:
            z_shrink = xy_shrink
            lz = max(1, n_pages // z_shrink)
        else:
            z_shrink = 1
            lz = n_pages

        if is_3d:
            shape = (bands, lz, lh, lw)
            chunks = (c_chunk, min(z_chunk_cfg, lz), tile_size, tile_size)
            z_mult = float(xy_shrink) if z_downsample else 1.0
            scale = [1.0, z_ratio * z_mult, float(xy_shrink), float(xy_shrink)]
        else:
            shape = (bands, lh, lw)
            chunks = (c_chunk, tile_size, tile_size)
            scale = [1.0, float(xy_shrink), float(xy_shrink)]

        out_px = px * xy_shrink
        out_py = py * xy_shrink
        out_pz = pz * (xy_shrink if z_downsample else 1.0)

        levels.append({
            "level": level,
            "xy_shrink": xy_shrink,
            "z_shrink": z_shrink,
            "shape": shape,
            "chunks": chunks,
            "scale": scale,
            "lz": lz,
            "lw": lw,
            "lh": lh,
            "out_physical": (out_pz, out_py, out_px),
        })

        level += 1
        # Ensure we generate at least one level (level 0)
        if lw <= min_dim and lh <= min_dim:
            break

    return levels


def write_tiles(img, arr, z_index, tile_size, is_3d):
    """Write one Z-plane of one pyramid level, tile by tile.

    Peak memory: one tile at a time (~256x256x3 = ~192KB for uint8 RGB).
    """
    if is_3d:
        # arr shape: (C, Z, H, W)
        height = arr.shape[2]
        width = arr.shape[3]
    else:
        # arr shape: (C, H, W)
        height = arr.shape[1]
        width = arr.shape[2]

    for ty in range(0, height, tile_size):
        for tx in range(0, width, tile_size):
            tw = min(tile_size, width - tx)
            th = min(tile_size, height - ty)
            tile = img.crop(tx, ty, tw, th)
            tile_np = np.ndarray(
                buffer=tile.write_to_memory(),
                shape=(th, tw, tile.bands),
                dtype=FORMAT_TO_DTYPE.get(tile.format, np.uint8),
            )
            # (H, W, C) → (C, H, W)
            tile_np = tile_np.transpose(2, 0, 1)

            if is_3d:
                arr[:, z_index, ty:ty + th, tx:tx + tw] = tile_np
            else:
                arr[:, ty:ty + th, tx:tx + tw] = tile_np



def write_ome_zarr(image_info, pyramid_plan, out_path, config,
                   z_downsampled, name=None):
    """Write pyramid levels as an OME-Zarr (v3) dataset using pyvips streaming."""
    tile_size = config["tile_size"]
    compression = config["compression"]
    compression_level = config["compression_level"]

    filepath = image_info["filepath"]
    bands = image_info["bands"]
    is_rgb = image_info["is_rgb"]
    dtype = image_info["dtype"]
    vips_format = image_info["vips_format"]
    n_pages = image_info["n_pages"]
    physical = image_info["physical"]
    is_3d = n_pages > 1

    codecs = [BloscCodec(cname=compression, clevel=compression_level,
                         shuffle="shuffle")]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    store = LocalStore(out_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    initial_ds = config["initial_downsample"]
    datasets = []

    # Pre-open source pages with random access so pyvips can reuse them
    # across pyramid levels without re-reading the entire file each time.
    src_pages = {}
    unique_pages = set()
    for plan in pyramid_plan:
        z_shrink = plan["z_shrink"]
        for zi in range(plan["lz"]):
            src_z = zi * z_shrink
            unique_pages.add(src_z)
            if z_downsampled and z_shrink > 1 and src_z + 1 < n_pages:
                unique_pages.add(src_z + 1)
    for page_idx in sorted(unique_pages):
        src_pages[page_idx] = pyvips.Image.new_from_file(
            filepath, page=page_idx, access="random"
        )

    for plan in pyramid_plan:
        level = plan["level"]
        shape = plan["shape"]
        chunks = plan["chunks"]
        scale = plan["scale"]
        xy_shrink = plan["xy_shrink"]
        z_shrink = plan["z_shrink"]
        lz = plan["lz"]
        out_phys = plan["out_physical"]

        arr = root.create_array(
            str(level),
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressors=codecs,
            overwrite=True,
        )

        for zi in range(lz):
            src_z = zi * z_shrink

            if z_downsampled and z_shrink > 1 and src_z + 1 < n_pages:
                avg = (src_pages[src_z] + src_pages[src_z + 1]) / 2
                img = avg.cast(vips_format)
            else:
                img = src_pages[src_z]

            if xy_shrink > 1:
                img = img.shrink(xy_shrink, xy_shrink)

            write_tiles(img, arr, zi, tile_size, is_3d)

        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })

        log.info(
            "  Level %d: shape=%s chunks=%s (XY %.0fx) "
            "voxel µm (z,y,x): (%.4f, %.4f, %.4f) scale=%s",
            level, shape, chunks, float(xy_shrink),
            out_phys[0], out_phys[1], out_phys[2], scale,
        )

    if is_3d:
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    else:
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": name or os.path.basename(out_path),
        "axes": axes,
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "description": (
                f"Pyramid with initial_downsample={initial_ds}, "
                f"z_downsampled={z_downsampled}."
            ),
        },
    }]

    if bands == 3:
        root.attrs["omero"] = {
            "channels": [
                {"color": "FF6699", "label": "R", "active": True,
                 "window": {"start": 0, "end": 255}},
                {"color": "00FF00", "label": "G", "active": True,
                 "window": {"start": 0, "end": 255}},
                {"color": "6699FF", "label": "B", "active": True,
                 "window": {"start": 0, "end": 255}},
            ],
            "rdefs": {"model": "color" if config.get("rgb_mode") == "combined" else "fluorescence"},
        }


def process_file(filepath, output_dir, config):
    """Process a single TIFF file into OME-Zarr."""
    basename = os.path.basename(filepath)
    log.info("Processing: %s", basename)

    out_path = output_dir

    zarr_name = basename
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if basename.lower().endswith(ext):
            zarr_name = basename[: len(basename) - len(ext)]
            break

    log.info("Opening image (lazy)...")
    image_info = open_image(filepath)

    z_ds = choose_z_downsample(config["z_downsample"], image_info["physical"])
    is_3d = image_info["n_pages"] > 1

    log.info(
        "  Image: %dx%d, %d bands, %d pages, dtype: %s, photometric: %s, "
        "physical (z,y,x) µm: %s, z_downsample: %s",
        image_info["width"], image_info["height"],
        image_info["bands"], image_info["n_pages"],
        image_info["dtype"].__name__,
        "RGB" if image_info["is_rgb"] else "grayscale",
        image_info["physical"], z_ds,
    )

    log.info("Computing pyramid plan...")
    pyramid_plan = compute_pyramid_plan(image_info, config, z_ds)
    log.info("  Planned %d levels", len(pyramid_plan))

    log.info("Writing OME-Zarr to %s (streaming tiles)...", out_path)
    write_ome_zarr(image_info, pyramid_plan, out_path, config,
                   z_ds, name=zarr_name)
    log.info("Done: %s", out_path)


def run():
    config = get_config()

    log.info("=== OME-Zarr Processor ===")
    log.info("Config:")
    for k, v in config.items():
        log.info("  %s = %s", k, v)

    if not validate_dirs(config):
        sys.exit(1)

    tiff_files = find_tiff_files(config["input_dir"])
    if not tiff_files:
        log.error("No TIFF files found in %s", config["input_dir"])
        sys.exit(1)

    log.info("Found %d TIFF file(s)", len(tiff_files))

    for filepath in tiff_files:
        process_file(filepath, config["output_dir"], config)

    log.info("=== Processing complete ===")


if __name__ == "__main__":
    run()
