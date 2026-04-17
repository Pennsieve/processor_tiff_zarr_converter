"""
OME-Zarr pyramid processor.

Reads OME-TIFF files from INPUT_DIR and writes multi-resolution OME-Zarr
pyramids to OUTPUT_DIR. Handles 2D and 3D (Z-stack) inputs. For 3D inputs,
Z downsampling is chosen from PhysicalSize metadata by default: isotropic
volumes get (2,2,2) downsampling, anisotropic ones keep Z at native res.
"""

import os
import re
import sys
import logging
import shutil

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore
from skimage.transform import downscale_local_mean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("processor-ome-zarr")


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
    """Return (pz, py, px) in micrometers, defaulting to 1.0 when absent."""
    def grab(key):
        m = re.search(rf'{key}="([\d.]+)"', ome_xml or "")
        return float(m.group(1)) if m else 1.0
    return grab("PhysicalSizeZ"), grab("PhysicalSizeY"), grab("PhysicalSizeX")


def read_volume(filepath):
    """Read TIFF into a (Z, Y, X, C) volume with physical voxel sizes."""
    with tifffile.TiffFile(filepath) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = series.axes
        page = tif.pages[0]
        is_rgb = page.photometric == tifffile.PHOTOMETRIC.RGB
        physical = _parse_physical_sizes(tif.ome_metadata)

    for ax in "ZYXC":
        has_channel_as_samples = ax == "C" and "S" in axes
        if ax not in axes and not has_channel_as_samples:
            arr = np.expand_dims(arr, 0)
            axes = ax + axes
    axes = axes.replace("S", "C")
    order = [axes.index(a) for a in "ZYXC"]
    return arr.transpose(order), physical, is_rgb


def choose_z_downsample(mode, physical_sizes):
    pz, py, px = physical_sizes
    if mode == "isotropic":
        return True
    if mode == "none":
        return False
    xy = min(py, px)
    return xy > 0 and pz / xy <= 2.0


def build_pyramid_levels(volume, config, z_downsample):
    """volume is (Z, Y, X, C). Returns a list of pyramid levels."""
    initial_ds = config["initial_downsample"]
    min_dim = config["min_dimension"]
    max_levels = config["max_levels"]
    factors = (2 if z_downsample else 1, 2, 2, 1)

    def halve(arr):
        cropped = tuple(
            slice(0, arr.shape[i] - (arr.shape[i] % factors[i]))
            for i in range(arr.ndim)
        )
        return downscale_local_mean(arr[cropped], factors).astype(volume.dtype)

    current = volume
    if initial_ds > 1:
        for _ in range(int(np.log2(initial_ds))):
            current = halve(current)

    levels = [current]
    while True:
        if max_levels > 0 and len(levels) >= max_levels:
            break
        h, w = current.shape[1], current.shape[2]
        if h <= min_dim and w <= min_dim:
            break
        current = halve(current)
        levels.append(current)

    return levels


def write_ome_zarr(levels, out_path, config, initial_downsample,
                   physical_sizes, z_downsampled, name=None, is_rgb=False):
    """Write pyramid levels as an OME-Zarr (v3) dataset."""
    tile_size = config["tile_size"]
    chunking = config["channel_chunking"]
    z_chunk_cfg = config["z_chunk"]

    num_channels = levels[0].shape[3]
    is_3d = levels[0].shape[0] > 1

    if chunking == "auto":
        per_channel = not is_rgb
        log.info("  Auto channel chunking: %s (is_rgb=%s)",
                 "bundled" if is_rgb else "per-channel", is_rgb)
    else:
        per_channel = chunking == "per-channel"

    codecs = [BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    store = LocalStore(out_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    c_chunk = 1 if per_channel else num_channels
    pz, py, px = physical_sizes
    xy_unit = min(py, px) if min(py, px) > 0 else 1.0
    z_ratio = pz / xy_unit

    datasets = []
    for i, level in enumerate(levels):
        if is_3d:
            data = level.transpose(3, 0, 1, 2)  # (Z,Y,X,C) → (C,Z,Y,X)
            chunks = (c_chunk, min(z_chunk_cfg, data.shape[1]),
                      tile_size, tile_size)
        else:
            data = level.transpose(3, 0, 1, 2).squeeze(1)  # → (C,Y,X)
            chunks = (c_chunk, tile_size, tile_size)

        root.create_array(
            str(i),
            data=data,
            chunks=chunks,
            compressors=codecs,
            overwrite=True,
        )

        xy_scale = float(initial_downsample * (2 ** i))
        z_mult = xy_scale if z_downsampled else 1.0
        if is_3d:
            scale = [1.0, z_ratio * z_mult, xy_scale, xy_scale]
        else:
            scale = [1.0, xy_scale, xy_scale]

        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })

        log.info("  Level %d: shape=%s chunks=%s (XY %.0fx)",
                 i, data.shape, chunks, xy_scale)

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
                f"Pyramid with initial_downsample={initial_downsample}, "
                f"z_downsampled={z_downsampled}."
            ),
        },
    }]

    if num_channels == 3:
        root.attrs["omero"] = {
            "channels": [
                {"color": "FF0000", "label": "R", "active": True,
                 "window": {"start": 0, "end": 255}},
                {"color": "00FF00", "label": "G", "active": True,
                 "window": {"start": 0, "end": 255}},
                {"color": "0000FF", "label": "B", "active": True,
                 "window": {"start": 0, "end": 255}},
            ],
            "rdefs": {"model": "color"},
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

    log.info("Reading image...")
    volume, physical, is_rgb = read_volume(filepath)
    z_ds = choose_z_downsample(config["z_downsample"], physical)
    log.info(
        "  Volume (Z,Y,X,C): %s, dtype: %s, photometric: %s, "
        "physical (z,y,x) µm: %s, z_downsample: %s",
        volume.shape, volume.dtype,
        "RGB" if is_rgb else "grayscale", physical, z_ds,
    )

    log.info("Building pyramid levels...")
    levels = build_pyramid_levels(volume, config, z_ds)
    log.info("  Generated %d levels", len(levels))

    log.info("Writing OME-Zarr to %s", out_path)
    write_ome_zarr(levels, out_path, config, config["initial_downsample"],
                   physical, z_ds, name=zarr_name, is_rgb=is_rgb)
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