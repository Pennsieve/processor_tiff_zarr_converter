"""
OME-Zarr pyramid processor.

Reads OME-TIFF files from INPUT_DIR and writes multi-resolution OME-Zarr
pyramids to OUTPUT_DIR. The full resolution level can optionally be excluded
to keep the output lightweight for web-based tile viewers.
"""

import os
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
        "channel_chunking": os.environ.get("CHANNEL_CHUNKING", "per-channel"),
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


def build_pyramid_levels(image, config):
    """Downsample the image into a list of pyramid levels."""
    initial_ds = config["initial_downsample"]
    min_dim = config["min_dimension"]
    max_levels = config["max_levels"]

    # Apply initial downsample (skip full resolution)
    current = image
    if initial_ds > 1:
        for _ in range(int(np.log2(initial_ds))):
            h, w = current.shape[0], current.shape[1]
            h_even, w_even = h - (h % 2), w - (w % 2)
            current = downscale_local_mean(
                current[:h_even, :w_even], (2, 2, 1)
            ).astype(image.dtype)

    levels = [current]

    while True:
        if max_levels > 0 and len(levels) >= max_levels:
            break
        h, w = current.shape[0], current.shape[1]
        if h <= min_dim and w <= min_dim:
            break
        h_even, w_even = h - (h % 2), w - (w % 2)
        current = downscale_local_mean(
            current[:h_even, :w_even], (2, 2, 1)
        ).astype(image.dtype)
        levels.append(current)

    return levels


def write_ome_zarr(levels, out_path, config, initial_downsample):
    """Write pyramid levels as an OME-Zarr (v3) dataset."""
    tile_size = config["tile_size"]
    cname = config["compression"]
    clevel = config["compression_level"]
    per_channel = config["channel_chunking"] == "per-channel"

    codecs = [BloscCodec(cname=cname, clevel=clevel, shuffle="bitshuffle")]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    store = LocalStore(out_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    num_channels = levels[0].shape[2] if levels[0].ndim == 3 else 1
    c_chunk = 1 if per_channel else num_channels

    datasets = []
    for i, level_data in enumerate(levels):
        h, w = level_data.shape[0], level_data.shape[1]
        c = level_data.shape[2] if level_data.ndim == 3 else 1

        if level_data.ndim == 3:
            data_5d = level_data.transpose(2, 0, 1).reshape(1, c, 1, h, w)
        else:
            data_5d = level_data.reshape(1, 1, 1, h, w)

        root.create_array(
            str(i),
            data=data_5d,
            chunks=(1, c_chunk, 1, tile_size, tile_size),
            compressors=codecs,
            overwrite=True,
        )

        scale_factor = float(initial_downsample * (2 ** i))
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [1.0, 1.0, 1.0, scale_factor, scale_factor],
            }],
        })

        log.info(
            "  Level %d: %d x %d (scale %.0fx)",
            i, w, h, scale_factor,
        )

    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": os.path.basename(out_path).replace(".ome.zarr", ""),
        "axes": axes,
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "description": (
                f"Pyramid with initial_downsample={initial_downsample}. "
                f"Level 0 = {initial_downsample}x downsampled from original."
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

    log.info("Reading image...")
    with tifffile.TiffFile(filepath) as tif:
        page = tif.pages[0]
        image = page.asarray()
    log.info(
        "  Image shape: %s, dtype: %s, tile: %dx%d",
        image.shape, image.dtype, page.tilewidth, page.tilelength,
    )

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    log.info("Building pyramid levels...")
    levels = build_pyramid_levels(image, config)
    log.info("  Generated %d levels", len(levels))

    log.info("Writing OME-Zarr to %s", out_path)
    write_ome_zarr(levels, out_path, config, config["initial_downsample"])
    log.info("Done: %s", out_name)


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
