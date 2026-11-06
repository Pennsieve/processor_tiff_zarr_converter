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
import xml.etree.ElementTree as ET

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
        "initial_downsample": int(os.environ.get("INITIAL_DOWNSAMPLE", "1")),
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


FALLBACK_CHANNEL_COLORS = ["00FF00", "FF00FF", "00FFFF", "FF0000", "0000FF", "FFFF00"]


def ome_color_to_hex(value):
    """Convert an OME Color attribute (signed 32-bit RGBA int) to 'RRGGBB'."""
    try:
        u = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    return f"{(u >> 24) & 255:02X}{(u >> 16) & 255:02X}{(u >> 8) & 255:02X}"


def parse_ome_pixels(ome_xml):
    """Read the dimension layout and channel metadata from OME-XML.

    Returns None if the description isn't OME-XML (e.g. ImageJ or plain TIFF).
    """
    if not ome_xml or "<OME" not in ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        log.warning("  Image description looks like OME-XML but could not be parsed")
        return None
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return None

    def size(key):
        try:
            return max(1, int(pixels.get(key, "1")))
        except ValueError:
            return 1

    channels = [
        {
            "name": ch.get("Name") or ch.get("Fluor"),
            "color": ome_color_to_hex(ch.get("Color")) if ch.get("Color") else None,
        }
        for ch in pixels.findall("{*}Channel")
    ]
    return {
        "size_c": size("SizeC"),
        "size_z": size("SizeZ"),
        "size_t": size("SizeT"),
        "dimension_order": pixels.get("DimensionOrder", "XYCZT").upper(),
        "channels": channels,
    }


def ome_page_index(dimension_order, c, z, t, size_c, size_z, size_t):
    """TIFF page holding plane (c, z, t), per OME DimensionOrder.

    In e.g. "XYCZT" the letter after XY varies fastest: pages go c0z0, c1z0, c0z1, ...
    """
    sizes = {"C": size_c, "Z": size_z, "T": size_t}
    index = {"C": c, "Z": z, "T": t}
    page, stride = 0, 1
    for dim in dimension_order[2:]:
        page += index[dim] * stride
        stride *= sizes[dim]
    return page


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

    # Work out how channels and Z planes are laid out across TIFF pages.
    # Default (RGB, plain TIFF, single-channel stacks): every page is one Z
    # plane and channels are the bands within a page.
    ome = parse_ome_pixels(ome_xml)
    size_c, size_z = bands, n_pages
    page_map = {(None, z): z for z in range(n_pages)}
    planar_channels = False
    if ome and not is_rgb and bands == 1 and ome["size_c"] > 1:
        expected = ome["size_c"] * ome["size_z"] * ome["size_t"]
        if n_pages == expected:
            planar_channels = True
            size_c, size_z = ome["size_c"], ome["size_z"]
            if ome["size_t"] > 1:
                log.warning("  SizeT=%d; only the first timepoint is converted", ome["size_t"])
            page_map = {
                (c, z): ome_page_index(ome["dimension_order"], c, z, 0,
                                       ome["size_c"], ome["size_z"], ome["size_t"])
                for c in range(size_c) for z in range(size_z)
            }
            log.info(
                "  OME multichannel: SizeC=%d SizeZ=%d DimensionOrder=%s, one page per plane",
                size_c, size_z, ome["dimension_order"],
            )
        else:
            log.warning(
                "  OME says SizeC=%d SizeZ=%d SizeT=%d (%d planes) but the file has %d pages; "
                "treating pages as Z", ome["size_c"], ome["size_z"], ome["size_t"], expected, n_pages,
            )

    channel_meta = (ome or {}).get("channels", [])

    return {
        "filepath": filepath,
        "width": width,
        "height": height,
        "bands": bands,
        "n_pages": n_pages,
        "size_c": size_c,
        "size_z": size_z,
        "planar_channels": planar_channels,
        "page_map": page_map,
        "channel_meta": channel_meta,
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
    bands = image_info["size_c"]
    n_pages = image_info["size_z"]
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


def write_strips(imgs, arr, z0, tile_size, is_3d, channel=None):
    """Write consecutive Z-planes of one pyramid level, one row of tiles at a time.

    imgs are the planes for slices z0, z0+1, ... (a single plane for 2D). Each
    write covers whole chunks (all of their slices, a full row of tiles), so no
    chunk is written twice. channel=None writes all of a plane's bands (RGB /
    interleaved); an int writes its single band into that channel (planar OME
    multichannel). Peak memory: one row of tiles per plane, e.g. 256 rows x
    26k px x 16 slices x 2 bytes = ~210 MB for a large 16-bit stack.
    """
    c_slice = slice(None) if channel is None else slice(channel, channel + 1)
    height, width = arr.shape[-2], arr.shape[-1]
    dtype = FORMAT_TO_DTYPE.get(imgs[0].format, np.uint8)

    for ty in range(0, height, tile_size):
        th = min(tile_size, height - ty)
        rows = []
        for img in imgs:
            strip = img.crop(0, ty, width, th)
            # (H, W, C) -> (C, H, W)
            rows.append(np.ndarray(
                buffer=strip.write_to_memory(),
                shape=(th, width, strip.bands),
                dtype=dtype,
            ).transpose(2, 0, 1))

        block = np.stack(rows, axis=1) if is_3d else rows[0]
        # One chunk per assignment. Assigning the whole row lets zarr write its
        # chunks from parallel threads into the same directory, and on EFS that
        # intermittently failed with FileNotFoundError when renaming a chunk's
        # temporary .partial file into place.
        for tx in range(0, width, tile_size):
            tw = min(tile_size, width - tx)
            if is_3d:
                # arr shape: (C, Z, H, W); the block holds all slices of the chunk
                arr[c_slice, z0:z0 + len(rows), ty:ty + th, tx:tx + tw] = block[..., tx:tx + tw]
            else:
                # arr shape: (C, H, W)
                arr[c_slice, ty:ty + th, tx:tx + tw] = block[..., tx:tx + tw]


def write_ome_zarr(image_info, pyramid_plan, out_path, config,
                   z_downsampled, name=None):
    """Write pyramid levels as an OME-Zarr (v3) dataset using pyvips streaming."""
    tile_size = config["tile_size"]
    compression = config["compression"]
    compression_level = config["compression_level"]

    filepath = image_info["filepath"]
    dtype = image_info["dtype"]
    vips_format = image_info["vips_format"]
    n_pages = image_info["size_z"]
    page_map = image_info["page_map"]
    is_3d = n_pages > 1
    # Planar multichannel: each channel is its own page. Otherwise a page
    # carries all channels as bands (key None).
    channel_keys = (list(range(image_info["size_c"]))
                    if image_info["planar_channels"] else [None])

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
            for ck in channel_keys:
                unique_pages.add(page_map[(ck, src_z)])
                if z_downsampled and z_shrink > 1 and src_z + 1 < n_pages:
                    unique_pages.add(page_map[(ck, src_z + 1)])
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

        def plane(ck, zi):
            """The source plane for output slice zi, shrunk to this level."""
            src_z = zi * z_shrink
            page = src_pages[page_map[(ck, src_z)]]
            if z_downsampled and z_shrink > 1 and src_z + 1 < n_pages:
                nxt = src_pages[page_map[(ck, src_z + 1)]]
                img = ((page + nxt) / 2).cast(vips_format)
            else:
                img = page
            if xy_shrink > 1:
                img = img.shrink(xy_shrink, xy_shrink)
            return img

        # Write one Z-chunk's worth of slices at a time, so every chunk is
        # written exactly once instead of once per slice (each partial write
        # is a read-decompress-recompress-write, slow on network storage).
        z_step = chunks[1] if is_3d else 1
        for z0 in range(0, lz, z_step):
            z1 = min(z0 + z_step, lz)
            for ck in channel_keys:
                imgs = [plane(ck, zi) for zi in range(z0, z1)]
                write_strips(imgs, arr, z0, tile_size, is_3d, channel=ck)

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

    smallest = root[datasets[-1]["path"]]
    omero = build_omero(image_info, smallest, config)
    if omero:
        root.attrs["omero"] = omero


def build_omero(image_info, smallest_level, config):
    """OMERO rendering metadata: channel colors, labels and display windows.

    RGB keeps the fixed R/G/B block (the viewer and thumbnail processor detect
    RGB from it). Planar OME multichannel images get one entry per channel,
    with the color and name from the OME-XML and a contrast window from the
    smallest pyramid level, so the viewer shows them as colored overlays.
    A single-channel image gets one such entry when the OME-XML gives it a
    color other than white, so it shows in that color instead of gray.
    """
    if not image_info["planar_channels"]:
        if image_info["size_c"] == 1 and not image_info["is_rgb"]:
            return _single_channel_omero(image_info, smallest_level)
        if image_info["size_c"] != 3:
            return None
        return {
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

    size_c = image_info["size_c"]
    meta = image_info["channel_meta"]
    colors = [(meta[c].get("color") if c < len(meta) else None) for c in range(size_c)]
    # Black is invisible as an overlay, and a repeated color (often all white)
    # can't be told apart, so only those channels get a fallback color, one
    # not already used by another channel. Good OME colors are kept.
    original = list(colors)
    seen = set()
    for c, col in enumerate(colors):
        if col in (None, "000000") or col in seen:
            colors[c] = None
        else:
            seen.add(col)
    spare = [col for col in FALLBACK_CHANNEL_COLORS if col not in seen]
    for c in range(size_c):
        if colors[c] is None:
            colors[c] = spare.pop(0) if spare else FALLBACK_CHANNEL_COLORS[c % len(FALLBACK_CHANNEL_COLORS)]
    if colors != original:
        log.info("  Replaced missing/black/duplicate OME colors: %s -> %s", original, colors)

    channels = [_omero_channel(image_info, smallest_level, c, colors[c])
                for c in range(size_c)]
    # No rdefs.model here: the viewer reads model "color" with 3 channels as RGB.
    return {"channels": channels}


def _single_channel_omero(image_info, smallest_level):
    """One colored channel when the OME-XML gives a real color, else None.

    White (or a missing/black color) means plain grayscale, which the viewer
    and thumbnail already show without an omero block.
    """
    meta = image_info["channel_meta"]
    color = meta[0].get("color") if meta else None
    if color in (None, "000000", "FFFFFF"):
        return None
    return {"channels": [_omero_channel(image_info, smallest_level, 0, color)]}


def _omero_channel(image_info, smallest_level, c, color):
    """One omero channel entry: color, OME name and a 0.1-99.9% contrast window
    measured on the smallest pyramid level (array axes are c[,z],y,x)."""
    dtype = np.dtype(image_info["dtype"])
    if np.issubdtype(dtype, np.integer):
        dmin, dmax = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
    else:
        dmin, dmax = 0.0, 1.0

    if smallest_level.ndim == 4:
        nz = smallest_level.shape[1]
        values = np.asarray(smallest_level[c, ::max(1, nz // 16)])
    else:
        values = np.asarray(smallest_level[c])
    lo, hi = (np.percentile(values, [0.1, 99.9]) if values.size else (dmin, dmax))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1
    meta = image_info["channel_meta"]
    name = meta[c].get("name") if c < len(meta) else None
    entry = {
        "color": color,
        "label": name or f"Channel {c + 1}",
        "active": True,
        "window": {"min": dmin, "max": dmax, "start": lo, "end": hi},
    }
    log.info("  Channel %d '%s': color #%s window [%g, %g]", c, entry["label"], color, lo, hi)
    return entry


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
    if image_info["is_rgb"]:
        layout = "RGB"
    elif image_info["planar_channels"]:
        layout = "multichannel (one page per channel)"
    else:
        layout = "grayscale"

    log.info(
        "  Image: %dx%d, %d bands, %d pages -> %d channels, %d z, dtype: %s, "
        "layout: %s, physical (z,y,x) µm: %s, z_downsample: %s",
        image_info["width"], image_info["height"],
        image_info["bands"], image_info["n_pages"],
        image_info["size_c"], image_info["size_z"],
        image_info["dtype"].__name__, layout,
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
