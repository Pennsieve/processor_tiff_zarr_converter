# processor_tiff_zarr_converter

Pennsieve processor that converts OME-TIFF files to multi-resolution OME-Zarr pyramids optimized for web-based tile viewers.

## What it does

- Reads `.ome.tiff` / `.tiff` files from `INPUT_DIR`
- Generates a downsampled multi-resolution pyramid (configurable initial downsample to exclude full resolution)
- Writes OME-Zarr v0.4 (Zarr v3) datasets to `OUTPUT_DIR`
- Per-channel chunking for independent channel access in viewers
- Blosc/zstd compression for efficient storage and serving
- Multichannel OME-TIFFs that store one page per channel (typical fluorescence/confocal exports) are read using the OME-XML `SizeC`, `SizeZ` and `DimensionOrder`, so channels go on the `c` axis rather than being mistaken for Z slices

## Image layouts

| Layout | Detected by | Output |
|---|---|---|
| RGB (brightfield) | TIFF photometric RGB, 3 samples per pixel | `c=3`, fixed R/G/B `omero` block with `rdefs.model` from `RGB_MODE` |
| Multichannel, one page per channel | OME-XML `SizeC > 1` and page count = `SizeC × SizeZ × SizeT` | `c=SizeC`, `z=SizeZ`; `omero` channels with the OME `Color` and `Name`, and a contrast window from the 0.1–99.9th percentile of the smallest level |
| Single channel / plain TIFF | anything else | each page is a Z slice; a one-channel `omero` block with the OME color, name and window when the OME-XML gives a color other than white, otherwise none (shown in grayscale) |

For multichannel images, the OME colors are kept; only a channel whose color is missing, black, or repeats an earlier channel gets a fallback color not already in use. Only the first timepoint is converted when `SizeT > 1`. No `rdefs.model` is written for multichannel images, because viewers treat `model: "color"` with three channels as RGB.

## Environment Variables

### Platform Variables (set by Pennsieve)

| Variable | Source (ECS) | Source (Lambda) | Description |
|---|---|---|---|
| `INPUT_DIR` | Container override | Payload (`inputDir`) | The input files, on the step's local disk (`/data/input` on ECS, under `/tmp` on Lambda) |
| `OUTPUT_DIR` | Container override | Payload (`outputDir`) | Where to write the OME-Zarr, on the same local disk (`/data/output` on ECS); uploaded when the step ends |
| `WORKFLOW_INSTANCE_ID` | Container override | Payload (`workflowInstanceId`) | Unique workflow execution identifier |
| `SESSION_TOKEN` | Container override | Payload (`sessionToken`) | Cognito access token |
| `REFRESH_TOKEN` | Container override | Payload (`refreshToken`) | Token refresh credential |
| `PENNSIEVE_API_HOST` | Container override | Lambda config | API v1 endpoint |
| `PENNSIEVE_API_HOST2` | Container override | Lambda config | API v2 endpoint |
| `ENVIRONMENT` | Container override | Lambda config | Environment name (dev/prod) |
| `REGION` | Container override | Lambda config | AWS region |
| `DEPLOYMENT_MODE` | Container override | Lambda config | Security scope (basic/secure/compliant) |

### Disk

A step runs on its own local disk, not on shared storage. The compute node sizes it from the inputs: the TIFF, the image, about twice the TIFF for what the step writes, and headroom. Over 200 GB the step gets its own EBS volume. The OME-Zarr pyramid is a little larger than a compressed TIFF, so the plan normally fits; a step that still runs out of disk (`OSError: [Errno 28] No space left on device`) can be given a floor: the **Disk** field in the run setup on the run page, or `runtime.diskGiB` in `app.yml` for every run of this app.

### Processor Parameters (set on Lambda config or container override; the ones in `app.yml` can be set on the workflow node)

| Variable | Default | Description |
|---|---|---|
| `INITIAL_DOWNSAMPLE` | `1` | Initial downsample factor. `1` = include full resolution (default), `2` = start from 2x downsampled, `4` = start from 4x, etc. Must be a power of 2. |
| `TILE_SIZE` | `256` | Chunk/tile size in pixels (both x and y). Standard for web tile viewers. |
| `COMPRESSION` | `zstd` | Blosc compression algorithm (`zstd`, `lz4`, `lz4hc`, `zlib`, `snappy`) |
| `COMPRESSION_LEVEL` | `5` | Compression level (1-9, higher = smaller but slower) |
| `MAX_LEVELS` | `0` | Maximum number of pyramid levels. `0` = auto (generate until min dimension reached) |
| `MIN_DIMENSION` | `256` | Stop generating levels when both dimensions are at or below this value |
| `CHANNEL_CHUNKING` | `auto` | `auto` = bundled for RGB images, per-channel for grayscale (detected from TIFF photometric tag). `per-channel` = each channel is an independent chunk. `bundled` = all channels in one chunk. |

## Output Format

Each input file `image.ome.tiff` produces `image.ome.zarr/`:

```
image.ome.zarr/
├── zarr.json            # Root group metadata (multiscales + OMERO)
├── 0/                   # Highest resolution level (after initial downsample)
│   ├── zarr.json        # Array metadata (dtype, chunks, codecs)
│   └── c/0/0/y/x        # Chunks (per-channel, tiled)
├── 1/                   # 2x further downsampled
├── 2/                   # 4x further downsampled
└── ...
```

Metadata includes:
- **multiscales**: OME-Zarr v0.4 with `coordinateTransformations` encoding scale relative to the original full-resolution image
- **omero**: channel rendering hints: fixed R/G/B for RGB images, or per-channel colors, names and windows from the OME-XML for multichannel images

## Dual-Mode Support

Runs in both ECS (Fargate) and Lambda modes via a runtime-detecting entrypoint:

- **ECS**: `python -m processor.main` reads env vars directly from container overrides
- **Lambda**: The `entrypoint.sh` detects `AWS_LAMBDA_RUNTIME_API` and starts the Lambda Runtime Interface Client (RIC). The handler (`processor.handler.handler`) bridges the camelCase event payload to env vars, then calls the same processing logic.

The container image includes `awslambdaric` so a single image works for both compute types. Processors declare supported compute types via `computeTypes` in the workflow definition.

## Local Development

1. Place input TIFF files in `data/input/`
2. Adjust `dev.env` if needed
3. Run:

```bash
make run
```

Output will appear in `data/output/`.

## Compatible Viewers

The output is compatible with any OME-Zarr / NGFF viewer:
- [Napari](https://napari.org/) (with `napari-ome-zarr` plugin)
- [Vizarr](https://github.com/hms-dbmi/vizarr)
- [Viv](https://github.com/hms-dbmi/viv)
- [OMERO](https://www.openmicroscopy.org/omero/)
- [Neuroglancer](https://github.com/google/neuroglancer)
