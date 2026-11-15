#!/usr/bin/env python3
"""
Fix the multiscales scale in OME-Zarr assets written by the TIFF converter before v1.3.5.

Those versions wrote the scale in pixels (XY 1, 2, 4, ... per level; Z the Z/XY
spacing ratio) while the axes say micrometer, so the shape is right but the
scale bar is off by the pixel size. This rewrites only each asset's root
zarr.json with the real voxel size per level, in micrometers. Chunks are not
touched.

The XY pixel size isn't stored in the zarr, so it is read from the source
TIFF's OME-XML (first IFD only, with HTTP range reads; the TIFF isn't
downloaded).

Reads go through the Pennsieve API (asset list, signed CloudFront URLs, source
file URLs), so a dry run needs only a Pennsieve session token. --apply writes
with boto3 and needs S3 write access to the asset buckets.

Usage:
  export PENNSIEVE_SESSION_TOKEN=...          # api2.pennsieve.io session token
  python patch_zarr_scale.py packages.txt                     # dry run
  python patch_zarr_scale.py packages.txt --apply --main-bucket <bucket>

packages.txt: one "N:package:... N:dataset:..." per line, or a run list with a
"# DATASET_ID=N:dataset:..." header and one package ID per line.

Only assets whose zarr.json carries this converter's signature ("Pyramid with
initial_downsample=") and whose current scale matches the old formula are
changed; everything else is reported and skipped (NIfTI assets, already-fixed
assets, files whose metadata doesn't match). Each original zarr.json is saved
under --backup-dir before it is overwritten, and the CloudFront paths to
invalidate are written to --invalidations.
"""

import argparse
import datetime
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

API = os.environ.get("PENNSIEVE_API_HOST", "https://api.pennsieve.io")
API2 = os.environ.get("PENNSIEVE_API2_HOST", "https://api2.pennsieve.io")
# CloudFront path prefix -> bucket. The prefix is md5(bucket)[:8]; assets without a
# prefix are in the platform's main storage bucket (--main-bucket).
PREFIX_BUCKETS = {"b11f440c": "prd-sparc-storage-use1"}
SIGNATURE = "Pyramid with initial_downsample="
FIX_MARKER = "scale_units"

UNIT_TO_UM = {
    "pm": 1e-6, "nm": 1e-3, "µm": 1.0, "μm": 1.0, "um": 1.0, "micron": 1.0,
    "micrometer": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6,
}


# ---------------------------------------------------------------- HTTP helpers

def _token():
    t = os.environ.get("PENNSIEVE_SESSION_TOKEN")
    if not t:
        sys.exit("Set PENNSIEVE_SESSION_TOKEN")
    return t


def api_get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def http_get(url, rng=None):
    headers = {"Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"} if rng else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120).read()


# ---------------------------------------------------------------- TIFF header

def tiff_description(url):
    """ImageDescription (tag 270) of the first IFD, via range reads."""
    head = http_get(url, (0, 16))
    if head[:2] not in (b"II", b"MM"):
        raise ValueError("not a TIFF")
    e = "<" if head[:2] == b"II" else ">"
    big = struct.unpack(e + "H", head[2:4])[0] == 43
    off = struct.unpack(e + "Q", head[8:16])[0] if big else struct.unpack(e + "I", head[4:8])[0]
    count_size, entry_size = (8, 20) if big else (2, 12)
    n = struct.unpack(e + ("Q" if big else "H"), http_get(url, (off, count_size)))[0]
    ifd = http_get(url, (off + count_size, n * entry_size))
    for k in range(n):
        ent = ifd[k * entry_size:(k + 1) * entry_size]
        tag = struct.unpack(e + "H", ent[:2])[0]
        if tag != 270:
            continue
        cnt = struct.unpack(e + ("Q" if big else "I"), ent[4:12] if big else ent[4:8])[0]
        val = ent[12:20] if big else ent[8:12]
        if cnt <= len(val):
            raw = val[:cnt]
        else:
            ptr = struct.unpack(e + ("Q" if big else "I"), val)[0]
            raw = http_get(url, (ptr, cnt))
        return raw.rstrip(b"\0").decode("utf-8", "replace")
    return ""


def physical_sizes_new(xml):
    """(pz, py, px) in µm, as the fixed converter reads them (units, exponents)."""
    attrs = {}
    if xml and "<OME" in xml:
        try:
            px = ET.fromstring(xml).find(".//{*}Pixels")
            if px is not None:
                attrs = dict(px.attrib)
        except ET.ParseError:
            pass
    if not attrs:
        for m in re.finditer(r'(PhysicalSize[XYZ](?:Unit)?)="([^"]*)"', xml or ""):
            attrs.setdefault(m.group(1), m.group(2))

    def grab(axis):
        try:
            v = float(attrs.get(f"PhysicalSize{axis}", ""))
        except ValueError:
            return 1.0
        if not v > 0:
            return 1.0
        f = UNIT_TO_UM.get(attrs.get(f"PhysicalSize{axis}Unit", "µm").strip())
        if f is None:
            raise ValueError(f"unknown PhysicalSize{axis}Unit")
        return v * f
    return grab("Z"), grab("Y"), grab("X")


def physical_sizes_old(xml):
    """(pz, py, px) exactly as converter <= v1.3.4 parsed them (regex, no units)."""
    def grab(key):
        m = re.search(rf'{key}="([\d.]+)"', xml or "")
        return float(m.group(1)) if m else 1.0
    return grab("PhysicalSizeZ"), grab("PhysicalSizeY"), grab("PhysicalSizeX")


# ---------------------------------------------------------------- scale math

def close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def new_multiscales(attrs, xml):
    """Return (new_attrs, note). new_attrs is None when nothing should change."""
    ms = (attrs.get("multiscales") or [None])[0]
    if not ms:
        return None, "no multiscales"
    if ms.get("metadata", {}).get(FIX_MARKER):
        return None, "already patched"
    if SIGNATURE not in ms.get("metadata", {}).get("description", ""):
        return None, "not written by the TIFF converter"
    z_down = "z_downsampled=True" in ms["metadata"]["description"]
    names = [a["name"] for a in ms["axes"]]
    is_3d = names == ["c", "z", "y", "x"]
    if not is_3d and names != ["c", "y", "x"]:
        return None, f"unexpected axes {names}"

    oz, oy, ox = physical_sizes_old(xml)
    nz, ny, nx = physical_sizes_new(xml)
    old_ratio = oz / (min(oy, ox) if min(oy, ox) > 0 else 1.0)

    new_ms = json.loads(json.dumps(ms))
    changed = False
    for ds in new_ms["datasets"]:
        t = next(t for t in ds["coordinateTransformations"] if t["type"] == "scale")
        s = t["scale"]
        shrink = s[-1]
        if s[-2] != shrink or shrink < 1 or shrink != int(shrink):
            return None, f"level {ds['path']} XY scale {s[-2:]} isn't a pixel shrink factor"
        z_mult = shrink if z_down else 1.0
        if is_3d and not close(s[1], old_ratio * z_mult):
            return None, (f"level {ds['path']} Z scale {s[1]} doesn't match the old formula "
                          f"({old_ratio * z_mult:.6g}); check by hand")
        new = ([1.0, nz * z_mult, ny * shrink, nx * shrink] if is_3d
               else [1.0, ny * shrink, nx * shrink])
        if any(not close(a, b) for a, b in zip(s, new)):
            changed = True
        t["scale"] = new
    if not changed:
        return None, "scale already correct (no PhysicalSize in the source, or 1 µm pixels)"
    new_ms.setdefault("metadata", {})[FIX_MARKER] = (
        f"micrometers (patched {datetime.date.today().isoformat()}; was pixels)")
    out = dict(attrs)
    out["multiscales"] = [new_ms] + attrs["multiscales"][1:]
    return out, f"pixel size (z,y,x) µm = ({nz:g}, {ny:g}, {nx:g})"


# ---------------------------------------------------------------- inputs

def read_packages(path):
    out, ds = [], None
    for line in open(path):
        line = line.strip()
        m = re.match(r"#\s*DATASET_ID=(N:dataset:[0-9a-f-]+)", line)
        if m:
            ds = m.group(1)
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        pkg = parts[0]
        d = next((p for p in parts[1:] if p.startswith("N:dataset:")), ds)
        if not d:
            sys.exit(f"No dataset ID for {pkg}")
        out.append((pkg, d))
    return out


def source_url(pkg):
    srcs = api_get(f"{API}/packages/{pkg}/sources")
    tiffs = [s["content"] for s in srcs
             if s["content"].get("filename", "").lower().endswith((".tif", ".tiff"))]
    if len(tiffs) != 1:
        raise ValueError(f"{len(tiffs)} TIFF source files")
    return api_get(f"{API}/packages/{pkg}/files/{tiffs[0]['id']}")["url"]


def s3_location(asset_url, base_url, main_bucket):
    """Bucket and key prefix of an asset, from its CloudFront URL."""
    path = asset_url.split("://", 1)[1].split("/", 1)[1].strip("/")
    first = path.split("/", 1)[0]
    if first in PREFIX_BUCKETS:
        return PREFIX_BUCKETS[first], path.split("/", 1)[1]
    if re.fullmatch(r"[0-9a-f]{8}", first):
        raise ValueError(f"unknown CloudFront prefix {first}; add it to PREFIX_BUCKETS")
    return main_bucket, path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packages")
    ap.add_argument("--apply", action="store_true", help="write the patched zarr.json files (default: dry run)")
    ap.add_argument("--main-bucket", help="bucket for assets without a CloudFront prefix (needed with --apply)")
    ap.add_argument("--backup-dir", default="zarr_scale_backups")
    ap.add_argument("--invalidations", default="cloudfront_invalidation_paths.txt")
    a = ap.parse_args()

    s3 = None
    if a.apply:
        import boto3
        s3 = boto3.client("s3")
        os.makedirs(a.backup_dir, exist_ok=True)

    seen, counts, inval = set(), {}, []
    for pkg, ds in read_packages(a.packages):
        try:
            r = api_get(f"{API2}/packages/assets?package_id={pkg}&dataset_id={ds}")
        except urllib.error.HTTPError as e:
            print(f"{pkg}\tERROR assets API {e.code}"); counts["error"] = counts.get("error", 0) + 1
            continue
        c = r["cloudfront"]
        q = f"?Policy={c['policy']}&Signature={c['signature']}&Key-Pair-Id={c['key_pair_id']}"
        zarrs = [x for x in r["assets"] if x["asset_type"] == "ome-zarr" and x["status"] == "ready"]
        if not zarrs:
            print(f"{pkg}\tSKIP no ready ome-zarr asset"); counts["skip"] = counts.get("skip", 0) + 1
            continue
        xml = None
        for asset in zarrs:
            if asset["id"] in seen:
                continue
            seen.add(asset["id"])
            tag = f"{pkg}\t{asset['id']}"
            try:
                url = asset["asset_url"].rstrip("/") + "/"
                attrs_doc = json.loads(http_get(url + "zarr.json" + q))
                attrs = attrs_doc.get("attributes", {})
                ms = (attrs.get("multiscales") or [{}])[0]
                if SIGNATURE in ms.get("metadata", {}).get("description", "") and xml is None:
                    xml = tiff_description(source_url(pkg))
                new_attrs, note = new_multiscales(attrs, xml or "")
            except Exception as e:  # report and keep going
                print(f"{tag}\tERROR {type(e).__name__}: {e}"); counts["error"] = counts.get("error", 0) + 1
                continue
            if new_attrs is None:
                print(f"{tag}\tSKIP {note}"); counts["skip"] = counts.get("skip", 0) + 1
                continue
            old_s = [d["coordinateTransformations"][0]["scale"] for d in ms["datasets"]]
            new_s = [d["coordinateTransformations"][0]["scale"] for d in new_attrs["multiscales"][0]["datasets"]]
            print(f"{tag}\t{'PATCH' if a.apply else 'WOULD PATCH'} {note}\n"
                  f"\t\tlevel 0: {old_s[0]} -> {[round(v, 6) for v in new_s[0]]} ({len(new_s)} levels)")
            new_doc = dict(attrs_doc, attributes=new_attrs)
            path = url.split("://", 1)[1].split("/", 1)[1]
            if a.apply:
                try:
                    bucket, prefix = s3_location(url, c["base_url"], a.main_bucket)
                    if not bucket:
                        raise ValueError("asset has no CloudFront prefix; pass --main-bucket")
                    key = prefix.rstrip("/") + "/zarr.json"
                    cur = s3.get_object(Bucket=bucket, Key=key)
                    body = cur["Body"].read()
                    if json.loads(body) != attrs_doc:
                        raise ValueError(f"s3://{bucket}/{key} differs from what CloudFront served; not touching it")
                    with open(os.path.join(a.backup_dir, f"{asset['id']}.zarr.json"), "wb") as f:
                        f.write(body)
                    extra = {k: cur[k] for k in ("ContentType", "CacheControl", "ContentEncoding") if cur.get(k)}
                    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(new_doc, indent=2).encode(), **extra)
                except Exception as e:
                    print(f"{tag}\tERROR writing: {type(e).__name__}: {e}"); counts["error"] = counts.get("error", 0) + 1
                    continue
            inval.append("/" + path + "zarr.json")
            counts["patch"] = counts.get("patch", 0) + 1

    with open(a.invalidations, "w") as f:
        f.write("\n".join(inval) + ("\n" if inval else ""))
    print(f"\nSummary: {counts.get('patch', 0)} {'patched' if a.apply else 'to patch'}, "
          f"{counts.get('skip', 0)} skipped, {counts.get('error', 0)} errors")
    if inval:
        print(f"CloudFront paths to invalidate ({len(inval)}): {a.invalidations}"
              + ("" if a.apply else " (dry run: nothing was written)"))


if __name__ == "__main__":
    main()
