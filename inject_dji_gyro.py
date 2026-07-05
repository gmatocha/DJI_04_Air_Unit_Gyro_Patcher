#!/usr/bin/env python3
"""
inject_dji_gyro.py

Write gyro/attitude data from a CSV file (in the format produced by
extract_dji_gyro.py) back into a DJI MP4 file's embedded "djmd"
telemetry track.

HOW IT WORKS
------------
extract_dji_gyro.py finds each attitude sample by recursively walking the
"djmd" Protobuf metadata track and locating length-delimited submessages
that look like a unit quaternion (4 fixed32 fields, tags 1..4, unit
norm). Crucially, editing the *values* of those 4 floats does not change
their byte length (every float stays 4 bytes), so the surrounding
Protobuf/MP4 structure - and every sample-table offset/size in the MP4 -
stays valid. That means the new quaternion values can be written
directly (in place) over the old ones at the exact same file byte
offsets, with no remuxing required.

This script:
  1. Copies the input MP4 to the output path (original is untouched).
  2. Re-locates every quaternion sample in the "djmd" track the same way
     extract_dji_gyro.py did, but this time also records each sample's
     absolute byte offset in the file.
  3. Reads the CSV and groups rows by frame_index/sample_index.
  4. For every frame where the CSV has exactly as many samples as were
     found in the MP4, overwrites the 4 quaternion floats (w, x, y, z)
     in the output file with the CSV's values. Frames where the sample
     count doesn't match are left untouched and reported as skipped.

The CSV must contain at least: frame_index, sample_index, quat_w,
quat_x, quat_y, quat_z. If quat_* columns are missing but roll_deg/
pitch_deg/yaw_deg are present, those are converted to a quaternion.

Requires: Python 3.8+, ffmpeg + ffprobe on PATH. No other dependencies.

USAGE
-----
    python3 inject_dji_gyro.py INPUT.MP4 DATA.csv OUTPUT.MP4
"""

import sys
import os
import json
import struct
import subprocess
import csv
import math
import shutil


# --------------------------------------------------------------------------
# Minimal, schema-less Protobuf wire-format parser (offset-tracking version)
# --------------------------------------------------------------------------

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")
    return result, pos


def parse_fields(buf):
    """Parse `buf` as protobuf. Returns a list of
    (field_num, wire_type, value_bytes, value_start_offset_in_buf)."""
    pos = 0
    end = len(buf)
    out = []
    while pos < end:
        tag, pos = read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 0:
            raise ValueError("invalid field number 0")
        if wire_type == 0:
            val, pos = read_varint(buf, pos)
            out.append((field_num, 0, val, None))
        elif wire_type == 1:
            if pos + 8 > end:
                raise ValueError("truncated fixed64")
            out.append((field_num, 1, buf[pos:pos + 8], pos))
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            if length < 0 or pos + length > end:
                raise ValueError("truncated length-delimited field")
            out.append((field_num, 2, buf[pos:pos + length], pos))
            pos += length
        elif wire_type == 5:
            if pos + 4 > end:
                raise ValueError("truncated fixed32")
            out.append((field_num, 5, buf[pos:pos + 4], pos))
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
    return out


def find_quaternion_offsets(buf, base_offset=0):
    """Recursively search `buf` for quaternion-shaped submessages. Returns a
    list of (values, absolute_byte_offsets) where `values` is [w, x, y, z]
    and `absolute_byte_offsets` is the 4 corresponding byte positions
    (relative to the start of the outermost `buf` first passed in, i.e.
    relative to the start of the packet) of each float's 4-byte payload."""
    results = []
    try:
        fields = parse_fields(buf)
    except ValueError:
        return results

    if (len(fields) == 4
            and all(f[1] == 5 for f in fields)
            and [f[0] for f in fields] == [1, 2, 3, 4]):
        vals = [struct.unpack('<f', f[2])[0] for f in fields]
        norm_sq = sum(v * v for v in vals)
        if 0.9 < norm_sq < 1.1:
            offsets = [base_offset + f[3] for f in fields]
            results.append((vals, offsets))
            return results

    for field_num, wire_type, val, val_start in fields:
        if wire_type == 2:
            results.extend(find_quaternion_offsets(val, base_offset + val_start))
    return results


def euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg) / 2
    p = math.radians(pitch_deg) / 2
    y = math.radians(yaw_deg) / 2
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y_, z


# --------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# --------------------------------------------------------------------------

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def find_djmd_stream_index(mp4_path):
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", mp4_path
    ])
    info = json.loads(out)
    for stream in info.get("streams", []):
        tag = stream.get("codec_tag_string", "")
        handler = stream.get("tags", {}).get("handler_name", "")
        if tag == "djmd" or "DJI meta" in handler:
            return stream["index"]
    raise RuntimeError(
        "No DJI metadata ('djmd') track found in this file. "
        "This script only supports DJI recordings that embed the "
        "'dvtm_*' Protobuf telemetry track."
    )


def get_packets(mp4_path, stream_index):
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_packets", "-select_streams", str(stream_index), mp4_path
    ])
    return json.loads(out)["packets"]


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------

def load_csv_samples(csv_path):
    """Returns dict: frame_index (int) -> list of (sample_index, (w,x,y,z))
    sorted by sample_index."""
    by_frame = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_quat = all(c in fieldnames for c in
                        ("quat_w", "quat_x", "quat_y", "quat_z"))
        has_euler = all(c in fieldnames for c in
                         ("roll_deg", "pitch_deg", "yaw_deg"))
        if not has_quat and not has_euler:
            raise ValueError(
                "CSV must contain either quat_w/quat_x/quat_y/quat_z or "
                "roll_deg/pitch_deg/yaw_deg columns."
            )
        for row in reader:
            frame_idx = int(float(row["frame_index"]))
            sample_idx = int(float(row["sample_index"]))
            if has_quat:
                w = float(row["quat_w"])
                x = float(row["quat_x"])
                y = float(row["quat_y"])
                z = float(row["quat_z"])
            else:
                w, x, y, z = euler_deg_to_quat(
                    float(row["roll_deg"]),
                    float(row["pitch_deg"]),
                    float(row["yaw_deg"]),
                )
            by_frame.setdefault(frame_idx, []).append((sample_idx, (w, x, y, z)))
    for frame_idx in by_frame:
        by_frame[frame_idx].sort(key=lambda t: t[0])
    return by_frame


# --------------------------------------------------------------------------
# Main injection pipeline
# --------------------------------------------------------------------------

def inject_gyro_data(input_mp4, csv_path, output_mp4):
    stream_index = find_djmd_stream_index(input_mp4)
    packets = get_packets(input_mp4, stream_index)
    csv_by_frame = load_csv_samples(csv_path)

    print(f"Copying {input_mp4} -> {output_mp4} ...")
    shutil.copyfile(input_mp4, output_mp4)

    frames_patched = 0
    frames_skipped = 0
    samples_written = 0

    with open(output_mp4, "r+b") as f:
        for frame_idx, pkt in enumerate(packets):
            pkt_pos = int(pkt["pos"])
            pkt_size = int(pkt["size"])

            csv_samples = csv_by_frame.get(frame_idx)
            if not csv_samples:
                continue  # no CSV data for this frame; leave as-is

            f.seek(pkt_pos)
            chunk = f.read(pkt_size)

            found = find_quaternion_offsets(chunk)

            if len(found) != len(csv_samples):
                print(f"  [skip] frame {frame_idx}: found {len(found)} "
                      f"quaternion sample(s) in MP4 but CSV has "
                      f"{len(csv_samples)} - counts must match to patch "
                      f"safely, leaving this frame unchanged.")
                frames_skipped += 1
                continue

            for i in range(len(found)):
                _, offsets = found[i]
                sample_idx, (w, x, y, z) = csv_samples[i]
                for value, rel_offset in zip((w, x, y, z), offsets):
                    abs_offset = pkt_pos + rel_offset
                    f.seek(abs_offset)
                    f.write(struct.pack("<f", value))
                    samples_written += 1
            frames_patched += 1

    print(f"Done. Patched {frames_patched} frame(s), "
          f"skipped {frames_skipped} frame(s), "
          f"wrote {samples_written} float value(s).")
    print(f"Output written to: {output_mp4}")


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} INPUT.MP4 DATA.csv OUTPUT.MP4")
        sys.exit(1)

    input_mp4, csv_path, output_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
    inject_gyro_data(input_mp4, csv_path, output_mp4)


if __name__ == "__main__":
    main()
