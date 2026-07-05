#!/usr/bin/env python3
"""
extract_dji_gyro.py

Extract gyroscope / attitude telemetry from a DJI-recorded MP4 file and
save it to a CSV file with timestamps.

HOW IT WORKS
------------
Recent DJI drones/goggles (O4 air unit generation - e.g. Avata 2, Neo,
Flip, Mini 5 Pro, goggles, etc.) embed a proprietary Protobuf-encoded
metadata track in the MP4 container (fourcc "djmd", handler name
"DJI meta"). This track carries a "dvtm_*.proto" message per video frame
that contains, among other things, a burst of high-rate unit
quaternions describing the aircraft/gimbal attitude as measured by the
IMU (i.e. gyro-derived orientation) for that frame's time window.

DJI does not publish the .proto schema, so this script does NOT decode
the full message. Instead it:
  1. Uses ffprobe/ffmpeg to locate and extract the "djmd" data track,
     packet-by-packet (one packet per video frame).
  2. Walks each packet's bytes as generic Protobuf (tag/wire-type
     parsing only, no schema) and recursively searches for any
     length-delimited submessage that looks like a unit quaternion:
     exactly 4 fields, field numbers 1..4, wire type 5 (fixed32/float),
     with w^2+x^2+y^2+z^2 ~= 1. This pattern is stable across firmware
     versions because it's how protobuf encodes 4 required floats,
     regardless of what DJI calls the field.
  3. Assigns a timestamp to every quaternion sample by evenly spacing
     the samples found in a packet across that packet's video frame
     duration (from ffprobe), anchored at the frame's presentation
     time.
  4. Converts each quaternion to roll/pitch/yaw (degrees) for
     convenience, and also numerically differentiates the quaternion
     sequence to estimate angular velocity (deg/s) about each axis -
     this is the closest equivalent to "raw gyro" data derivable from
     what DJI exposes; the true raw gyro-rate stream is not present in
     the MP4 in an openly readable form.

Requires: Python 3.8+, ffmpeg + ffprobe on PATH. No other dependencies.

USAGE
-----
    python3 extract_dji_gyro.py INPUT.MP4 [OUTPUT.csv]

If OUTPUT.csv is omitted, it defaults to INPUT_gyro.csv next to the
input file.
"""

import sys
import os
import json
import struct
import subprocess
import csv
import math


# --------------------------------------------------------------------------
# Minimal, schema-less Protobuf wire-format parser
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
    """Parse `buf` as a flat sequence of protobuf (field_num, wire_type, value)
    tuples. Raises ValueError if `buf` doesn't parse cleanly as protobuf."""
    pos = 0
    end = len(buf)
    out = []
    while pos < end:
        tag, pos = read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 0:
            raise ValueError("invalid field number 0")
        if wire_type == 0:  # varint
            val, pos = read_varint(buf, pos)
            out.append((field_num, 0, val))
        elif wire_type == 1:  # fixed64
            if pos + 8 > end:
                raise ValueError("truncated fixed64")
            out.append((field_num, 1, buf[pos:pos + 8]))
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = read_varint(buf, pos)
            if length < 0 or pos + length > end:
                raise ValueError("truncated length-delimited field")
            out.append((field_num, 2, buf[pos:pos + length]))
            pos += length
        elif wire_type == 5:  # fixed32
            if pos + 4 > end:
                raise ValueError("truncated fixed32")
            out.append((field_num, 5, buf[pos:pos + 4]))
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
    return out


def find_quaternions(buf):
    """Recursively search `buf` (raw protobuf bytes) for submessages that
    look like a unit quaternion: 4 fixed32 fields, numbered 1..4, whose
    squared components sum to ~1.0. Returns a list of [w, x, y, z] in the
    order encountered (which is chronological within a packet)."""
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
            results.append(vals)
            return results  # a quaternion message has no further nesting

    for field_num, wire_type, val in fields:
        if wire_type == 2:
            results.extend(find_quaternions(val))
    return results


def quat_to_euler_deg(w, x, y, z):
    """Convert a unit quaternion (w, x, y, z) to roll/pitch/yaw in degrees
    (aerospace ZYX convention)."""
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_angular_velocity_deg_s(q1, q2, dt):
    """Estimate angular velocity (deg/s, body frame x,y,z) between two unit
    quaternions q1 -> q2 separated by dt seconds, via the quaternion
    derivative formula omega = 2 * (dq/dt) * conj(q)."""
    if dt <= 0:
        return 0.0, 0.0, 0.0
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dw = (w2 - w1) / dt
    dx = (x2 - x1) / dt
    dy = (y2 - y1) / dt
    dz = (z2 - z1) / dt
    # conjugate of the (midpoint) orientation
    w, x, y, z = w1, -x1, -y1, -z1
    # quaternion multiplication: (dq) * conj(q), take vector part *2
    ow = dw * w - dx * x - dy * y - dz * z
    ox = dw * x + dx * w + dy * z - dz * y
    oy = dw * y - dx * z + dy * w + dz * x
    oz = dw * z + dx * y - dy * x + dz * w
    return (2 * ox * 180 / math.pi,
            2 * oy * 180 / math.pi,
            2 * oz * 180 / math.pi)


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
    """Return the ffmpeg stream index of the DJI metadata ('djmd') track."""
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


def extract_track_bytes(mp4_path, stream_index, tmp_path):
    run([
        "ffmpeg", "-y", "-v", "quiet", "-i", mp4_path,
        "-map", f"0:{stream_index}", "-c", "copy", "-f", "data", tmp_path
    ])
    with open(tmp_path, "rb") as f:
        return f.read()


def get_packets(mp4_path, stream_index):
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_packets", "-select_streams", str(stream_index), mp4_path
    ])
    return json.loads(out)["packets"]


# --------------------------------------------------------------------------
# Main extraction pipeline
# --------------------------------------------------------------------------

def extract_gyro_data(mp4_path):
    """Returns a list of dict rows, one per quaternion/gyro sample."""
    stream_index = find_djmd_stream_index(mp4_path)
    packets = get_packets(mp4_path, stream_index)

    # Use the current working directory for the scratch file rather than
    # /tmp, since /tmp doesn't exist on Windows.
    tmp_path = os.path.join(os.getcwd(), f"_djmd_track_{os.getpid()}.tmp")
    try:
        track_bytes = extract_track_bytes(mp4_path, stream_index, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    rows = []
    offset = 0
    prev_quat = None
    prev_t = None

    for frame_idx, pkt in enumerate(packets):
        size = int(pkt["size"])
        chunk = track_bytes[offset:offset + size]
        offset += size

        pkt_time = float(pkt.get("pts_time", pkt.get("dts_time", 0.0)))
        pkt_duration = float(pkt.get("duration_time", 0.0)) or 0.01

        quats = find_quaternions(chunk)
        n = len(quats)
        if n == 0:
            continue

        for i, (w, x, y, z) in enumerate(quats):
            t = pkt_time + (i / n) * pkt_duration
            roll, pitch, yaw = quat_to_euler_deg(w, x, y, z)

            if prev_quat is not None and t > prev_t:
                gx, gy, gz = quat_angular_velocity_deg_s(
                    prev_quat, (w, x, y, z), t - prev_t
                )
            else:
                gx, gy, gz = 0.0, 0.0, 0.0

            rows.append({
                "frame_index": frame_idx,
                "sample_index": i,
                "timestamp_s": round(t, 6),
                "quat_w": w,
                "quat_x": x,
                "quat_y": y,
                "quat_z": z,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "gyro_x_deg_s": gx,
                "gyro_y_deg_s": gy,
                "gyro_z_deg_s": gz,
            })
            prev_quat = (w, x, y, z)
            prev_t = t

    return rows


def write_csv(rows, csv_path):
    fieldnames = [
        "frame_index", "sample_index", "timestamp_s",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} INPUT.MP4 [OUTPUT.csv]")
        sys.exit(1)

    mp4_path = sys.argv[1]
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        csv_path = base + "_gyro.csv"

    print(f"Reading DJI telemetry from: {mp4_path}")
    rows = extract_gyro_data(mp4_path)
    if not rows:
        print("No gyro/attitude samples were found in this file.")
        sys.exit(2)

    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} samples to: {csv_path}")
    dur = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]
    rate = len(rows) / dur if dur > 0 else 0
    print(f"Approx sample rate: {rate:.1f} Hz over {dur:.3f} s")


if __name__ == "__main__":
    main()
