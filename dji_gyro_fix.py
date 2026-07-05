#!/usr/bin/env python3
"""
dji_gyro_fix.py

All-in-one pipeline for DJI recordings affected by the camera's
occasional "bad gyro data" bug:

    1. EXTRACT  - read the embedded gyro/attitude telemetry out of the
                  MP4 into a CSV.
    2. PATCH    - detect and smooth over the bad bursts in that CSV.
    3. INJECT   - write the patched telemetry back into a copy of the
                  MP4.

This combines extract_dji_gyro.py + patch_dji_gyro_csv.py +
inject_dji_gyro.py into one tool so you don't have to run three
separate commands. Every intermediate CSV is still written to disk (so
you can inspect or re-run just one stage), but the pipeline carries the
data in memory between stages rather than re-reading files.

USAGE
-----
    python3 dji_gyro_fix.py INPUT.MP4 [options]

By default, given INPUT.MP4, this writes:
    INPUT_gyro_raw.csv       - the raw extracted telemetry
    INPUT_gyro_patched.csv   - the same data after patching
    INPUT_patched.MP4        - a copy of INPUT.MP4 with the patched
                               telemetry written back in

Use --raw-csv / --patched-csv / --output-mp4 to override any of those
paths, or --report-only to stop after showing what WOULD be patched
without writing anything.

Requires: Python 3.8+, ffmpeg + ffprobe on PATH. No other dependencies.


BACKGROUND - HOW EACH STAGE WORKS
----------------------------------

EXTRACT
Recent DJI drones/goggles (O4 air unit generation) embed a proprietary
Protobuf-encoded metadata track in the MP4 container (fourcc "djmd").
This track carries, among other things, a burst of high-rate unit
quaternions describing the aircraft/gimbal attitude for each video
frame's time window. DJI doesn't publish the schema, so this script
walks the raw Protobuf bytes generically, looking for any
length-delimited submessage that looks like a unit quaternion (4
fixed32 fields, tags 1-4, unit norm) - a pattern that's stable across
firmware versions because it's simply how Protobuf encodes 4 required
floats. Each sample's timestamp is assigned by spacing the quaternions
found in a packet evenly across that video frame's duration.

PATCH
The bad-data bug shows up as a short burst of rapid, oscillatory
("ringing") attitude change - the gyro rate swings wildly, often
reversing sign within a few milliseconds - which is different from a
genuine fast camera rotation (which tracks its own recent trend
closely, even if fast). So detection runs on the RESIDUAL: each
sample's gyro vector minus a short local moving-average trend. A real
rotation's residual stays small; the glitch's residual spikes hugely.
Flagged bursts are grown outward through their decaying "ringdown"
tail, then bridged with a quaternion SLERP between the last-good
sample before and first-good sample after - the smooth path the camera
would have taken if the glitch hadn't happened. A duration safety
valve avoids treating genuinely long, real motion as a glitch.
Multiple passes let smaller anomalies (masked by the file's big
glitches on the first pass) get caught once the big ones are patched.

INJECT
Editing the 4 quaternion floats doesn't change their byte length, so
the surrounding Protobuf/MP4 structure stays valid - the new values
can be written directly over the old ones at the same file byte
offsets, with no remuxing needed.
"""

import sys
import os
import json
import struct
import subprocess
import csv
import math
import shutil
import argparse


CSV_FIELDNAMES = [
    "frame_index", "sample_index", "timestamp_s",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s",
]


# ==========================================================================
# Shared: schema-less Protobuf wire-format parser (offset-tracking)
# ==========================================================================

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
    (field_num, wire_type, value_bytes_or_int, value_start_offset_in_buf)."""
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
            return results  # a quaternion message has no further nesting

    for field_num, wire_type, val, val_start in fields:
        if wire_type == 2:
            results.extend(find_quaternion_offsets(val, base_offset + val_start))
    return results


# ==========================================================================
# Shared: quaternion math
# ==========================================================================

def quat_to_euler_deg(w, x, y, z):
    """Convert a unit quaternion (w, x, y, z) to roll/pitch/yaw in degrees
    (aerospace ZYX convention)."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


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


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions q0 and q1,
    t in [0, 1]."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1

    if dot < 0.0:
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        w = w0 + t * (w1 - w0)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        return quat_normalize((w, x, y, z))

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    w = s0 * w0 + s1 * w1
    x = s0 * x0 + s1 * x1
    y = s0 * y0 + s1 * y1
    z = s0 * z0 + s1 * z1
    return quat_normalize((w, x, y, z))


def quat_angular_velocity_deg_s(q1, q2, dt):
    """Angular velocity (deg/s, body frame x,y,z) between two unit
    quaternions q1 -> q2 separated by dt seconds."""
    if dt <= 0:
        return 0.0, 0.0, 0.0
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dw = (w2 - w1) / dt
    dx = (x2 - x1) / dt
    dy = (y2 - y1) / dt
    dz = (z2 - z1) / dt
    w, x, y, z = w1, -x1, -y1, -z1
    ox = dw * x + dx * w + dy * z - dz * y
    oy = dw * y - dx * z + dy * w + dz * x
    oz = dw * z + dx * y - dy * x + dz * w
    return (2 * ox * 180 / math.pi,
            2 * oy * 180 / math.pi,
            2 * oz * 180 / math.pi)


# ==========================================================================
# Shared: ffprobe / ffmpeg helpers
# ==========================================================================

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


def get_packets(mp4_path, stream_index):
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_packets", "-select_streams", str(stream_index), mp4_path
    ])
    return json.loads(out)["packets"]


def extract_track_bytes(mp4_path, stream_index, tmp_path):
    run([
        "ffmpeg", "-y", "-v", "quiet", "-i", mp4_path,
        "-map", f"0:{stream_index}", "-c", "copy", "-f", "data", tmp_path
    ])
    with open(tmp_path, "rb") as f:
        return f.read()


# ==========================================================================
# STAGE 1: EXTRACT
# ==========================================================================

def extract_gyro_data(mp4_path):
    """Returns (rows, packets, stream_index). `rows` is a list of dicts,
    one per quaternion/gyro sample. `packets` and `stream_index` are
    returned too so the inject stage can reuse them without re-probing
    the (untouched) input file."""
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

        found = find_quaternion_offsets(chunk)
        quats = [vals for vals, _offsets in found]
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

    return rows, packets, stream_index


# ==========================================================================
# STAGE 2: PATCH
# ==========================================================================

def compute_gyro_components(rows):
    n = len(rows)
    gx = [0.0] * n
    gy = [0.0] * n
    gz = [0.0] * n
    for i in range(1, n):
        dt = rows[i]["timestamp_s"] - rows[i - 1]["timestamp_s"]
        q1 = (rows[i - 1]["quat_w"], rows[i - 1]["quat_x"],
              rows[i - 1]["quat_y"], rows[i - 1]["quat_z"])
        q2 = (rows[i]["quat_w"], rows[i]["quat_x"],
              rows[i]["quat_y"], rows[i]["quat_z"])
        a, b, c = quat_angular_velocity_deg_s(q1, q2, dt)
        gx[i], gy[i], gz[i] = a, b, c
    return gx, gy, gz


def moving_average(values, window):
    """O(n) centered moving average via a running sum."""
    n = len(values)
    if window < 1:
        return list(values)
    half = window // 2
    out = [0.0] * n
    running = 0.0
    lo, hi = 0, -1
    for i in range(n):
        want_lo = max(0, i - half)
        want_hi = min(n - 1, i + half)
        while hi < want_hi:
            hi += 1
            running += values[hi]
        while lo < want_lo:
            running -= values[lo]
            lo += 1
        out[i] = running / (hi - lo + 1)
    return out


def compute_residual_magnitudes(rows, smoothing_window_s):
    """Compute a HIGH-PASS gyro-rate magnitude per sample: the raw gyro
    vector minus a locally-smoothed (moving-average) trend, magnitude of
    the remainder. A real, smooth, even fast rotation closely tracks its
    own local average (residual stays small); the glitch's oscillatory
    ringing swings wildly around its own trend (residual spikes hugely).
    """
    n = len(rows)
    dt = 0.0
    for i in range(1, min(n, 50)):
        d = rows[i]["timestamp_s"] - rows[i - 1]["timestamp_s"]
        if d > 0:
            dt = d
            break
    if dt <= 0:
        dt = 0.0005
    window = max(3, int(round(smoothing_window_s / dt)))
    if window % 2 == 0:
        window += 1

    gx, gy, gz = compute_gyro_components(rows)
    sx = moving_average(gx, window)
    sy = moving_average(gy, window)
    sz = moving_average(gz, window)

    return [math.sqrt((gx[i] - sx[i]) ** 2 + (gy[i] - sy[i]) ** 2 +
                        (gz[i] - sz[i]) ** 2) for i in range(n)]


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def rolling_max(mags, times, i, half_window_s):
    n = len(mags)
    t0 = times[i] - half_window_s
    t1 = times[i] + half_window_s
    lo = i
    while lo > 0 and times[lo - 1] >= t0:
        lo -= 1
    hi = i
    while hi < n - 1 and times[hi + 1] <= t1:
        hi += 1
    return max(mags[lo:hi + 1])


def expand_region(rows, mags, lo, hi, expand_threshold, quiet_duration,
                   smoothing_window):
    """Grow [lo, hi] outward through a decaying ringdown tail."""
    times = [r["timestamp_s"] for r in rows]
    n = len(rows)
    half_win = smoothing_window / 2.0

    i = lo
    quiet_since = None
    while i > 0:
        i -= 1
        local_mag = rolling_max(mags, times, i, half_win)
        if local_mag > expand_threshold:
            quiet_since = None
            lo = i
        else:
            if quiet_since is None:
                quiet_since = times[i]
            if (quiet_since - times[i]) >= quiet_duration:
                break

    j = hi
    quiet_since = None
    while j < n - 1:
        j += 1
        local_mag = rolling_max(mags, times, j, half_win)
        if local_mag > expand_threshold:
            quiet_since = None
            hi = j
        else:
            if quiet_since is None:
                quiet_since = times[j]
            if quiet_since is not None and (times[j] - quiet_since) >= quiet_duration:
                break

    return lo, hi


def detect_bad_regions(rows, mags, threshold_multiplier, absolute_floor,
                        merge_gap, expand_multiplier, quiet_duration,
                        smoothing_window, max_duration, force_long_regions):
    baseline_p99 = percentile(mags, 0.99)
    threshold = max(baseline_p99 * threshold_multiplier, absolute_floor)
    expand_threshold = max(baseline_p99 * expand_multiplier,
                            absolute_floor * expand_multiplier / threshold_multiplier)

    flagged_idx = [i for i, m in enumerate(mags) if m > threshold]

    if not flagged_idx:
        return [], [], baseline_p99, threshold, expand_threshold

    core_regions = []
    start = flagged_idx[0]
    prev = flagged_idx[0]
    for i in flagged_idx[1:]:
        gap = rows[i]["timestamp_s"] - rows[prev]["timestamp_s"]
        if gap > merge_gap:
            core_regions.append((start, prev))
            start = i
        prev = i
    core_regions.append((start, prev))

    accepted_core = []
    suspicious = []
    for (lo, hi) in core_regions:
        dur = rows[hi]["timestamp_s"] - rows[lo]["timestamp_s"]
        if dur > max_duration and not force_long_regions:
            suspicious.append((lo, hi, dur))
        else:
            accepted_core.append((lo, hi))

    expanded = []
    for (lo, hi) in accepted_core:
        new_lo, new_hi = expand_region(
            rows, mags, lo, hi, expand_threshold, quiet_duration,
            smoothing_window,
        )
        expanded.append((new_lo, new_hi))

    if not expanded:
        return [], suspicious, baseline_p99, threshold, expand_threshold

    merged = [expanded[0]]
    for (lo, hi) in expanded[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi + 1:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))

    return merged, suspicious, baseline_p99, threshold, expand_threshold


def patch_regions(rows, regions):
    n = len(rows)
    for (lo, hi) in regions:
        before_idx = lo - 1
        after_idx = hi + 1
        if before_idx < 0 or after_idx >= n:
            anchor_idx = after_idx if before_idx < 0 else before_idx
            anchor = rows[anchor_idx]
            for i in range(lo, hi + 1):
                rows[i]["quat_w"] = anchor["quat_w"]
                rows[i]["quat_x"] = anchor["quat_x"]
                rows[i]["quat_y"] = anchor["quat_y"]
                rows[i]["quat_z"] = anchor["quat_z"]
            continue

        q0 = (rows[before_idx]["quat_w"], rows[before_idx]["quat_x"],
              rows[before_idx]["quat_y"], rows[before_idx]["quat_z"])
        q1 = (rows[after_idx]["quat_w"], rows[after_idx]["quat_x"],
              rows[after_idx]["quat_y"], rows[after_idx]["quat_z"])
        t0 = rows[before_idx]["timestamp_s"]
        t1 = rows[after_idx]["timestamp_s"]
        span = t1 - t0

        for i in range(lo, hi + 1):
            t = rows[i]["timestamp_s"]
            frac = 0.0 if span <= 0 else (t - t0) / span
            frac = max(0.0, min(1.0, frac))
            w, x, y, z = quat_slerp(q0, q1, frac)
            rows[i]["quat_w"] = w
            rows[i]["quat_x"] = x
            rows[i]["quat_y"] = y
            rows[i]["quat_z"] = z


def recompute_derived_columns(rows):
    prev_quat = None
    prev_t = None
    for row in rows:
        w, x, y, z = row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]
        roll, pitch, yaw = quat_to_euler_deg(w, x, y, z)
        row["roll_deg"] = roll
        row["pitch_deg"] = pitch
        row["yaw_deg"] = yaw

        t = row["timestamp_s"]
        if prev_quat is not None and t > prev_t:
            gx, gy, gz = quat_angular_velocity_deg_s(prev_quat, (w, x, y, z),
                                                       t - prev_t)
        else:
            gx, gy, gz = 0.0, 0.0, 0.0
        row["gyro_x_deg_s"] = gx
        row["gyro_y_deg_s"] = gy
        row["gyro_z_deg_s"] = gz

        prev_quat = (w, x, y, z)
        prev_t = t


def patch_gyro_data(rows, args):
    """Runs the full multi-pass detect+patch pipeline in place on `rows`.
    Returns (total_samples_patched, total_regions_patched)."""
    total_samples = 0
    total_regions = 0

    for pass_num in range(1, args.passes + 1):
        mags = compute_residual_magnitudes(rows, args.residual_window)
        regions, suspicious, baseline_p99, threshold, expand_threshold = \
            detect_bad_regions(
                rows, mags,
                args.threshold_multiplier, args.absolute_floor,
                args.merge_gap, args.expand_multiplier, args.quiet_duration,
                args.smoothing_window, args.max_duration,
                args.force_long_regions,
            )

        print(f"\n--- Patch pass {pass_num} ---")
        print(f"99th percentile residual magnitude: {baseline_p99:.1f} deg/s")
        print(f"Core detection threshold: {threshold:.1f} deg/s")
        print(f"Expansion (ringdown) threshold: {expand_threshold:.1f} deg/s")

        if suspicious:
            print(f"{len(suspicious)} region(s) exceeded --max-duration "
                  f"({args.max_duration}s) at their core and were NOT "
                  f"auto-patched (could be real sustained motion):")
            for (lo, hi, dur) in suspicious:
                print(f"  t=[{rows[lo]['timestamp_s']:.4f}s, "
                      f"{rows[hi]['timestamp_s']:.4f}s]  duration={dur:.4f}s  "
                      f"peak_residual={max(mags[lo:hi + 1]):.1f} deg/s")

        if not regions:
            print("No (more) bad regions detected.")
            break

        print(f"Detected {len(regions)} bad region(s):")
        for (lo, hi) in regions:
            t0 = rows[lo]["timestamp_s"]
            t1 = rows[hi]["timestamp_s"]
            n = hi - lo + 1
            peak = max(mags[lo:hi + 1])
            print(f"  t=[{t0:.4f}s, {t1:.4f}s]  duration={t1 - t0:.4f}s  "
                  f"samples={n}  peak_residual={peak:.1f} deg/s")

        if args.report_only:
            print("(--report-only: not patching or iterating further passes)")
            break

        patch_regions(rows, regions)
        recompute_derived_columns(rows)

        total_samples += sum(hi - lo + 1 for (lo, hi) in regions)
        total_regions += len(regions)

    return total_samples, total_regions


# ==========================================================================
# STAGE 3: INJECT
# ==========================================================================

def rows_to_by_frame(rows):
    """Group patched rows by frame_index -> [(sample_index, (w,x,y,z)), ...]
    sorted by sample_index, matching the shape inject needs."""
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame_index"], []).append(
            (row["sample_index"],
             (row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]))
        )
    for frame_idx in by_frame:
        by_frame[frame_idx].sort(key=lambda t: t[0])
    return by_frame


def inject_gyro_data(input_mp4, packets, rows, output_mp4):
    """Copies input_mp4 to output_mp4, then overwrites the quaternion
    floats in the djmd track using the (already-patched) in-memory
    `rows`, re-locating each sample's exact byte offset from the
    ORIGINAL file bytes (packets/offsets are unaffected by patching,
    since only float values change, never lengths)."""
    by_frame = rows_to_by_frame(rows)

    print(f"Copying {input_mp4} -> {output_mp4} ...")
    shutil.copyfile(input_mp4, output_mp4)

    frames_patched = 0
    frames_skipped = 0
    samples_written = 0

    with open(output_mp4, "r+b") as f:
        for frame_idx, pkt in enumerate(packets):
            pkt_pos = int(pkt["pos"])
            pkt_size = int(pkt["size"])

            frame_samples = by_frame.get(frame_idx)
            if not frame_samples:
                continue

            f.seek(pkt_pos)
            chunk = f.read(pkt_size)

            found = find_quaternion_offsets(chunk)

            if len(found) != len(frame_samples):
                print(f"  [skip] frame {frame_idx}: found {len(found)} "
                      f"quaternion sample(s) in MP4 but data has "
                      f"{len(frame_samples)} - counts must match to patch "
                      f"safely, leaving this frame unchanged.")
                frames_skipped += 1
                continue

            for i in range(len(found)):
                _, offsets = found[i]
                sample_idx, (w, x, y, z) = frame_samples[i]
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


# ==========================================================================
# CSV I/O (for the intermediate files written between stages)
# ==========================================================================

def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract, patch, and re-inject gyro/attitude "
                    "telemetry for a DJI MP4 affected by the camera's "
                    "occasional bad-gyro-data bug, in one step."
    )
    parser.add_argument("input_mp4")
    parser.add_argument("--raw-csv", default=None,
                        help="Path for the raw extracted CSV "
                             "(default: INPUT_gyro_raw.csv)")
    parser.add_argument("--patched-csv", default=None,
                        help="Path for the patched CSV "
                             "(default: INPUT_gyro_patched.csv)")
    parser.add_argument("--output-mp4", default=None,
                        help="Path for the final patched MP4 "
                             "(default: INPUT_patched.MP4)")

    # Patch-stage tuning (see patch_dji_gyro_csv.py for full details)
    parser.add_argument("--residual-window", type=float, default=0.04,
                        help="Smoothing window (s) for the local trend "
                             "residual is computed against (default 0.04)")
    parser.add_argument("--threshold-multiplier", type=float, default=1.5,
                        help="Flag samples above N x the file's 99th "
                             "percentile residual magnitude as a region's "
                             "core (default 1.5)")
    parser.add_argument("--absolute-floor", type=float, default=195.0,
                        help="Never flag below this residual deg/s value "
                             "(default 195.0)")
    parser.add_argument("--merge-gap", type=float, default=0.5,
                        help="Merge flagged samples within N seconds of "
                             "each other into one core region (default 0.5)")
    parser.add_argument("--expand-multiplier", type=float, default=1.15,
                        help="Grow regions outward through their ringdown "
                             "tail while local residual stays above N x "
                             "baseline (default 1.15)")
    parser.add_argument("--quiet-duration", type=float, default=0.05,
                        help="Stop growing once quiet for N seconds "
                             "(default 0.05)")
    parser.add_argument("--smoothing-window", type=float, default=0.01,
                        help="Rolling-max window (s) used while growing "
                             "regions (default 0.01)")
    parser.add_argument("--max-duration", type=float, default=2.5,
                        help="Candidate regions longer than this are "
                             "treated as possible real motion (default 2.5)")
    parser.add_argument("--force-long-regions", action="store_true",
                        help="Patch regions longer than --max-duration "
                             "anyway")
    parser.add_argument("--passes", type=int, default=2,
                        help="Detection passes, recomputing the baseline "
                             "each time (default 2)")

    parser.add_argument("--report-only", action="store_true",
                        help="Extract and show what would be patched, but "
                             "don't write the patched CSV or output MP4")
    parser.add_argument("--skip-inject", action="store_true",
                        help="Extract and patch only; write both CSVs but "
                             "don't produce the output MP4")
    args = parser.parse_args()

    input_mp4 = args.input_mp4
    base = os.path.splitext(os.path.basename(input_mp4))[0]
    input_ext = os.path.splitext(input_mp4)[1] or ".mp4"
    raw_csv = args.raw_csv or f"{base}_gyro_raw.csv"
    patched_csv = args.patched_csv or f"{base}_gyro_patched.csv"
    output_mp4 = args.output_mp4 or f"{base}_patched{input_ext}"

    # ---- Stage 1: Extract ----
    print(f"=== Stage 1/3: Extracting gyro telemetry from {input_mp4} ===")
    rows, packets, stream_index = extract_gyro_data(input_mp4)
    if not rows:
        print("No gyro/attitude samples were found in this file.")
        sys.exit(2)

    write_csv(rows, raw_csv)
    dur = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]
    rate = len(rows) / dur if dur > 0 else 0
    print(f"Wrote {len(rows)} samples to: {raw_csv}")
    print(f"Approx sample rate: {rate:.1f} Hz over {dur:.3f} s")

    # ---- Stage 2: Patch ----
    print(f"\n=== Stage 2/3: Detecting and patching bad regions ===")
    total_samples, total_regions = patch_gyro_data(rows, args)

    if args.report_only:
        print("\n(--report-only: stopping before writing the patched CSV "
              "or output MP4)")
        return

    write_csv(rows, patched_csv)
    print(f"\nPatched {total_samples} sample(s) across {total_regions} "
          f"region(s).")
    print(f"Wrote: {patched_csv}")

    if args.skip_inject:
        print("\n(--skip-inject: stopping before writing the output MP4)")
        return

    # ---- Stage 3: Inject ----
    print(f"\n=== Stage 3/3: Injecting patched telemetry back into MP4 ===")
    inject_gyro_data(input_mp4, packets, rows, output_mp4)

    print(f"\nAll done:")
    print(f"  Raw telemetry:     {raw_csv}")
    print(f"  Patched telemetry: {patched_csv}")
    print(f"  Patched video:     {output_mp4}")


if __name__ == "__main__":
    main()
