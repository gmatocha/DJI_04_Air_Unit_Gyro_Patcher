#!/usr/bin/env python3
"""
patch_dji_gyro_csv.py

Detect and patch the "bad gyro data" regions that DJI's O4-generation
cameras occasionally inject into their telemetry, working on a CSV
produced by extract_dji_gyro.py.

WHAT THE BAD DATA LOOKS LIKE
-----------------------------
Analysis of an affected recording showed a consistent signature at every
bad region:
  - It isn't a single corrupted sample: it's a short burst of rapid,
    oscillatory ("ringing") attitude change lasting anywhere from a few
    milliseconds to a couple of seconds, where the gyro rate swings
    wildly (often reversing sign within a few milliseconds).
  - The reported orientation does not necessarily return to its
    pre-event baseline afterward - the glitch can leave a permanent
    offset.
  - Critically, this is different from a genuine fast camera/drone
    rotation (e.g. an intentional spin or pan), which can also reach a
    high gyro rate but does so *smoothly* - the rate tracks its own
    recent trend closely rather than oscillating around it.

HOW DETECTION WORKS
--------------------
1. RESIDUAL SIGNAL: rather than flagging on raw gyro magnitude (which
   can't tell a genuine fast rotation from a glitch - both can reach a
   high instantaneous rate), this script computes each sample's gyro
   vector minus a short local moving-average trend (--residual-window,
   default 40ms), and detects on the MAGNITUDE OF THAT RESIDUAL. A
   smooth rotation, however fast, tracks its own trend closely so the
   residual stays small (empirically under ~200 deg/s even during a
   deliberate multi-second spin). The oscillatory glitch swings wildly
   around its own trend, so the residual spikes hugely (500-2000+
   deg/s). This is what lets detection be pushed much more aggressive
   without risking real camera motion getting patched away.
2. Establish a baseline (99th percentile of the whole file's residual)
   and flag samples whose residual exceeds baseline * threshold-
   multiplier as the "core" of a bad region, merging nearby flagged
   samples (gaps smaller than --merge-gap) into one core region.
3. Grow each core region outward through its decaying "ringdown" tail:
   keep extending the cut as long as the local residual stays above a
   lower expand-threshold (baseline * --expand-multiplier), only
   stopping once things have been genuinely quiet for --quiet-duration
   seconds. This is what catches the lower-amplitude wobble that
   lingers after the main spike but is still visibly shaky on video - a
   fixed padding window often stops short of where the ringing actually
   settles.
4. SAFETY VALVE: a real, physically-plausible camera/drone rotation
   can't hold a huge residual for very long. Any candidate region
   longer than --max-duration is treated as suspicious and reported
   separately rather than silently patched, in case it's real motion
   rather than a glitch.
5. MULTI-PASS REFINEMENT: the baseline is computed from the whole file,
   so it gets pulled upward by the big glitches themselves - a
   second-tier of subtler jitter can hide underneath that inflated
   baseline on the first pass. Each --passes iteration re-patches the
   big stuff first, recomputes the baseline from the now-cleaner data,
   and re-detects - exposing smaller residual glitches that were
   previously masked. Iteration stops early once a pass finds nothing
   new.

HOW PATCHING WORKS
-------------------
For each (now fully-grown) bad region, take the last good quaternion
sample before it and the first good quaternion sample after it, and
replace every sample inside the region with a spherical interpolation
(SLERP) between those two - i.e. the smooth, physically-plausible
attitude path the camera would have taken if the glitch hadn't
happened. Then recompute roll/pitch/yaw and gyro_x/y/z for the entire
file from the (now partly-patched) quaternion sequence.

This does NOT try to guess what the drone "really" did during the
glitch - it just bridges smoothly across it, which is the right
behavior for feeding clean-looking data back into the MP4 (e.g. for
stabilization/EIS purposes) even though it discards whatever real
motion (if any) happened during that window.

USAGE
-----
    python3 patch_dji_gyro_csv.py INPUT.csv OUTPUT.csv [options]

Options:
    --residual-window N          Smoothing window (s) for the local
                                 trend that residual is computed against
                                 (default 0.04)
    --threshold-multiplier N    Flag samples above N x the 99th
                                 percentile RESIDUAL magnitude as a
                                 region's core (default 1.5)
    --absolute-floor N          Never flag below this residual deg/s
                                 value (default 220.0)
    --merge-gap N                Merge flagged samples into one core
                                 region if within N seconds of each
                                 other (default 0.5)
    --expand-multiplier N       Grow each region outward through its
                                 ringdown tail while local magnitude
                                 stays above N x baseline. Lower = more
                                 aggressive/cuts more (default 1.15)
    --quiet-duration N           Stop growing once magnitude has been
                                 quiet for N seconds (default 0.05)
    --smoothing-window N         Rolling-max window (s) used while
                                 growing, to ride through brief dips
                                 mid-ringdown (default 0.01)
    --max-duration N             Candidate regions (before ringdown
                                 growth) longer than N seconds are
                                 treated as possible real motion and
                                 reported instead of auto-patched,
                                 unless --force-long-regions is given
                                 (default 2.5)
    --force-long-regions         Patch regions longer than
                                 --max-duration anyway instead of just
                                 warning about them
    --report-only                Detect and print regions, but do not
                                 write a patched CSV.

Requires: Python 3.8+. No external dependencies.
"""

import sys
import csv
import math
import argparse


FIELDNAMES = [
    "frame_index", "sample_index", "timestamp_s",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s",
]


# --------------------------------------------------------------------------
# Quaternion math
# --------------------------------------------------------------------------

def quat_to_euler_deg(w, x, y, z):
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

    # Take the shortest path
    if dot < 0.0:
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        # Very close: linear interpolation + normalize is numerically safer
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
    if dt <= 0:
        return 0.0, 0.0, 0.0
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dw = (w2 - w1) / dt
    dx = (x2 - x1) / dt
    dy = (y2 - y1) / dt
    dz = (z2 - z1) / dt
    w, x, y, z = w1, -x1, -y1, -z1
    ow = dw * w - dx * x - dy * y - dz * z
    ox = dw * x + dx * w + dy * z - dz * y
    oy = dw * y - dx * z + dy * w + dz * x
    oz = dw * z + dx * y - dy * x + dz * w
    return (2 * ox * 180 / math.pi,
            2 * oy * 180 / math.pi,
            2 * oz * 180 / math.pi)


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------

def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "frame_index": int(float(row["frame_index"])),
                "sample_index": int(float(row["sample_index"])),
                "timestamp_s": float(row["timestamp_s"]),
                "quat_w": float(row["quat_w"]),
                "quat_x": float(row["quat_x"]),
                "quat_y": float(row["quat_y"]),
                "quat_z": float(row["quat_z"]),
            })
    return rows


def write_rows(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def compute_gyro_components(rows):
    """Compute per-axis instantaneous gyro rate (gx, gy, gz) per sample
    using consecutive quaternion differences."""
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


def compute_magnitudes(rows):
    """Compute an instantaneous gyro-rate magnitude per sample using
    consecutive quaternion differences (independent of whatever gyro_*
    columns may already be in the source CSV)."""
    gx, gy, gz = compute_gyro_components(rows)
    return [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(gx, gy, gz)]


def moving_average(values, window):
    """O(n) centered moving average via a running sum."""
    n = len(values)
    if window < 1:
        return list(values)
    half = window // 2
    out = [0.0] * n
    running = 0.0
    # initial window
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
    the remainder.

    This is a much better detection signal than raw magnitude: a real,
    smooth, even fast rotation (e.g. a deliberate multi-second pan or
    spin) closely tracks its own local average, so the residual stays
    small - but the short, oscillatory "ringing" bursts characteristic
    of the bad-data glitch swing wildly around their local average
    (often reversing sign within a few milliseconds), so the residual
    spikes hugely. This lets detection be far more aggressive without
    risking flagging genuine sustained camera motion.
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
    """Max magnitude within +/- half_window_s seconds of index i."""
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
    """Grow [lo, hi] outward through a decaying ringdown tail. Keeps
    extending in each direction as long as a local rolling-max magnitude
    stays above expand_threshold; stops once magnitude has been quiet
    (below threshold) for at least quiet_duration seconds."""
    times = [r["timestamp_s"] for r in rows]
    n = len(rows)
    half_win = smoothing_window / 2.0

    # Expand backward
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
            # stop once we've been quiet for quiet_duration seconds
            if (quiet_since - times[i]) >= quiet_duration:
                break

    # Expand forward
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

    # Merge seed detections into contiguous core regions based on timestamp gap
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

    # Duration safety check BEFORE ringdown growth: a candidate region
    # that is already long at its raw/core stage (high amplitude
    # sustained for a long time) is a poor fit for "short duration
    # noise" and more likely to be real motion - flag it separately.
    accepted_core = []
    suspicious = []
    for (lo, hi) in core_regions:
        dur = rows[hi]["timestamp_s"] - rows[lo]["timestamp_s"]
        if dur > max_duration and not force_long_regions:
            suspicious.append((lo, hi, dur))
        else:
            accepted_core.append((lo, hi))

    # Grow each accepted core region outward through its decaying
    # ringdown tail
    expanded = []
    for (lo, hi) in accepted_core:
        new_lo, new_hi = expand_region(
            rows, mags, lo, hi, expand_threshold, quiet_duration,
            smoothing_window,
        )
        expanded.append((new_lo, new_hi))

    if not expanded:
        return [], suspicious, baseline_p99, threshold, expand_threshold

    # Merge any expanded regions that now overlap/touch
    merged = [expanded[0]]
    for (lo, hi) in expanded[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi + 1:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))

    return merged, suspicious, baseline_p99, threshold, expand_threshold


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------

def patch_regions(rows, regions):
    n = len(rows)
    for (lo, hi) in regions:
        before_idx = lo - 1
        after_idx = hi + 1
        if before_idx < 0 or after_idx >= n:
            # Can't safely interpolate at the very start/end of the file;
            # hold the nearest good value instead.
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect and patch bad gyro/attitude bursts in a "
                    "DJI gyro CSV (as produced by extract_dji_gyro.py)."
    )
    parser.add_argument("input_csv")
    parser.add_argument("output_csv", nargs="?", default=None)
    parser.add_argument("--residual-window", type=float, default=0.04,
                        help="Smoothing window (seconds) used to compute "
                             "each sample's local trend; detection runs "
                             "on the RESIDUAL (raw minus this trend), not "
                             "raw magnitude. This is what tells apart a "
                             "genuine fast rotation (which tracks its own "
                             "trend closely, so residual stays small) "
                             "from glitchy ringing (which swings wildly "
                             "around its trend). (default 0.04)")
    parser.add_argument("--threshold-multiplier", type=float, default=1.5,
                        help="Flag samples above N x the file's 99th "
                             "percentile RESIDUAL magnitude as the "
                             "initial 'core' of a bad region (default 1.5)")
    parser.add_argument("--absolute-floor", type=float, default=220.0,
                        help="Never flag below this residual deg/s value "
                             "(default 220.0 - set just above the largest "
                             "residual peak observed during genuine fast "
                             "camera motion)")
    parser.add_argument("--merge-gap", type=float, default=0.5,
                        help="Merge flagged samples within N seconds of "
                             "each other into one core region (default 0.5)")
    parser.add_argument("--expand-multiplier", type=float, default=1.15,
                        help="Once a bad region's core is found, keep "
                             "growing it outward through the decaying "
                             "ringdown tail as long as the local residual "
                             "magnitude stays above N x baseline "
                             "(default 1.15 - more aggressive/lower = cuts "
                             "more of the tail)")
    parser.add_argument("--quiet-duration", type=float, default=0.05,
                        help="Stop growing a region once magnitude has "
                             "been below the expand-threshold for this "
                             "many consecutive seconds (default 0.05)")
    parser.add_argument("--smoothing-window", type=float, default=0.01,
                        help="Local rolling-max window (seconds) used "
                             "while growing regions, to avoid stopping "
                             "on a single quiet sample mid-ringdown "
                             "(default 0.01)")
    parser.add_argument("--max-duration", type=float, default=2.5,
                        help="Candidate regions longer than this (before "
                             "ringdown growth) are treated as possible "
                             "real motion and reported instead of "
                             "auto-patched (default 2.5)")
    parser.add_argument("--force-long-regions", action="store_true",
                        help="Patch regions longer than --max-duration "
                             "anyway instead of just warning about them")
    parser.add_argument("--passes", type=int, default=2,
                        help="Run detection multiple times, recomputing "
                             "the baseline from the already-patched data "
                             "each time, exposing subtler residual "
                             "anomalies that were masked by bigger ones "
                             "on the first pass. Stops early if a pass "
                             "finds nothing new. (default 2)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only detect and print regions; don't "
                             "write an output CSV")
    args = parser.parse_args()

    output_csv = args.output_csv
    if output_csv is None and not args.report_only:
        base = args.input_csv
        if base.lower().endswith(".csv"):
            base = base[:-4]
        output_csv = base + "_patched.csv"

    print(f"Reading {args.input_csv} ...")
    rows = load_rows(args.input_csv)
    print(f"Loaded {len(rows)} samples "
          f"({rows[0]['timestamp_s']:.3f}s - {rows[-1]['timestamp_s']:.3f}s)")

    total_samples = 0
    total_regions = 0
    pass_num = 0

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

        print(f"\n--- Pass {pass_num} ---")
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
            print("(--report-only: no output file written; not "
                  "iterating further passes)")
            break

        patch_regions(rows, regions)
        recompute_derived_columns(rows)

        total_samples += sum(hi - lo + 1 for (lo, hi) in regions)
        total_regions += len(regions)

    if args.report_only:
        return

    write_rows(rows, output_csv)
    print(f"\nPatched {total_samples} sample(s) across {total_regions} "
          f"region(s) total over {pass_num} pass(es).")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
