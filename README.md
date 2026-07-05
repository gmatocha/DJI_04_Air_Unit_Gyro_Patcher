# DJI_04_Air_Unit_Gyro_Patcher
Cleans up Gyro data in MP4 files recorded by DJI Air Units with stabilization issue (mostly 2026 w/ 1469D gyro)

Demonstration video including raw, stabilized, and patched stabilized video here: https://youtu.be/W29mtHVhrRo

All code and documentation below was generated with Claude Sonnet 5 Medium. May contain errors.

Requirements:
- FFMPEG and associated tools (ffprobe) must be accessible in the path, or in the execution directory.
- Python 3.8+
- Video recorded with DJI Air Unit with Rocksteady turned off
- Stabilization then done in Gyroflow with your preferred settings. Test video was done w/ default settings 110% Zoom Limit

Test videos were 4k 100fps DLogM (Test videos not provided due to GitHub size restrictions.)


dji_gyro_fix.py - extracts, cleans up, and adds cleaned gyro data back to MP4 file
USAGE
---
    python3 dji_gyro_fix.py INPUT.MP4 [options]

This is the "single step" code. Use programs below for individual extract, fix, inject functions.

By default, given INPUT.MP4, this writes:
    INPUT_gyro_raw.csv       - the raw extracted telemetry
    INPUT_gyro_patched.csv   - the same data after patching
    INPUT_patched.MP4        - a copy of INPUT.MP4 with the patched
                               telemetry written back in


extract_dji_gyro.py - extractor for gyro data from DJI MP4 files.
USAGE
---
    python3 extract_dji_gyro.py INPUT.MP4 [OUTPUT.csv]

If OUTPUT.csv is omitted, it defaults to INPUT_gyro.csv next to the
input file.



inject_dji_gyro.py - Write gyro/attitude data from a CSV file (in the format produced by extract_dji_gyro.py) back into a DJI MP4 file
USAGE
---
    python3 inject_dji_gyro.py INPUT.MP4 DATA.csv OUTPUT.MP4




patch_dji_gyro_csv.py
---
Detect and patch the "bad gyro data" regions that DJI's O4-generation
cameras occasionally inject into their telemetry, working on a CSV
produced by extract_dji_gyro.py.

WHAT THE BAD DATA LOOKS LIKE
-----------------------------
Analysis of an affected recording showed a consistent signature at every
bad region:
  - The gyro magnitude (sqrt(gx^2+gy^2+gz^2)) spikes far above the
    file's normal peak (often 3-10x+ higher than anywhere else in the
    same recording).
  - It isn't a single corrupted sample: it's a short burst of rapid,
    oscillatory ("ringing") attitude change lasting anywhere from a few
    milliseconds to a couple of seconds.
  - The reported orientation does not necessarily return to its
    pre-event baseline afterward - the glitch can leave a permanent
    offset.

Because the bad segments vary a lot in duration and severity, but are
always anomalously large in *rate* compared to the rest of the same
file, this script detects them adaptively (relative to each file's own
baseline) rather than using one fixed hard-coded threshold.

HOW DETECTION WORKS
--------------------
1. Compute gyro magnitude per sample and establish a baseline (the 99th
   percentile of the whole file, away from extreme outliers).
2. Flag samples whose magnitude exceeds baseline * threshold-multiplier
   as the "core" of a bad region, and merge nearby flagged samples
   (gaps smaller than --merge-gap) into one core region.
3. AGGRESSIVELY grow each core region outward through its decaying
   "ringdown" tail: keep extending the cut as long as the local gyro
   magnitude stays above a lower expand-threshold (baseline *
   --expand-multiplier), only stopping once things have been genuinely
   quiet for --quiet-duration seconds. This is what catches the
   lower-amplitude wobble that lingers after the main spike but is
   still visibly shaky on video - a fixed padding window often stops
   short of where the ringing actually settles.
4. SAFETY VALVE: a real, physically-plausible camera/drone rotation
   can't hold a huge angular rate for very long - inertia limits how
   fast a large amplitude excursion can be sustained. So "high
   amplitude AND short duration" together are a much stronger, safer
   signal of a glitch than amplitude alone, and let the amplitude
   threshold be pushed much lower (catching fainter bursts) without
   also catching genuine sustained camera motion elsewhere in the
   file. Any candidate region longer than --max-duration is treated as
   suspicious and reported separately rather than silently patched,
   in case it's real motion rather than a glitch.

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
    --threshold-multiplier N    Flag samples above N x the 99th
                                 percentile gyro magnitude as a region's
                                 core (default 1.5)
    --absolute-floor N          Never flag below this deg/s value even
                                 on a very noisy file (default 300.0)
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




