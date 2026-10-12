# BorealHDR

## Setup and reusing a reference

Run these commands from the VSLAM-LAB checkout. `borealhdr run` is headless;
`borealhdr demo` requires a desktop terminal with an X display. Both commands
prepare the stereo images, generate or reuse the FAST-LIO reference, run SLAM,
and evaluate the result. A separate `boreal-fastlio` invocation is unnecessary.

On a machine without a reusable *shared or local* reference, install native ROS 2 Humble and the
pinned SPARK FAST-LIO build on Ubuntu 22.04 using
[`Utilities/setup_fastlio_humble.sh`](../Utilities/setup_fastlio_humble.sh).
This is a system/workspace dependency, not part of the Pixi environment. The
current Boreal runner expects that build at `/home/yashturkar/humble_ws` and
checks `/opt/ros/humble/setup.zsh` before starting a new FAST-LIO run. The
installer's `FAST_LIO_WORKSPACE` override does **not** change the runner's
hard-coded path yet.

For the April backpack sequence, the current defaults expect:

- Raw sequence: `/mnt/share/local/eph/BorealHDR/backpack_2023-04-20-09-29-14`
- Boreal calibration checkout: `/home/yashturkar/Workspace/TFR24_BorealHDR/BorealHDR`
- Prepared images and result output: `/mnt/share/local/eph/VSLAM/borealhdr`
- Shared reference sidecars: `/mnt/share/nas/eph/BorealHDR-references/fastlio-v1`

`--root`, `--code`, and `--output` override the first three base directories,
respectively; `--reference-root` overrides the read-only shared lookup directory.
With a verified shared bundle, the following command prints
`Boreal FAST-LIO: reusing shared ...` and does not launch FAST-LIO again:

```bash
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-09-29-14 \
  --slam orbslam3 --mode stereo --shared-only
```

The April reference is published under
`/mnt/share/nas/eph/BorealHDR-references/fastlio-v1/backpack_2023-04-20-09-29-14/`.
Keep the **whole sidecar directory**, including `manifest.json` and
`trajectory.csv`. Reuse checks the path-independent sensor fingerprint (full IMU
hash, all lidar scan names and sizes, and hashes of three scans), FAST-LIO config
hash, trajectory hash, and quality checks. This supports the same raw sequence at
a different absolute path; it does not prove every lidar CSV byte is identical.
If a shared bundle is present but invalid, the command fails rather than silently
replacing or regenerating it. If no shared bundle exists, it reuses a verified
local run or generates one with ROS 2/FAST-LIO. Prepared camera poses and SLAM
results remain local. The runner never writes to the shared reference directory.
Pass `--shared-only` to `run` or `demo` to require a validated shared sidecar:
an unpublished sequence fails before SLAM and will not start FAST-LIO. This is
the recommended mode while working with the currently validated subset.

To check the published April bundle without generating anything:

```bash
pixi run -e vslamlab boreal-reference-batch \
  --sequence backpack_2023-04-20-09-29-14 --check-only
```

The [local batch report](/mnt/share/local/eph/VSLAM/borealhdr/reference_batch/report.md)
lists pose count, temporal coverage, and GNSS-aligned horizontal RMSE. The bundle
also includes its config, logs, generation manifest, and GNSS comparison PNG/PDF.
GNSS is a sanity check, not surveyed ground truth. The April pilot has 143 poses,
98.6% sensor-span coverage, 0.56 m aligned 2D difference from GNSS, and 521/528
selected camera timestamps covered by interpolation. Camera extrinsics are
URDF-derived and should still be independently checked before treating camera
orientation errors as definitive.

To generate, validate, and publish all available sequences, run this yourself:

```bash
pixi run -e vslamlab boreal-reference-batch --all
```

The batch command resumes by skipping verified sidecars, reports progress per
sequence, continues after a failure, and writes `report.json` and `report.md` to
`/mnt/share/local/eph/VSLAM/borealhdr/reference_batch`. A failed validation is
reported and **not published**. Use `--all --check-only` afterward to audit the
published set without running FAST-LIO or writing to NAS. Generating missing
references requires the native FAST-LIO installation above. The script publishes
only small reference bundles, never raw recordings or prepared images.

## Experimental FAST-LIO lidar reference

The released BorealHDR sequence archives include VLP-16 lidar scans and Xsens IMU
measurements but no pose trajectory. `boreal-fastlio` replays those CSV files to
the installed SPARK FAST-LIO build and records an independent lidar/IMU reference.
It is **not** the authors' published libpointmatcher reference or surveyed ground
truth. Generation outputs are stored under
`/mnt/share/local/eph/VSLAM/borealhdr/fastlio_reference`; validated sidecars
are separately published under the shared root above. Input CSVs are read only.

```bash
# 30 sensor seconds at half-speed, suitable for checking timestamps and motion.
pixi run -e vslamlab boreal-fastlio \
  /mnt/share/local/eph/BorealHDR/backpack_2023-04-20-09-29-14

# Entire sequence at real-time playback speed.
pixi run -e vslamlab boreal-fastlio \
  /mnt/share/local/eph/BorealHDR/backpack_2023-04-20-09-29-14 \
  --duration 0 --rate 1.0
```

Each run creates a timestamped directory containing `trajectory.csv` in VSLAM-LAB
nanosecond pose format, the original TUM path, `manifest.json`, and separate player,
mapper, and recorder logs. The `timestamp` column in `imu.csv` and lidar filenames
is used as the common sensor clock. Lidar CSV `time` values are per-point offsets
relative to a scan stamp near its end; the player moves the cloud stamp to the
first point and converts offsets to seconds after scan start. The FAST-LIO config
uses the VLP-16 mode, 16 rings, and the lidar-to-IMU transform from the upstream
backpack URDF. The raw FAST-LIO trajectory describes the IMU frame.

`borealhdr run` and `borealhdr demo` now find a verified full reference run or
generate it automatically. They transform IMU poses into the rectified left-camera
frame, interpolate at the exact selected image timestamps, and write the
prepared variant's `groundtruth.csv` and `groundtruth_reference.json`. Only image
timestamps within the FAST-LIO trajectory are included; the April sample covers
521 of 528 selected frames. The camera mounting transform is recorded in
[`configs/borealhdr_reference.yaml`](../configs/borealhdr_reference.yaml), and
the rectification rotation is read from that season's `left.yaml` calibration.
Generated camera references are fingerprinted and reused on repeat runs.

After SLAM, both `run` and `demo` call the same evaluator as
`eval-metrics-single`. Each result directory gets `metrics.json` (APE, RPE,
associations and temporal completion), `trajectory_report.pdf`, and `run.json`.
The reference is labelled `boreal_fast_lio` in metrics. Stereo evaluation uses
SE(3) alignment without scale correction; mono uses Sim(3). A low RMSE on a
short trajectory does not imply full-sequence tracking, so check
`temporal_completion_ratio` and matched poses as well.

For the previously tested exposure schedule:

```bash
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-09-29-14 \
  --slam orbslam2 --mode stereo \
  --exposure-yaml configs/borealhdr_exposure_demo_maria_test1.yaml
```

`prepare` still prepares images alone. For a generated `config.yaml`, the usual
`pixi run -e vslamlab eval-metrics-single <config.yaml>` also sees the prepared
camera reference, but that task runs SLAM again before evaluating.

The `borealhdr` task converts the recorded **4 ms** stereo bracket into a local
VSLAM-LAB dataset and runs the selected baseline. It does not emulate exposure or
run an exposure controller by default. A supplied exposure YAML overrides the
fixed bracket for both cameras. Defaults:

- Recordings: `/mnt/share/local/eph/BorealHDR`
- Calibration checkout: `/home/yashturkar/Workspace/TFR24_BorealHDR/BorealHDR`
- Prepared images and results: `/mnt/share/local/eph/VSLAM/borealhdr`

From the VSLAM-LAB repository:

```bash
# List available recordings
pixi run -e vslamlab borealhdr list

# Headless ORB-SLAM2 stereo, including preparation
pixi run -e vslamlab borealhdr run backpack_2023-04-20-09-29-14 --slam orbslam2

# GUI demo (requires an X display, e.g. a desktop terminal)
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-09-29-14 --slam orbslam2

# Choose a different registered baseline
pixi run -e vslamlab borealhdr run backpack_2023-04-20-09-29-14 --slam orbslam3
```

## Exposure schedule

Use a plain YAML mapping from **zero-based exposure-cycle frame number** to
**milliseconds**. An entry takes effect at that frame and holds until the next
entry; frame 0 is required. You may also supply an entry for every frame.

```yaml
0: 4.0
60: 8.0
120: 16.0
```

### Exposure YAML specification

Pass the file using `--exposure-yaml PATH` to `prepare`, `run`, or `demo`.
Relative paths are resolved from the current working directory. Omitting the
option selects the recorded 4 ms bracket for every frame.

| Item | Requirement / meaning |
| --- | --- |
| Document | One top-level YAML mapping; no `exposures:` wrapper, list, or separate camera sections. |
| Key | A unique, unquoted integer frame number, starting at zero. Strings (`"0"`), fractional keys, booleans, and duplicate keys are rejected. |
| Frame bounds | For a sequence with `N` synchronized cycles, every key must satisfy `0 <= frame_num < N`. An entry for frame `0` is required. |
| Value | A finite nonnegative YAML number in **milliseconds**, e.g. `4`, `8.0`, or `6.5`. Strings, booleans, negative values, NaN and infinity are rejected. Zero supports replaying a possible AE50 output and renders a black frame. |
| Application | The specified exposure starts at the keyed frame, inclusive, and holds until the next keyed frame. The last setting holds to the end. There is no interpolation between entries. |
| Ordering | Keys may appear in any order; frame numbers determine application order. Ascending order is recommended for readability. |
| Stereo | One requested exposure applies to both cameras. Per-camera exposure settings are not supported. |

Frame numbers refer to **exposure cycles**, not raw image filenames, nanosecond
timestamps, SLAM keyframes, or individual images across all six brackets.
For each needed bracket, left/right images are paired by identical timestamp
filenames, unpaired images are excluded, and pairs are sorted by timestamp.
Frame `i` selects pair `i` from the chosen bracket's table. Needed brackets must
have the same number of synchronized pairs as the reference 4 ms table; selected
timestamps must remain strictly increasing. Counts alone cannot detect matching
gaps in every bracket, so schedules assume the recording's bracket cycles are
aligned, as in the upstream emulator.

For the 528-cycle April sample, valid keys are `0` through `527`. The demo's
last change is at `480`, so that value applies to frames `480–527`. A shorter
sequence needs a schedule with all change points inside its own frame range.

Sparse schedules are convenient for intervals; a dense mapping is supported too:

```yaml
# First four frames, then hold the final setting through the remaining frames.
0: 4.0
1: 6.5
2: 8.0
3: 4.0
```

Recorded exposures are **1, 2, 4, 8, 16, and 32 ms**. Requests matching those values
use the corresponding recorded images. Other nonnegative values, including values
outside that recorded range, are emulated from the nearest bracket in log
exposure using the calibrated camera response curve. Zero uses the 1 ms source
bracket and renders black. Saturation is clipped;
emulation does not recover lost detail or reproduce changes in motion blur.
Choosing a different recorded bracket also changes the actual acquisition time
within the cycle. See the processing details below.

The resolved schedule is saved as `exposure.yaml` (one entry per frame), and
`exposure.csv` records `frame_num`, `exposure_ms`, `source_bracket_ms`,
`timestamp_ns`, `left_source`, and `right_source`. These files are copied into
each scheduled run's result directory. Editing a schedule selects a separate
cache when its effective exposures change; older prepared variants remain intact.

### Commands with a schedule

The supplied demo varies exposure from 1 to 32 ms over the 528-cycle April sample:

```bash
pixi run -e vslamlab borealhdr run backpack_2023-04-20-09-29-14 --slam orbslam2 --mode stereo --exposure-yaml configs/borealhdr_exposure_demo.yaml
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-09-29-14 --slam orbslam2 --mode stereo --exposure-yaml configs/borealhdr_exposure_demo.yaml
```

Each cycle groups the nth image in each recorded bracket, following BorealHDR's
emulator convention. Both cameras select the same bracket and requested exposure.
Recorded values (1, 2, 4, 8, 16, 32 ms) use those images directly. Other positive
values select the nearest bracket in log exposure and rescale radiance through
the upstream `pcalib_forest2024.txt` camera response curve, clipping saturation.
This is offline exposure emulation, not live camera control. Actual acquisition
timestamps are retained, so changing the selected bracket shifts sampling time
within its cycle. Unpaired camera images are excluded and reported before cycle
indexing. Unequal synchronized cycle counts are rejected.

Scheduled preparations use separate fingerprinted directories under
`exposure_variants/`, keeping previous runs' images intact. Calibration, source
files, response curve and requested exposures contribute to the fingerprint.
Each run saves the expanded `exposure.yaml` plus an `exposure.csv` listing the
requested exposure, source bracket, timestamp and source files for both cameras.

The terminal prints the current left/right exposure on every preparation exposure
change, every 50 pairs, and at completion. Frame numbers are zero-based. Cached
preparations print the input schedule by frame range before SLAM starts; these
are input summaries, not live SLAM playback progress.

### AE50 automatic exposure

`--ae` runs the **classical-50** controller from the
[pinned BorealHDR source](https://github.com/norlab-ulaval/BorealHDR/blob/042befac24efb256ab2a3c787c758a88a490c290/scripts/classes/class_auto_exposure_methods.py#L117-L159).
`--ae classical` is equivalent. `--ae zhang` and `--ae kim` select the other
two [controllers in that pinned revision](https://github.com/norlab-ulaval/BorealHDR/blob/042befac24efb256ab2a3c787c758a88a490c290/scripts/classes/class_auto_exposure_methods.py#L159-L416).
All three work with `prepare`, `run`, and `demo`, but cannot be combined with
`--exposure-yaml`:

```bash
pixi run -e vslamlab borealhdr prepare backpack_2023-04-20-14-07-00 --ae
pixi run -e vslamlab borealhdr run backpack_2023-04-20-14-07-00 --slam orbslam2 --mode stereo --shared-only --ae
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-14-07-00 --slam orbslam3 --mode stereo --shared-only --ae

# Alternative BorealHDR controllers (headless examples):
pixi run -e vslamlab borealhdr run backpack_2023-04-20-14-07-00 --slam pycuvslam --mode stereo --shared-only --ae zhang
pixi run -e vslamlab borealhdr run backpack_2023-04-20-14-07-00 --slam pycuvslam --mode stereo --shared-only --ae kim
```

Add `--ae-plot` to a `demo --ae` command to open a second window with the
per-frame exposure curve and a cursor on the frame currently being loaded by
ORB-SLAM2 or ORB-SLAM3:

```bash
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-14-07-00 --slam orbslam3 --mode stereo --shared-only --ae --ae-plot
```

The plot watches opens of the prepared `rgb_0` PNGs and maps each filename to
its exact exposure in `exposure.csv`; it does not estimate progress from a
timer or alter SLAM inputs. It marks the **image-read event**, not the exact
instant `TrackStereo` processes the image. Avoid opening those PNGs in another
application during the demo, because Linux inotify cannot identify which
process opened a file. The watcher starts before SLAM and closes when the run
ends. It requires Linux and a graphical desktop. To attach to an already
running AE demo, run `pixi run -e vslamlab python
Utilities/boreal_ae_plot.py <prepared_sequence_path>`; it can observe only
frames opened after it starts.

Frame 0 starts at 4 ms. Both cameras use the same requested exposure at each
frame; all controllers meter the emulated **left Bayer image** and choose the
next frame's exposure. Classical-50 converts that image to 12-bit gray with
`COLOR_BAYER_RG2GRAY` and meters its mean. The target is 2048 digital counts
(50% of 4096); errors strictly below 15 counts hold exposure. Otherwise the
next frame uses the pinned controller's `abs(current_ms + 0.002 ×
(2048 − measured_mean))`. There is **no controller clamp**: requested times can
exceed the 1–32 ms recorded bracket range or equal zero. The image renderer
uses an available bracket and the response curve to emulate that time; zero
renders black. Exposures are controlled in milliseconds, not stops.

For `--ae`, image emulation now also follows the pinned BorealHDR
`Image_Emulator`: it uses `calibration_files/pcalib_inside1.txt` and selects
the first recorded bracket at or above the requested time, falling back to the
previous bracket if that candidate has more zero/4094-valued pixels. Above
32 ms it uses the 32 ms bracket. This selection is made **separately for left
and right**, as in the upstream code, so the two source images in a stereo pair
can have different exposure brackets and acquisition timestamps. The requested
exposure time remains the same for both cameras; `rgb.csv`, `exposure.csv`, and
`ae_trace.csv` record the selected sources. Fixed and YAML-scheduled runs keep
the harness's previous `pcalib_forest2024.txt`/nearest-log behavior.

Zhang uses the pinned soft-percentile gradient/CRF-derivative update, switches
to classical-50 when mean brightness falls outside its 70–190/256 bounds, and
enforces its 0.02 ms minimum. The pinned code hard-codes a weight vector for
1920×1200 pixels; the supplied Boreal Bayer PNGs are 960×600, so this harness
computes the **same weight formula at the actual image size**. Without that
size adaptation, the pinned code cannot multiply the per-pixel gradients for
these files. Kim uses the pinned entropy-weighted gradient score, Gaussian
process with ten optimizer restarts, MAXMI acquisition, and five-observation
window. Its upstream GP restarts are unseeded; this harness uses a persistent
seed of 0 so identical inputs produce reproducible cached exposures. These
two adaptations mean Zhang/Kim are faithful implementations of the pinned
controller logic, **not bit-for-bit invocations of unmodified upstream code**.
Kim's optimization can make preparation substantially slower than classical-50.

**Kim interpretation caveat.** Its five-observation GP can jump directly to
either end of the 0.5–35 ms search range; this implementation deliberately does
not add a step limiter or clipped-pixel veto, because those would change the
pinned baseline. In one `backpack_2023-04-20-14-07-00` PyCuVSLAM run, Kim jumped
from 4.713 to 35 ms at frame 310. The selected 32 ms source image was already
90.9% clipped, yet Kim's image-quality metric increased and it stayed at 35 ms
for the remaining 33 frames. The resulting translation APE RMSE was 14.57 m
despite 100% temporal completion. A seeded, full 344-frame replay through the
upstream `Metric_Kim` produced exactly the same exposure decisions, so this is
an observed limitation of this controller on those inputs, not evidence of a
porting discrepancy. Inspect `ae_trace.csv`, saturation, and path-length ratio
alongside SLAM metrics when comparing controllers.

The earlier AE implementation reproduced the pinned **controller** but used
the harness's existing forest CRF and nearest-log bracket choice. Those were
reasonable implementation defaults, not an exact BorealHDR AE emulator match;
old AE image variants and results should not be cited as such. The new AE
fingerprint prevents old prepared images from being silently reused. This
still does not guarantee bit-for-bit agreement with BorealHDR's complete
upstream pipeline: this harness aligns complete stereo capture cycles and
rectifies/exports images for VSLAM-LAB. AE is
computed sequentially while preparing the dataset, then SLAM runs on those
static images; it is not feedback from SLAM or a live camera. Brackets are
aligned to the 4 ms capture-cycle timestamps; cycles missing any paired bracket
are excluded before AE starts, so a missing frame cannot shift later cycles.
The generated `exposure.yaml` records the per-frame requested times. For an
**exact rerun**, reuse the prepared sequence and its saved run configuration:
ordinary `--exposure-yaml` preparation can differ because AE meters before
demosaicing and excludes incomplete cycles. `ae_trace.csv` records measured
mean, the classical-50 target difference (diagnostic only for Zhang/Kim),
applied and next exposure (ms), stops relative to 4 ms (blank at zero), and
source bracket. Zhang/Kim traces also record method, controller metric, and
which control branch was used; `borealhdr.json` and `run.json` record the
method, pinned-source provenance, and Kim GP seed.
Both files, `exposure.csv`, and controller provenance are copied into each run
directory. AE preparations have their own fingerprinted cache.

If the native SLAM process crashes, the command now stops before evaluation and
reports its exit code/signal and log path. `run.json` records these details;
any partial keyframe trajectory is retained for diagnosis, not treated as a
valid evaluation result.

Use `--mode mono` for a monocular baseline; stereo is the default. `--root`,
`--code`, and `--output` override the paths above. `prepare` converts a sequence
without running SLAM. Verified prepared frames are reused across methods and runs.
Each run gets a separate timestamped directory under `results/<sequence>/`, with
`config.yaml`, `run.json`, the trajectory CSV, a trajectory PDF and runner logs.
The generated config also works with `demo-single`.

Run output and `run.json` report pose count, input/trajectory spans, reference
coverage, RMSE, and temporal completion. The initial full ORB-SLAM2 check on the
April sample processed 528 pairs but retained only 7 keyframes spanning 5.5
seconds. This verifies execution, not robust tracking across that recording; the
fixed 4 ms bracket can be unsuitable for some lighting conditions.

Conversion uses exact matching nanosecond filenames for stereo pairing. Unmatched
frames are counted in `borealhdr.json`. Raw Bayer RG images are demosaiced to gray,
mapped from 12-bit to 8-bit using a fixed division by 16, and rectified using the
April or September calibration from the source checkout. Intrinsics are scaled
to the actual PNG dimensions; both cameras use a common rectified intrinsic
matrix. Source images are not changed.

The inspected local recordings contain raw LiDAR, GNSS and IMU measurements but
no supplied pose ground-truth trajectory. FAST-LIO generates a locally derived
reference; its accuracy is not equivalent to surveyed ground truth or the authors'
published lidar reference.
