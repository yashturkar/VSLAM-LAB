# BorealHDR

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
| Value | A finite positive YAML number in **milliseconds**, e.g. `4`, `8.0`, or `6.5`. Strings, booleans, zero, negative values, NaN and infinity are rejected. |
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
use the corresponding recorded images. Other positive values, including values
outside that recorded range, are emulated from the nearest bracket in log
exposure using the calibrated camera response curve. Saturation is clipped;
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

Use `--mode mono` for a monocular baseline; stereo is the default. `--root`,
`--code`, and `--output` override the paths above. `prepare` converts a sequence
without running SLAM. Verified prepared frames are reused across methods and runs.
Each run gets a separate timestamped directory under `results/<sequence>/`, with
`config.yaml`, `run.json`, the trajectory CSV, a trajectory PDF and runner logs.
The generated config also works with `demo-single`.

Run output and `run.json` report the saved pose count and input/trajectory time
spans. The initial full ORB-SLAM2 check on the April sample processed 528 pairs but
retained only 7 keyframes spanning 5.5 seconds. This verifies execution, not robust
tracking across that recording; the fixed 4 ms bracket can be unsuitable for some
lighting conditions.

Conversion uses exact matching nanosecond filenames for stereo pairing. Unmatched
frames are counted in `borealhdr.json`. Raw Bayer RG images are demosaiced to gray,
mapped from 12-bit to 8-bit using a fixed division by 16, and rectified using the
April or September calibration from the source checkout. Intrinsics are scaled
to the actual PNG dimensions; both cameras use a common rectified intrinsic
matrix. Source images are not changed.

The inspected local recordings contain raw LiDAR, GNSS and IMU measurements but
no pose ground-truth trajectory. This integration does not turn those measurements
into a reference trajectory: plots show SLAM only and no reference RMSE is claimed.
`eval-metrics-single` still requires a supplied `groundtruth.csv` in the prepared
sequence. Automatic exposure control is deferred.
