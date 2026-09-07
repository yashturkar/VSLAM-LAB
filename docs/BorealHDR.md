# BorealHDR

The `borealhdr` task converts the recorded **4 ms** stereo bracket into a local
VSLAM-LAB dataset and runs the selected baseline. It does not emulate exposure or
run an exposure controller. Defaults:

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
sequence. Exposure emulation and control are deferred.
