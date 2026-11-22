# CLID component relighting

`clid-relight` turns CLID V6.3 component archives into rectified VSLAM-LAB mono
sequences. It accepts independent onboard-light power and exposure EV controls at
sparse frame change points, caches the rendered sequence, and can run and evaluate
a selectable monocular SLAM implementation.

The current interchange contract is
`/mnt/share/local/eph/VSLAM/CLID_V63_RELIGHTING_INTERCHANGE_SPEC.md`. Inputs must
contain `rgb.csv`, `groundtruth.csv`, `extraction_metadata.json`, `raw_0`,
`clid_components_0`, and a complete `clid_components_manifest.json`. The source is
read-only. Generated files default to `/mnt/share/local/eph/VSLAM/relighting`.

## Schedule YAML

The YAML is a top-level mapping of zero-based frame numbers to complete control
settings:

```yaml
0:
  light: 0.0
  exposure_ev: -2.0
100:
  light: 0.5
  exposure_ev: 0.0
200:
  light: 1.0
  exposure_ev: 2.0
```

Frame 0 is required. Every entry must contain exactly `light` and `exposure_ev`;
the pair holds until the next entry. Frame keys must be unique, unquoted,
nonnegative integers within the sequence. `light` is commanded hardware power in
`[0,1]`, not calibrated effective power. `exposure_ev` is a finite stop value in
`[-2,+2]`. Booleans, strings, NaN, infinity, duplicate keys, and partial settings
are rejected. See [`configs/clid_relight_demo.yaml`](../configs/clid_relight_demo.yaml)
for a 760-frame sweep.

The renderer interpolates commanded light through the checked-in calibrated power
response and evaluates:

```text
H(q) = A + qF + q(1-q)C1 + q(1-q)(2q-1)C2
LDR  = clip(2^EV H(q), 0, 1)
```

## Commands

From the repository root:

```bash
SEQUENCE=/mnt/share/local/eph/VSLAM/runs/kitchen2corr_20260915_203647_seq001/vslamlab/kitchen2corr_seq001

# Render or verify the cached input only.
pixi run -e vslamlab clid-relight prepare "$SEQUENCE" \
  --schedule-yaml configs/clid_relight_demo.yaml \
  --profile viewer-rgb

# Render/reuse, run ORB-SLAM2 mono headlessly, then evaluate against odometry.
pixi run -e vslamlab clid-relight run "$SEQUENCE" \
  --schedule-yaml configs/clid_relight_demo.yaml \
  --profile viewer-rgb --slam orbslam2

# GUI run followed by the same evaluation when the window closes.
pixi run -e vslamlab clid-relight demo "$SEQUENCE" \
  --schedule-yaml configs/clid_relight_demo.yaml \
  --profile viewer-rgb --slam orbslam2
```

`viewer-rgb` (the default) follows the V6.3 viewer white balance, tone map, and
sRGB transfer function, but stores lossless PNG rather than JPEG. `linear-gray`
is an experimental SLAM-analysis profile: it converts demosaiced scene-linear RGB
directly to 8-bit gray with no per-frame normalization.

The packed components unpack to 1280x800, while CameraInfo describes 1920x1200.
The utility reads the original unrectified CameraInfo from the staged MCAP, scales
its camera and projection matrices uniformly by 2/3, rectifies at 1280x800, and
writes a corresponding zero-distortion mono calibration. Exact frame identity and
timestamps come from `rgb.csv`, not the manifest's rounded timestamp field.

Variants are content-addressed by source/component identity, schedule, profile,
renderer, relighting calibration, and camera calibration. Preparation is locked,
uses atomic image writes, validates cached images, and resumes partial work.
Each variant includes `relight.csv`, a dense resolved `schedule.yaml`, and
`relight.json` provenance. Each run gets a timestamped results directory containing
the trajectory, Sim(3)-aligned monocular EVO metrics, temporal completion, PDF,
configuration, and run state.

## Legacy V6.3 mono command limitation

Only camera-0 components exist. `--mode stereo` deliberately fails instead of
mixing synthesized left frames with recorded right frames. Stereo support is
complete only after synchronized `clid_components_1` files and their manifest
contract are available; the same schedule must then be applied to both cameras
while preserving the calibrated baseline.

## V6.3.4 release-2 stereo pipeline

The older `clid-relight` command above remains the V6.3 mono consumer. For a
processed CLID sequence captured by the two release-2 camera serials, use the
separate `clid-stereo` pipeline. It runs IllumiNet inference on each camera's
BayerRG16 RAW and exact metadata, pairs the verified coefficient manifests by
their RAW header timestamps, and scans bag odometry without extracting preview
images. It never pairs by frame number. The same
light/EV schedule is applied to both cameras, but each uses its own calibrated
power response and Bayer response curve. The 400×640 packed coefficients are
quantized to float16 in memory before relighting, demosaiced to 1280×800,
rectified, and paired with a calibration scaled from the bag's CameraInfo.
There is no per-camera or per-frame display-exposure adjustment.

Inputs are read-only, including the NAS bag and the release-2 artifacts at
`/mnt/share/nas/eph/clid-hdr-releases/clid-hdr-v634-release-2` (override with
`--artifacts`). The release manifest is SHA-256 pinned. IllumiNet inference is
installed from a fixed commit on the
[`release/clid-hdr-v634-stereo` branch](https://github.com/yashturkar/IllumiNet/tree/release/clid-hdr-v634-stereo)
by the separate `clid-infer` Pixi CUDA
environment, so users do not need an IllumiNet checkout or a separate Python
environment. On first inference, Pixi installs that environment; this path
requires Linux, an NVIDIA GPU supported by the pinned PyTorch CUDA 12.8 wheel,
and a CUDA 12.8-compatible driver. A GPU/FFmpeg preflight runs before inference.
The optional `--illuminet` and `--illuminet-python` flags are only a
paired development override.

Choose a machine-local output root with `--local-root` or `VSLAM_LOCAL_ROOT`.
If this checkout has a local `.vslamlab-storage` setting, its `clid_stereo/`
subdirectory is used instead. Otherwise the command refuses to guess where
large (tens of GiB) coefficient exports should go. NAS and network-mounted
output paths are rejected, as is an unmounted `/mnt/share/local` path. Before
new inference, the command checks that both RAW topics exist and that estimated
coefficient/review space is available; the estimate is approximate.

```bash
BAG=/mnt/share/nas/eph/clid-v2-sequences/113through/113through-3_20260826_214558/113through-3_seq001_20260826_214616_889091
VSLAM_LOCAL_ROOT=/mnt/share/local/eph/VSLAM/clid_stereo  # replace with your local disk path
export VSLAM_LOCAL_ROOT

pixi run -e vslamlab clid-stereo check "$BAG"
pixi run -e vslamlab clid-stereo prepare-coefficients "$BAG"
pixi run -e vslamlab clid-stereo prepare "$BAG" \
  --schedule-yaml configs/clid_stereo_demo_113through.yaml
pixi run -e vslamlab clid-stereo run "$BAG" \
  --schedule-yaml configs/clid_stereo_demo_113through.yaml --slam orbslam3
```

`prepare-coefficients` needs no schedule YAML: it runs or verifies the full
left/right IllumiNet coefficient export, then stops before bag odometry indexing,
image rendering, or SLAM. It prints the coefficient directory. By default it is
`<VSLAM_LOCAL_ROOT>/<source-name>_<source-fingerprint>/coefficients/`;
for the example bag it is
`/mnt/share/local/eph/VSLAM/clid_stereo/113through-3_seq001_20260826_214616_889091_033585b54a22/coefficients/`.
This contains `left/frame-000000.npz` and `right/frame-000000.npz` through
frame 574, `left/manifest.json`, `right/manifest.json`, and
`stereo-manifest.json`. Review artifacts are stored beside it under `review/`.
The output root is controlled by `--local-root` or `VSLAM_LOCAL_ROOT`; a partial/stale export is
preserved and reported rather than overwritten. Do not pass `--schedule-yaml`
to this action. Later `prepare` or `run` reuses verified coefficients.

`prepare` does not launch SLAM. `run` launches headless stereo SLAM and the
usual VSLAM-LAB evaluation; `demo` launches the viewer. `check`/`status` show
missing, partial/stale, or ready RAW index and inference. Complete stages
are verified and skipped. Inference writes fresh immutable coefficient/review
directories; if it fails partway, the pipeline preserves them for inspection
and refuses to overwrite them. Choose a fresh local root after diagnosing a
partial export. Relit variants are content-addressed by schedule and inputs.
The printed prepared sequence path contains `rgb_0`, `rgb_1`, exact-pair
`rgb.csv`, scaled `calibration.yaml`, `groundtruth.csv`, `relight.csv`, and
`relight.json`.
For another sequence, provide a schedule whose largest frame index is below
its exact paired-RAW frame count; frame 0 is required, and values hold until
the next specified frame. The sample schedule above is for the 575-pair
113through bag, not a universal length.

For a fixed-light auto-exposure baseline, use `--ae --light` instead of
`--schedule-yaml`. Light is a commanded fraction in `[0,1]`; these commands
prepare the 10% and 100% baselines without launching SLAM:

```bash
pixi run -e vslamlab clid-stereo prepare "$BAG" --ae --light 0.1
pixi run -e vslamlab clid-stereo prepare "$BAG" --ae --light 1.0
pixi run -e vslamlab clid-stereo run "$BAG" --ae --light 0.1 --slam orbslam3
pixi run -e vslamlab clid-stereo demo "$BAG" --ae --light 1.0 --slam orbslam3 --control-plot
```

This reuses BorealHDR's pinned `Metric_Classical(50)` update: meter the left
camera's **unrectified, synthetic 12-bit Bayer** image after the CLID calibrated
response (before demosaic and display gamma), convert it to grayscale with
OpenCV's Bayer RG conversion, and measure the mean in 12-bit DN. The target is
2048 DN, with a strict ±15 DN deadband and proportional gain 0.002 ms/DN.
Frame `n` uses the exposure chosen from frame `n−1`; the initial exposure is
4 ms (0 EV). The requested next exposure uses the upstream absolute-value
update, then is clamped to 1–16 ms, CLID's −2 to +2 EV range. Exposure is
continuous within that interval, not rounded to discrete stops. Both cameras
use the same chosen EV, but retain their own calibrated light and Bayer
responses. The synthetic feedback is calibrated to the CLID response, not a
physical BorealHDR sensor measurement; controller arithmetic is the same, but
its closed-loop behavior is not claimed identical to BorealHDR.

Each AE variant writes a dense `schedule.yaml`, `ae_trace.csv`, the usual
`relight.csv`, and `relight.json` provenance beside the images. The trace has
applied exposure in ms/EV, left mean/error in 12-bit DN, requested and clamped
next exposure, and clamp status. At low light the target may remain unreachable
even at +2 EV; `max` in the trace makes this explicit. Complete AE variants
are verified and reused. `--ae` cannot be combined with `--schedule-yaml`,
and `--light` is required with `--ae`.

For a smooth per-frame test on that bag, use
[`configs/clid_stereo_sine_113through.yaml`](../configs/clid_stereo_sine_113through.yaml).
It specifies all 575 frames: commanded light follows `0.5 + 0.4 sin(2πn/115 + 0.19)`
and exposure follows `1.25 cos(2πn/115 + 0.47)` EV, for five cycles. The two
controls use distinct phases; each is independently applied to both cameras.

```bash
pixi run -e vslamlab clid-stereo demo "$BAG" \
  --schedule-yaml configs/clid_stereo_sine_113through.yaml --slam orbslam3 \
  --control-plot
```

`--control-plot` is available only with `demo` on Linux with an X display. It
starts before SLAM and shows commanded light in `[0,1]` and exposure in EV on
separate live traces, with the current left-frame marker and both cameras'
effective-power values. Like the Boreal AE plot, its marker follows image-file
opens; it does not prove that native SLAM processed every opened frame. The plot
closes when SLAM exits and does not alter the prepared images or the run.

The extracted `groundtruth.csv` is bag odometry in the body frame; the bag has
no surveyed camera-frame GT transform. Evaluation is therefore an odometry
reference, not an absolute GT accuracy claim. CLID release-2 sequence data
also have no matched relighting-image GT. The published model's ambient-only
recovery can retain active-light leakage, so P=0 appearance must not be judged
from aggregate dark-pixel-dominated metrics alone.
