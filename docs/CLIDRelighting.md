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

## Current stereo blocker

Only camera-0 components exist. `--mode stereo` deliberately fails instead of
mixing synthesized left frames with recorded right frames. Stereo support is
complete only after synchronized `clid_components_1` files and their manifest
contract are available; the same schedule must then be applied to both cameras
while preserving the calibrated baseline.
