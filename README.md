<p align="center">
<div align="center">
    <img src="docs/header.png" width="500"/>
</div>

<h3 align="center"> A Comprehensive Framework for Visual SLAM Baselines and Datasets</h3>

<p align="center">
    <a href="https://scholar.google.com/citations?user=SDtnGogAAAAJ&hl=en"><strong>Alejandro Fontan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=eq46ylAAAAAJ&hl=en"><strong>Tobias Fischer</strong></a>
    ·
    <a href="https://nmarticorena.github.io/"><strong>Nicolas Marticorena</strong></a>
</p>

 <p align="center">
     <a href="https://www.linkedin.com/in/somayeh-hussaini/?originalSubdomain=au"><strong>Somayeh Hussaini</strong></a>
    ·
     <a href="https://github.com/TedVanderfeen"><strong>Ted Vanderfeen </strong></a>
    ·
     <a href="https://scholar.google.com/citations?hl=es&user=s3eIy0YAAAAJ"><strong>Beverley Gorry </strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en"><strong>Javier Civera</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en"><strong>Michael Milford</strong></a>
</p>

<br/>
<div align="left">

![Maintained? yes](https://img.shields.io/badge/Maintained%3F-yes-success) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md) ![Last commit](https://img.shields.io/github/last-commit/VSLAM-LAB/VSLAM-LAB) [![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/VSLAM-LAB/VSLAM-LAB/blob/main/LICENSE.txt) [![arXiv](https://img.shields.io/badge/arXiv-2410.23690-B31B1B.svg)](https://arxiv.org/abs/2504.04457)

</div>

## Introduction

**VSLAM-LAB** is designed to simplify the development, evaluation, and application of Visual SLAM (VSLAM) systems.
This framework enables users to compile and configure VSLAM systems, download and process datasets, and design, run, and
evaluate experiments — **all from a single command line**!

**Why Use VSLAM-LAB?**
- **Unified Framework:** Streamlines the management of VSLAM systems and datasets.
- **Ease of Use:** Run experiments with minimal configuration and single command executions.
- **Broad Compatibility:** Supports a wide range of VSLAM systems and datasets.
- **Reproducible Results:** Standardized methods for evaluating and analyzing results.

<!--
<div align="center">
    <img src="docs/diagram.svg" width="960"/>
</div>
-->

## Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
*After installation, restart your terminal or source your shell for the changes to take effect*. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

*If you already have pixi remember to update:* `pixi self-update`

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/VSLAM-LAB/VSLAM-LAB.git && cd VSLAM-LAB
```

## Quick Demo
You can now execute any baseline on any sequence from any dataset within VSLAM-LAB using the following command:
```bash
pixi run demo <baseline> <dataset> <sequence> <mode>
```
For a full list of available systems and datasets, see the [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).
Example commands:
```bash
pixi run demo mast3rslam eth table_3 mono
pixi run demo droidslam rgbdtum rgbd_dataset_freiburg1_xyz rgbd
pixi run demo orbslam2 kitti 04 stereo
pixi run demo pycuvslam euroc MH_01_easy stereo-vi
```
*To change the paths where VSLAM-LAB-Benchmark or/and VSLAM-LAB-Evaluation data are stored (for example, to /media/${USER}/data), use the following commands:*
```bash
pixi run set-benchmark-path /media/${USER}/data
pixi run set-evaluation-path /media/${USER}/data
```

## Configure your own experiments
With **VSLAM-LAB**, you can easily design and configure experiments using a YAML file and run them with a single command.
To **run** the experiment demo, execute the following command:
```bash
pixi run vslamlab configs/exp_vslamlab.yaml (--overwrite)
```

Experiments in **VSLAM-LAB** are sequences of entries in a YAML file (see example **~/VSLAM-LAB/configs/exp_vslamlab.yaml**):
```yaml
exp_vslamlab:
  Config: config_vslamlab.yaml  # YAML file containing the sequences to be run
  NumRuns: 1                    # Maximum number of executions per sequence
  Parameters: {verbose: 1}      # Vector with parameters that will be input to the baseline executable
  Module: droidslam             # droidslam/monogs/orbslam2/mast3rslam/dpvo/...
```
**Config** files are YAML files containing the list of sequences to be executed in the experiment (see example **~/VSLAM-LAB/configs/config_vslamlab.yaml**):
```yaml
rgbdtum:
  - 'rgbd_dataset_freiburg1_xyz'
hamlyn:
  - 'rectified01'
7scenes:
  - 'chess_seq-01'
eth:
  - 'table_3'
euroc:
  - 'MH_01_easy'
monotum:
  - 'sequence_01'
```
For a full list of available VSLAM systems and datasets, refer to the section [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).

## VSLAM-LAB Pipeline Commands
In addition to running the full automated pipeline, **VSLAM-LAB** provides modular commands to interact directly with datasets and baselines. For a comprehensive list of all available commands consult [Wiki: Comand‐line Interface](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Comand%E2%80%90line-Interface)

```bash
pixi run install-baseline <baseline>                     # Example: pixi run install-baseline droidslam
pixi run download-sequence <dataset> <sequence>          # Example: pixi run download-sequence eth table_3
pixi run run-exp <exp_yaml>                              # Example: pixi run run-exp configs/exp_vslamlab.yaml
pixi run evaluate-exp <exp_yaml>                         # Example: pixi run evaluate-exp configs/exp_vslamlab.yaml
pixi run compare-exp <exp_yaml>                          # Example: pixi run compare-exp configs/exp_vslamlab.yaml
pixi run eval-metrics <exp_yaml>                         # Run, evaluate, and write metrics.json
pixi run eval-metrics-single <config_yaml>               # Headless custom-sequence evaluation
pixi run demo-single <config_yaml>                       # GUI custom-sequence demo
pixi run fastlio-reference <config_yaml>                 # Generate/reuse a FAST-LIO reference
pixi run demo-fastlio <config_yaml>                      # FAST-LIO playback with RViz
pixi run lightning-fastlio <processed_sequence_root>     # Stage, extract, run, and evaluate
```

### Research evaluation workflows

`eval-metrics` runs the standard experiment pipeline and writes a machine-readable
`metrics.json` under each sequence's `vslamlab_evaluation` directory. For a local
sequence, copy `configs/single_lightning.yaml`, set `base_path` and `name`, then run:

```bash
pixi run eval-metrics-single configs/single_lightning.yaml
pixi run demo-single configs/single_lightning.yaml
```

The preferred portable dataset selector is `dataset: lightning`. A repository-relative
or absolute path to a custom Python dataset module is also accepted. Relative
`output_dir` values are created below `base_path`. The output `metrics.json` contains
ATE/RMSE values, trajectory lengths, symmetric length coverage (`length_ratio`), and
coverage-weighted RMSE.

Single-sequence configs may also include a `DATASET.parameters` mapping to override
baseline defaults. This is useful for fast classical smoke tests; for example,
`configs/single_lightning_colmap_p0.yaml` limits COLMAP to 100 evenly sampled images.
Ready-to-run LIGHTNING examples are also provided for monocular ORB-SLAM2 and
ORB-SLAM3. Use `eval-metrics-single` for headless servers; `demo-single` requires an X
display for baseline viewers such as Pangolin.

LIGHTNING stereo sequences are detected from `image_0` and `image_1`. Put the calibrated
OpenCV left-to-right transform in the sequence calibration as flattened `Stereo.R` and
`Stereo.T` values; the adapter converts it to the VSLAM-LAB camera-pose convention.
The measured rig values are recorded in `configs/calibration_lightning_stereo.yaml`.
Examples for `101backdoor_p0.0_extract` are provided for ORB-SLAM2 and ORB-SLAM3 stereo.

Synchronized ROS 2 MCAP recordings can be converted directly when they contain the
default `image_preview`, `camera_info`, and `/odometry` topics:

```bash
pixi exec --spec uv uv run Utilities/extract_lightning_mcap.py \
  --bag /path/to/research-bag_0.mcap \
  --output /path/to/vslamlab/sequence_name
```

Use the extractor's topic arguments when a recording uses different names. It rectifies
both image streams from `camera_info`, writes synchronized stereo metadata, and exports
`/odometry` as ground truth.

#### FAST-LIO reference evaluation

On Ubuntu 22.04, install native ROS 2 Humble and the pinned SPARK FAST-LIO workspace:

```bash
Utilities/setup_fastlio_humble.sh
```

The setup installs ROS 2 Desktop, MCAP support and build tools, then builds
`MIT-SPARK/spark-fast-lio` in `~/humble_ws`. It does not edit shell startup files;
the VSLAM-LAB runner sources the ROS environments explicitly.

Enable the reference in a single-sequence config whose extracted sequence contains
`extraction_metadata.json`:

```yaml
EVALUATION:
  max_time_difference_s: 0.02
  fast_lio:
    enabled: true
    workspace: /home/yashturkar/humble_ws
```

Generate or inspect the LiDAR trajectory independently:

```bash
pixi run -e vslamlab fastlio-reference configs/single_slam_test_2_seq001_orbslam2_stereo.yaml
pixi run -e vslamlab fastlio-reference configs/single_slam_test_2_seq001_orbslam2_stereo.yaml --force
pixi run -e vslamlab demo-fastlio configs/single_slam_test_2_seq001_orbslam2_stereo.yaml
```

`eval-metrics-single` automatically generates or reuses the same sequence-level cache.
Its schema-v2 `metrics.json` reports VSLAM vs robot odometry, FAST-LIO vs robot
odometry, and VSLAM vs FAST-LIO through EVO. Metric stereo and LiDAR trajectories use
rigid SE(3) alignment without scale correction; monocular VSLAM uses Sim(3). The bags
do not contain measured camera/body and Ouster/body mount transforms, so cross-sensor
translation and especially rotation results are explicitly marked as approximate until
those transforms are supplied. Robot-reference associations use the configured 20 ms
window. For direct VSLAM-to-FAST-LIO evaluation, the approximately 1 Hz corrected
FAST-LIO `/path` is linearly interpolated in position and quaternion-Slerped at VSLAM
timestamps before EVO alignment and metric calculation.

#### One-command resumable pipeline

A processed CLID sequence root can be staged and evaluated end to end with one
foreground command:

```bash
pixi run -e vslamlab lightning-fastlio \
  /mnt/share/nas/eph/clid-v2-sequences/session/sequence
```

The command validates the recording and required topics, checks disk space and
runtimes, stages `research-bag` under `/mnt/share/local/eph/VSLAM`, extracts the
stereo dataset, runs ORB-SLAM2 and FAST-LIO, and writes pairwise metrics. Source paths
are opened read-only, and outputs beneath `/mnt/share/nas` or inside the source tree
are refused.

Every stage records fingerprints and integrity checks in `pipeline.json`. Repeating
the command prints `SKIP (verified)` for valid work and resumes the first missing,
partial, or stale stage. Progress and failures are appended to `pipeline.log`.

```bash
pixi run -e vslamlab lightning-fastlio-check <processed_sequence_root>
pixi run -e vslamlab lightning-fastlio-status <processed_sequence_root>
```

Advanced recovery and path overrides use the underlying CLI:

```bash
pixi run -e vslamlab python Utilities/lightning_fastlio_pipeline.py run \
  <processed_sequence_root> --force-from fastlio

pixi run -e vslamlab python Utilities/lightning_fastlio_pipeline.py run \
  <processed_sequence_root> --local-root /mnt/share/local/eph/VSLAM
```

`--force-from` accepts `stage`, `extract`, `config`, `orbslam2`, `fastlio`, or
`metrics` and reruns that stage and every downstream stage. Version one requires one
indexed MCAP inside `research-bag`; split processed research bags fail preflight.

#### Streamlit SLAM viewer

Browse every processed capture below `/mnt/share/nas/eph/clid-v2-sequences`, launch
timestamped headless runs, and inspect aligned trajectories and EVO APE/RPE metrics:

```bash
pixi run -e vslamlab slam-viewer
```

Open `http://localhost:8501`. The server binds to all interfaces, so it can also be
opened through the host's Tailscale address. The NAS is treated as read-only. MCAPs
and extracted stereo data are cached below `/mnt/share/local/eph/VSLAM/runs`; each
button-triggered result is saved separately below
`/mnt/share/local/eph/VSLAM/web_results/<sequence>/<timestamp>_<baseline>`.

The viewer offers installed stereo-capable baselines from the VSLAM-LAB registry.
FAST-LIO is optional per run. Metrics and the downloadable trajectory report are
generated by the existing EVO evaluation pipeline.

### BorealHDR stereo runs and exposure schedules

BorealHDR uses the local recordings at `/mnt/share/local/eph/BorealHDR` and
calibration from `/home/yashturkar/Workspace/TFR24_BorealHDR/BorealHDR`.
Stereo is the default. Select a sequence and SLAM method from the repo root:

```bash
# List recordings
pixi run -e vslamlab borealhdr list

# Headless stereo with the default recorded 4 ms exposure
pixi run -e vslamlab borealhdr run backpack_2023-04-20-09-29-14 --slam orbslam2 --mode stereo

# Prepare images using the supplied varying-exposure schedule
pixi run -e vslamlab borealhdr prepare backpack_2023-04-20-09-29-14 \
  --exposure-yaml configs/borealhdr_exposure_demo.yaml

# Headless stereo with that schedule (preparation is automatic/reused)
pixi run -e vslamlab borealhdr run backpack_2023-04-20-09-29-14 \
  --slam orbslam2 --mode stereo --exposure-yaml configs/borealhdr_exposure_demo.yaml

# Stereo GUI demo with that schedule; requires an X display
pixi run -e vslamlab borealhdr demo backpack_2023-04-20-09-29-14 \
  --slam orbslam2 --mode stereo --exposure-yaml configs/borealhdr_exposure_demo.yaml
```

Replace `orbslam2` with another registered stereo-capable method, such as
`orbslam3`. To use your own schedule, replace the `--exposure-yaml` path.

An exposure YAML is a plain **zero-based frame number: exposure in milliseconds**
mapping. Each entry applies to **both left and right cameras**, starting at that
frame and holding until the next entry:

```yaml
0: 4.0     # Frames 0–59: 4 ms
60: 8.0    # Frames 60–119: 8 ms
120: 16.0  # Frame 120 onward: 16 ms
```

See the [exposure YAML specification](docs/BorealHDR.md#exposure-yaml-specification)
for validation rules, frame indexing, and emulation behavior, and the
[demo YAML](configs/borealhdr_exposure_demo.yaml) for a complete example.
Preparation prints exposure changes; cached inputs print their schedule before
SLAM starts. These are preparation/input reports, not live SLAM playback updates.

Outputs live under `/mnt/share/local/eph/VSLAM/borealhdr`: prepared inputs are
cached, and each run gets timestamped results with its trajectory, PDF, and logs.
Scheduled runs also save their resolved exposure YAML and per-frame exposure CSV.
The local sample has no supplied pose ground truth, so reference RMSE is not
computed. See [BorealHDR details](docs/BorealHDR.md) for path overrides and limitations.

### Shared runtime storage

Keep Pixi environments, baseline checkouts, checkpoints, and model caches off the
source disk with:

```bash
python Utilities/setup_shared_storage.py --root /mnt/share/local/eph/VSLAM
```

The setup is idempotent. It links `.pixi`, configures project caches, and causes future
baseline installs to relocate ignored third-party checkouts beneath the shared root.
Existing baseline checkouts are copied and verified first, with timestamped local
backups retained for manual removal after validation. Benchmark and evaluation paths
remain controlled by `set-benchmark-path` and `set-evaluation-path`.

RTX 50-series and RTX PRO Blackwell GPUs (`sm_120`) require a PyTorch build made with
CUDA 12.8 or newer. After installing the MASt3R-SLAM environment, apply its compatible
PyTorch wheel and rebuild the MASt3R matching kernels once with:

```bash
pixi run -e mast3rslam setup-blackwell
```

DROID-SLAM also ships custom CUDA extensions. Rebuild its DROID, lietorch, and
torch-scatter kernels once on Blackwell with:

```bash
pixi run -e droidslam setup-blackwell
```

## Add a new VSLAM Dataset

Expand the evaluation suite by integrating custom datasets. Follow the instructions in [Wiki: Integrate a new VSLAM Dataset](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Integrate-a-new-VSLAM-Dataset).

## Add a new VSLAM Baseline

Incorporate new algorithms into the framework. Follow the guide in [Wiki: Integrate a new VSLAM Baseline](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Integrate-a-new-VSLAM-Baseline). Benchmark your method against state-of-the-art baselines across all supported datasets.

For a reference implementation, see the VGGT-SLAM integration in commit [259f7ae](https://github.com/VSLAM-LAB/VSLAM-LAB/commit/259f7aec88d4576880f3cc98983660f508af13a9).


## License
**VSLAM-LAB** is released under a **LICENSE.txt**. For a list of code dependencies which are not property of the authors of **VSLAM-LAB**, please check **docs/Dependencies.md**.


## Citation
If you're using **VSLAM-LAB** in your research, please cite:
```bibtex
@INPROCEEDINGS{fontan2025vslam,
  author={Fontan, Alejandro and Fischer, Tobias and Civera, Javier and Milford, Michael},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={VSLAM-LAB: A Comprehensive Framework for Visual SLAM Methods and Datasets},
  year={2025},
  volume={},
  number={},
  pages={9013-9020},
  keywords={Visualization;Simultaneous localization and mapping;Robot vision systems;Standardization;Benchmark testing;Programming;Reproducibility of results;Intelligent robots;Faces;Software development management},
  doi={10.1109/IROS60139.2025.11247218}
}
```

<!-- ## Acknowledgements

To [awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets) -->

# VSLAM-LAB Supported Baselines and Datasets
| Baselines                                                                   | System |      Modes       |                                   License                                   |    Label     |  Conda Pkg     |  Camera Models     |
|:----------------------------------------------------------------------------|:------:|:------:|:----------------:|:---------------------------------------------------------------------------:|:------------:|:------------:|
| [**VGGT-SLAM**](https://github.com/MIT-SPARK/VGGT-SLAM) |  VSLAM   |  `mono`  |  [BSD-2](https://github.com/MIT-SPARK/VGGT-SLAM/blob/main/LICENSE)  |   `vggtslam`   | ✅ | `pinhole` |
| [**MASt3R-SLAM**](https://github.com/rmurai0610/MASt3R-SLAM)                | VSLAM  |       `mono`       |    [CC BY-NC-SA 4.0](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)    | `mast3rslam`  | ✅ | `radtan5` `unknown` |
| [**DPVO**](https://github.com/princeton-vl/DPVO)                            | VSLAM  |       `mono`       |    [License](https://github.com/princeton-vl/DPVO/blob/main/LICENSE)    | `dpvo`  | ✅ | `radtan5` |
| [**DROID-SLAM**](https://github.com/princeton-vl/DROID-SLAM)                | VSLAM  |`mono` `rgbd` `stereo`|    [BSD-3](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE)    | `droidslam`  | ✅ | `radtan5` |
| [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2)               | VSLAM  |`mono` `rgbd` `stereo`| [GPLv3](https://github.com/raulmur/ORB_SLAM2/blob/master/LICENSE.txt)|  `orbslam2`  | ✅ | `radtan5`  |
| [**MonoGS**](https://github.com/muskie82/MonoGS)                            | VSLAM  | `mono` `rgbd` |     [License](https://github.com/muskie82/MonoGS?tab=License-1-ov-file)     |   `monogs`   | ✅ | `radtan5` |
| [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM) | VSLAM  | `mono` | [GPLv3](https://github.com/alejandrofontan/VSLAM-LAB/blob/main/LICENSE.txt) | `anyfeature` | ✅ |  `radtan5` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**PyCuVSLAM**](https://github.com/VSLAM-LAB/PyCuVSLAM/tree/main) | VSLAM  |`mono` `rgbd` `stereo(-vi)` | [NVIDIA](https://github.com/VSLAM-LAB/PyCuVSLAM/blob/main/LICENSE) |  `pycuvslam`  | ➖ | `radtan5` `equid4` |
| [**ORB-SLAM3**](https://github.com/UZ-SLAMLab/ORB_SLAM3)               | VSLAM  | `mono(-vi)` `rgbd(-vi)` `stereo(-vi)` |    [GPLv3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/LICENSE)    |  `orbslam3`  | ✅ | `radtan5` `equid4`|
| [**OKVIS2**](https://github.com/ethz-mrl/okvis2)               | VSLAM  | `mono-vi` |    [BSD-3](https://github.com/ethz-mrl/okvis2/blob/main/LICENSE)    |  `okvis2`  | ✅ | `radtan5` `equid4` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**COLMAP**](https://colmap.github.io/)                                     |  SfM   |       `mono`       |                [BSD](https://colmap.github.io/license.html)                 |   `colmap`   | ✅ | `radtan5` `equid4` `unknown` |
| [**VGGT**](https://vgg-t.github.io/) |  SfM   |  `mono`  |  [VGGT](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)  |   `vggt`   | ➖ | `pinhole` |

| Datasets                                                                                                                        | Features |   Label    |      Modes       |  Camera Models     |
|:--------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|:----------:|
| [**ETH3D SLAM Benchmarks**](https://www.eth3d.net/slam_datasets)                                                                |  📸🏠🤳 |   `eth`    |`mono` `rgbd`| `pinhole` |
| [**RGB-D SLAM Dataset and Benchmark**](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)                                       |  📸🏠🤳 |  `rgbdtum`  |`mono` `rgbd`| `radtan5` |
| [**The KITTI Vision Benchmark Suite**](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)                                 |  📸🏞️🚗 |   `kitti`   |`mono` `stereo` | `pinhole` |
| [**The EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)                       |  📸🏞️🚁 |   `euroc`   | `mono(-vi)` `stereo(-vi)` | `radtan4` |
| [**The Replica Dataset**](https://github.com/facebookresearch/Replica-Dataset) - [**iMAP**](https://edgarsucar.github.io/iMAP/) |  💻🏠🤳 |  `replica`  | `mono` `rgbd`  | `pinhole` |
| [**TartanAir: A Dataset to Push the Limits of Visual SLAM**](https://theairlab.org/tartanair-dataset/)                          |  💻🏞️🤳 | `tartanair` | `mono`  | `pinhole` |
| [**ICL-NUIM RGB-D Benchmark Dataset**](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)                                    |  💻🏠🤳 |   `nuim`    | `mono` `rgbd`  | `pinhole` |
| [**RGB-D Dataset 7-Scenes**](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)                          |  📸🏠🤳 |   `7scenes` | `mono` `rgbd`  | `pinhole` |
| [**OpenLORIS-Scene Dataset**](https://lifelong-robotic-vision.github.io/dataset/scene.html) |  📸🏠🤳 |   `openloris-d400/t265` | `mono(-vi)` `rgbd(-vi)` `stereo(-vi)`  | `pinhole` `equid4` |
| [**Monado SLAM Dataset - Valve Index**](https://huggingface.co/datasets/collabora/monado-slam-datasets)                         |  📸🏠🥽 | `msd` | `mono(-vi)` `stereo(-vi)` | `equid4` |
| [**ROVER: A Multiseason Dataset for Visual SLAM**](https://iis-esslingen.github.io/rover/)   | 📸🏞️🚗 | `rover-picam/d435i/t265` |`mono(-vi)` `rgbd` `stereo(-vi)` | `radtan5` `equid4` |
| [**The UT Campus Object Dataset**](https://amrl.cs.utexas.edu/coda/) | 📸🏞️🤖 |  `ut-coda`  |`mono` `stereo`| `radtan5` |
| [**Sesoko campaign**](https://www.southampton.ac.uk/smmi/news/2017/06/20-southampton-tokyo-collaboration.page) | 📸🏞️🌊 |    `sesoko`    |`mono` | `pinhole` |
| [**The MADMAX data set for visual-inertial rover navigation on Mars**](https://datasets.arches-projekt.de/morocco2018/) |  📸🏞️🤳 | `madmax` | `mono(-vi)` `stereo(-vi)`| `pinhole` |
| [**Soneva Corals**](https://huggingface.co/datasets/wildflow/soneva-corals) | 📸🏞️🌊🤳 |    `soneva`    |`mono` | `pinhole` |
| [**Sweet Corals**](https://huggingface.co/datasets/wildflow/sweet-corals) | 📸🏞️🌊🤳 |    `sweetcorals`    |`mono` | `pinhole` `unknown` |
| [**Eiffel Tower: A Deep-Sea Underwater Dataset for Long-Term Visual Localization**](https://www.seanoe.org/data/00810/92226/) | 📸🏞️🌊🤖 |    `eiffel-tower`    |`mono` | `radtan4` |

| Tools                                                                                                                        | Features |   Label    |      Modes       |  Camera Models     |
|:--------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|:----------:|
| [**Stray Scanner App**](https://github.com/strayrobots/scanner) |  📸🤳 | `strayscanner` | `mono` `rgbd` | `pinhole` |
<!-- | [**Ariel**](https://huggingface.co/datasets/ntnu-arl/underwater-datasets) | 📸🏞️🌊 |    `ariel`    |`mono(-vi)` `stereo(-vi)`  | `equid4` | -->
<!-- | [**HILTI Challenge Dataset 2022**](https://hilti-challenge.com/dataset-2022) | 📸🏠🏞️🤳 |    `hilti2022`    |`mono(-vi)` `stereo(-vi)`  | `equid4` | -->
<!-- | [**HILTI Challenge Dataset 2026**](https://github.com/Hilti-Research/hilti-trimble-slam-challenge-2026) | 📸🏠🏞️🤳 |    `hilti2026`    |`mono(-vi)` | `equid4` | -->
<!-- | [**The Drunkard's Dataset**](https://davidrecasens.github.io/TheDrunkard%27sOdometry/#download-dataset)                                    |  💻🏠🤳 |   `drunkards`    | `mono` `rgbd`  | `pinhole` | -->
<!-- | [**Underwater caves sonar and vision data set**](https://cirs.udg.edu/caves-dataset/)  |  📸🏞️🌊 |   `caves`  | `mono` | `pinhole` | -->
<!-- | [**Hamlyn Rectified Dataset**](https://davidrecasens.github.io/EndoDepthAndMotion/) |   📸🫀🤳 |  `hamlyn`   | `mono` `rgbd` | `pinhole` | -->
<!-- | [**The TUM VI Benchmark for Evaluating Visual-Inertial Odometry**](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) |  📸🏠🤳 | `vitum` | `mono(-vi)` `stereo(-vi)` | `equid4` | -->
<!-- | [**ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes**](https://scannetpp.mlsg.cit.tum.de/scannetpp/) |  📸🏠🤳 | `scannetplusplus` | `mono`| `pinhole` | -->
<!-- [**Monocular Visual Odometry Dataset**](https://cvg.cit.tum.de/data/datasets/mono-dataset) | 📸🏠🤳 |  `monotum`  | `mono` | `pinhole` | -->

Real / Synthetic : 📸 / 💻

Indoor / Outdoor / Underwater / Intracorporeal : 🏠 / 🏞️ /  🌊 / 🫀

Handheld / Headmounted / Vehicle / UAV  / Robot : 🤳 / 🥽 / 🚗 / 🚁 / 🤖

## VSLAM-LAB  Roadmap
### Baselines
- [ ] Extend `orbslam3` and `orbslam3-dev` to `rgbd-vi`
- [ ] Extend `okvis2` and `okvis2-dev` to `rgbd-vi` and `stereo-vi`

### Datasets
- [ ] Implement `monotum`

<!--
## VSLAM-LAB v1.0 Roadmap

### Core
- [ ] Build system set up (CMake + options for CUDA/CPU)
- [ ] Docker dev image (CUDA + ROS optional)
- [ ] Pre-commit hooks (clang-format, clang-tidy, black/isort if Python)
- [ ] Licensing & citation (LICENSE + CITATION.cff + BibTeX snippet)
- [ ] Example dataset download script (`scripts/get_data.sh`)

### Datasets
- [ ] KITTI extension to `stereo`
- [ ] ROVER extension to `stereo`, `mono-vi`, `stereo-vi`
- [ ] TartanAir extension to `stereo`
- [ ] EuRoC extension to `stereo-vi`
- [ ] monotum re-implement `mono`
- [ ] 7scenes re-implement `mono`, `rgbd`
- [ ] drunkards re-implement `mono`, `rgbd`
- [ ] hamlyn re-implement mono `mono`
- [ ] caves re-implement `mono`
- [ ] hilti2022 re-implement `mono`
- [ ] scannetplusplus re-implement `mono`
- [ ] ariel re-implement `mono`
- [ ] lamar implement `mono`
- [ ] squidle implement `mono`
- [ ] openloris re-implement `mono`
- [ ] madmax implement `mono`, `rgbd`, `stereo`, `mono-vi`, `stereo-vi`
- [ ] sweetcorals implement `mono`
- [ ] reefslam implement `mono`
- [ ] ...

### Baselines
- [ ] AnyFeature VSLAM implement `mono`, `rgbd`, `stereo`
- [ ] DSO VSLAM implement `mono`
- [ ] MonoGS re-implement `mono`, `rgbd`
- [ ] VGGT implement SfM
- [ ] ORBSLAM3 implement `mono`, `rgbd`, `stereo`, `stereo-vi`, `rgbd-vi`
- [ ] OKVIS2 implement `mono`, `stereo-vi`
- [ ] pyCuVSLAM implement `mono`, `rgbd`, `stereo`, `mono-vi`, `stereo-vi`

### Metrics
- [ ] Include RPE
- [ ] Link metrics with modes

### Tooling
- [ ] Ablation tools
- [ ] ROS

### Docs
- [ ] README quickstart (build, run, datasets)
- [ ] Config reference (YAML/TOML)
- [ ] Architecture diagram
- [ ] Contributing guide

### Demos
- [ ] Example video/gif of live run

### Project Management
- [ ] Define statuses: Backlog → In Progress → Review → Done
- [ ] Convert key items above to sub-issues
-->
