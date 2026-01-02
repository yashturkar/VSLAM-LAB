# LIGHTNING Dataset - VSLAM-LAB Usage Guide

This guide explains how to use your custom KITTI-format dataset (LIGHTNING) with VSLAM-LAB to run experiments, evaluate trajectories, and generate comparison plots.

## Dataset Overview

The LIGHTNING dataset is stored at:
```
/mnt/share/nas/eph/clid-dataset/davis-lightning-12-2025/Sequences/Processed
```

Available sequences include:
- `labin_p0.0_extract`, `labin_p0.5_extract`, `labin_p1.0_extract`
- `101backdoor_p0.0_extract`, `101backdoor_p0.5_extract`, `101backdoor_p1.0_extract`
- `101out_p0.0_extract`, `101out_p0.5_extract`, `101out_p1.0_extract`
- `113through_p0.0_extract`, `113through_p0.5_extract`, `113through_p1.0_extract`
- `lab2113_p0.0_extract`, `lab2113_p0.5_extract`, `lab2113_p1.0_extract`
- `tunnelin_p0.0_extract`, `tunnelin_p0.5_extract`, `tunnelin_p1.0_extract`
- `tunnelout_p0.0_extract`, `tunnelout_p0.5_extract`, `tunnelout_p1.0_extract`

## Quick Start

### 1. Run a Single Sequence with a Baseline

```bash
pixi run demo <baseline_name> lightning <sequence_name>
```

**Example:**
```bash
pixi run demo mast3rslam lightning labin_p0.0_extract
```

This will:
- Download/prepare the sequence
- Run the SLAM algorithm
- Evaluate the trajectory
- Generate results

### 2. Run Multiple Sequences for Comparison

Edit `configs/config_lightning.yaml` to specify which sequences to compare:

```yaml
lightning:
  - 'labin_p0.0_extract'
  - 'labin_p0.5_extract'
  - 'labin_p1.0_extract'
```

Then run the full pipeline:

```bash
pixi run vslamlab configs/exp_lightning.yaml
```

This will:
- Run experiments on all specified sequences
- Evaluate trajectories (compute ATE/RMSE)
- Generate comparison plots

### 3. Change the Baseline

Edit `configs/exp_lightning.yaml` to change the baseline:

```yaml
exp_lightning:
  Config: config_lightning.yaml
  NumRuns: 1
  Parameters: {verbose: 1}
  Module: mast3rslam  # Change this to: orbslam2, droidslam, dso, etc.
```

Available baselines: `mast3rslam`, `orbslam2`, `droidslam`, `dso`, `dpvo`, `monogs`, `glomap`, etc.

## Detailed Workflow

### Step 1: Configure Your Experiment

**File: `configs/config_lightning.yaml`**
```yaml
lightning:
  - 'labin_p0.0_extract'
  - 'labin_p0.5_extract'
  - 'labin_p1.0_extract'
  # Add more sequences as needed
```

**File: `configs/exp_lightning.yaml`**
```yaml
exp_lightning:
  Config: config_lightning.yaml
  NumRuns: 1                    # Number of runs per sequence
  Parameters: {verbose: 1}     # Baseline-specific parameters
  Module: mast3rslam           # SLAM algorithm to use
```

### Step 2: Run Experiments

```bash
# Full pipeline (run + evaluate + compare)
pixi run vslamlab configs/exp_lightning.yaml

# Or step by step:
pixi run run-exp configs/exp_lightning.yaml          # Run SLAM
pixi run evaluate-exp configs/exp_lightning.yaml     # Compute metrics
pixi run compare-exp configs/exp_lightning.yaml      # Generate plots
```

### Step 3: View Results

#### RMSE/ATE Metrics

Metrics are stored in CSV files:
```
~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/<sequence_name>/vslamlab_evaluation/ate.csv
```

**View metrics:**
```bash
# View metrics for a specific sequence
cat ~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p0.0_extract/vslamlab_evaluation/ate.csv

# Or use Python to analyze
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p0.0_extract/vslamlab_evaluation/ate.csv')
print(df[['traj_name', 'rmse', 'mean', 'median', 'std', 'min', 'max']])
EOF
```

**CSV columns:**
- `traj_name`: Trajectory file name
- `rmse`: Root Mean Square Error (ATE) in meters
- `mean`: Mean error in meters
- `median`: Median error in meters
- `std`: Standard deviation
- `min`: Minimum error
- `max`: Maximum error
- `num_tracked_frames`: Number of poses in trajectory
- `num_evaluated_frames`: Number of poses successfully evaluated

#### Trajectory Plots

Comparison plots are generated at:
```
~/Workspace/VSLAM-LAB-Evaluation/comp_exp_lightning/figures/
```

**Key files:**
- `trajectories.pdf` - 2D trajectory plots (ground truth vs estimated)
- `rmse_boxplot.pdf` - RMSE boxplots for all sequences
- `rmse_radar.pdf` - Radar plot showing relative performance
- `num_frames_boxplot.pdf` - Frame count comparisons

**View plots:**
```bash
# Open in PDF viewer
evince ~/Workspace/VSLAM-LAB-Evaluation/comp_exp_lightning/figures/trajectories.pdf

# Or any PDF viewer
xdg-open ~/Workspace/VSLAM-LAB-Evaluation/comp_exp_lightning/figures/trajectories.pdf
```

#### Detailed Trajectory Files

Aligned trajectories (for custom analysis):
```
~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/<sequence_name>/vslamlab_evaluation/
  - 00000_KeyFrameTrajectory.tum  # Aligned estimated trajectory (TUM format)
  - 00000_gt.tum                   # Aligned ground truth (TUM format)
  - 00000_KeyFrameTrajectory.csv  # Aligned estimated trajectory (CSV format)
  - 00000_gt.csv                   # Aligned ground truth (CSV format)
```

## Comparing Different Parameter Sets (P0.0, P0.5, P1.0, etc.)

To compare sequences with different parameters (e.g., `labin_p0.0_extract` vs `labin_p1.0_extract`):

### Method 1: Single Experiment with Multiple Sequences

Edit `configs/config_lightning.yaml`:
```yaml
lightning:
  - 'labin_p0.0_extract'
  - 'labin_p0.5_extract'
  - 'labin_p1.0_extract'
```

Run:
```bash
pixi run vslamlab configs/exp_lightning.yaml
```

This creates comparison plots showing all sequences side-by-side.

### Method 2: Multiple Experiments (Different Baselines)

Create separate experiment configs:
- `configs/exp_lightning_p0.yaml` - for P0.0 sequences
- `configs/exp_lightning_p1.yaml` - for P1.0 sequences

Then compare them (requires modifying comparison config - see advanced usage).

## Extracting RMSE Numbers Programmatically

### Python Script

```python
import pandas as pd
import os

# Get RMSE for all sequences in an experiment
exp_folder = '~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING'
sequences = ['labin_p0.0_extract', 'labin_p0.5_extract', 'labin_p1.0_extract']

results = []
for seq in sequences:
    ate_csv = os.path.join(exp_folder, seq, 'vslamlab_evaluation', 'ate.csv')
    if os.path.exists(ate_csv):
        df = pd.read_csv(ate_csv)
        results.append({
            'sequence': seq,
            'rmse': df['rmse'].values[0],
            'mean': df['mean'].values[0],
            'median': df['median'].values[0],
            'std': df['std'].values[0],
            'max': df['max'].values[0],
            'min': df['min'].values[0]
        })

# Create summary DataFrame
summary = pd.DataFrame(results)
print(summary.to_string(index=False))

# Save to CSV
summary.to_csv('rmse_summary.csv', index=False)
```

### Command Line

```bash
# Extract RMSE for all sequences
for seq in labin_p0.0_extract labin_p0.5_extract labin_p1.0_extract; do
    echo "=== $seq ==="
    cat ~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/$seq/vslamlab_evaluation/ate.csv | \
        awk -F',' 'NR==2 {print "RMSE:", $2, "m | Mean:", $3, "m | Max:", $7, "m"}'
done
```

## File Structure

After running experiments, your directory structure will be:

```
~/Workspace/VSLAM-LAB-Evaluation/
└── exp_lightning/
    └── LIGHTNING/
        ├── labin_p0.0_extract/
        │   ├── 00000_KeyFrameTrajectory.csv    # Original trajectory
        │   ├── groundtruth.csv                  # Ground truth
        │   └── vslamlab_evaluation/
        │       ├── ate.csv                      # RMSE/ATE metrics
        │       ├── 00000_KeyFrameTrajectory.tum # Aligned trajectory
        │       └── 00000_gt.tum                 # Aligned ground truth
        ├── labin_p0.5_extract/
        │   └── ...
        └── labin_p1.0_extract/
            └── ...

~/Workspace/VSLAM-LAB-Evaluation/
└── comp_exp_lightning/
    └── figures/
        ├── trajectories.pdf      # Trajectory comparison plots
        ├── rmse_boxplot.pdf      # RMSE boxplots
        ├── rmse_radar.pdf        # Radar plot
        └── ...
```

## Common Commands Reference

```bash
# Run single sequence
pixi run demo mast3rslam lightning labin_p0.0_extract

# Run full experiment (all sequences in config)
pixi run vslamlab configs/exp_lightning.yaml

# Re-evaluate trajectories (if you modified evaluation code)
pixi run evaluate-exp configs/exp_lightning.yaml --overwrite

# Regenerate comparison plots
pixi run compare-exp configs/exp_lightning.yaml

# Check experiment status
pixi run check-experiment-state configs/exp_lightning.yaml

# Download sequences (if needed)
pixi run download-sequence lightning labin_p0.0_extract
```

## Troubleshooting

### Empty Plots

If trajectory plots are empty:
1. Ensure evaluation completed: `pixi run evaluate-exp configs/exp_lightning.yaml --overwrite`
2. Check that `.tum` files exist in evaluation folders
3. Verify `ate.csv` files have valid data

### Missing Trajectory Files

If trajectory files are missing:
1. Check experiment log: `~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/vslamlab_exp_log.csv`
2. Verify baseline ran successfully (check `SUCCESS` column)
3. Re-run the experiment if needed

### Evaluation Fails

If evaluation fails:
1. Check that `groundtruth.csv` exists for the sequence
2. Verify trajectory CSV format (should have columns: ts, tx, ty, tz, qx, qy, qz, qw)
3. Check EVO is installed: `pixi run -e vslamlab evo_ape --help`

## Advanced: Custom Analysis

### Extract Trajectory Data

```python
import pandas as pd

# Read aligned trajectories
traj = pd.read_csv('~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p0.0_extract/vslamlab_evaluation/00000_KeyFrameTrajectory.csv')
gt = pd.read_csv('~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p0.0_extract/vslamlab_evaluation/00000_gt.csv')

# Compute per-frame errors
errors = ((traj[['tx', 'ty', 'tz']] - gt[['tx', 'ty', 'tz']]) ** 2).sum(axis=1) ** 0.5
print(f"Per-frame errors:\n{errors.describe()}")
```

### Compare Multiple Baselines

Create separate experiment configs for each baseline, then manually compare the `ate.csv` files or create a custom comparison script.

## Configuration Files

### `Datasets/dataset_lightning.yaml`
- Dataset configuration
- Lists all available sequences
- Defines dataset parameters (RGB Hz, modes, etc.)

### `configs/config_lightning.yaml`
- Experiment sequence selection
- Specify which sequences to run

### `configs/exp_lightning.yaml`
- Experiment configuration
- Baseline selection
- Parameters

## Quick Reference: Common Workflows

### Compare P0.0 vs P1.0 for labin sequences

1. Edit `configs/config_lightning.yaml`:
```yaml
lightning:
  - 'labin_p0.0_extract'
  - 'labin_p1.0_extract'
```

2. Run full pipeline:
```bash
pixi run vslamlab configs/exp_lightning.yaml
```

3. View results:
```bash
# RMSE numbers
cat ~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p0.0_extract/vslamlab_evaluation/ate.csv
cat ~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING/labin_p1.0_extract/vslamlab_evaluation/ate.csv

# Plots
xdg-open ~/Workspace/VSLAM-LAB-Evaluation/comp_exp_lightning/figures/trajectories.pdf
```

### Get RMSE Summary Table

```bash
python3 << 'EOF'
import pandas as pd
import os

sequences = ['labin_p0.0_extract', 'labin_p0.5_extract', 'labin_p1.0_extract']
base_path = os.path.expanduser('~/Workspace/VSLAM-LAB-Evaluation/exp_lightning/LIGHTNING')

results = []
for seq in sequences:
    csv_path = os.path.join(base_path, seq, 'vslamlab_evaluation', 'ate.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        results.append({
            'Sequence': seq,
            'RMSE (m)': f"{df['rmse'].values[0]:.4f}",
            'Mean (m)': f"{df['mean'].values[0]:.4f}",
            'Max (m)': f"{df['max'].values[0]:.4f}",
            'Frames': int(df['num_tracked_frames'].values[0])
        })

summary = pd.DataFrame(results)
print("\n=== RMSE Summary ===")
print(summary.to_string(index=False))
EOF
```

### Regenerate Plots Only (After Re-evaluation)

```bash
# If you re-ran evaluation and want fresh plots
pixi run compare-exp configs/exp_lightning.yaml
```

## Notes

- Trajectories are automatically aligned using Horn's method before evaluation
- RMSE/ATE is computed using EVO (evo_ape)
- Plots use PCA projection for 2D visualization
- All times are in seconds
- All positions are in meters
- Quaternions are in [x, y, z, w] format
- Evaluation results are stored in `~/Workspace/VSLAM-LAB-Evaluation/`

