# EWMVC & Fixed-Epsilon DBSCAN Cone-Mapping

Modular pipeline for Formula Student / autonomous racing cone map building. Starts with a fixed-epsilon DBSCAN baseline and extends to Entropy-Weighted Multi-View Clustering (EWMVC).

---

## 1. Project Purpose
Autonomous racing vehicles perceive track boundaries (blue, yellow, and orange cones) using perception sensors. Building an accurate, noise-free global landmark map from high-frequency, noisy perception observations is critical for path planning and vehicle control. 

This repository provides:
1. A **fixed-epsilon DBSCAN cone mapping baseline**.
2. A **24-configuration DBSCAN parameter sensitivity experiment**.
3. A **detailed failure analysis** identifying the structural limitations of Euclidean DBSCAN.

---

## 2. Dataset Schema

The pipeline consumes two raw CSV log files:

### **Perception Log (`data/raw/perception_log.csv`)**:
- `timestamp` (float): Perception observation timestamp in seconds.
- `cone_type` (str): Detected cone color/class (`left`, `right`, `unknown`).
- `rel_x_sensor` (float): Relative X distance from sensor (m).
- `rel_y_sensor` (float): Relative Y distance from sensor (m).
- `confidence` (float): Detection confidence score $[0.0, 1.0]$.
- `label` (str): Frame-level observation verification label (`real` vs `ghost`).

### **Telemetry Log (`data/raw/telemetry_log.csv`)**:
- `timestamp` (float): Telemetry frame timestamp in seconds ($10\text{ Hz}$).
- `x` (float): Vehicle global X position (m).
- `y` (float): Vehicle global Y position (m).
- `yaw_rad` (float): Vehicle heading orientation (radians).
- `speed_mps` (float): Vehicle speed (m/s).
- `steering_rad` (float): Steering angle (radians).
- `yaw_rate_rps` (float): Yaw angular velocity (rad/s).

---

## 3. Pipeline Architecture

```
                       [Raw Perception CSV]         [Raw Telemetry CSV]
                                │                            │
                                └─────────────┬──────────────┘
                                              ▼
                                    (1. Data Loading & Schema Check)
                                              │
                                              ▼
                                    (2. Telemetry Interpolation)
                                              │
                                              ▼
                                    (3. Sensor -> Vehicle Rotation)
                                              │
                                              ▼
                                    (4. Vehicle -> Global SE(2) Kinematics)
                                              │
                                              ▼
                                    (5. Fixed-Epsilon DBSCAN Clustering)
                                              │
                                              ▼
                                    (6. Weighted Centroid Cone Map)
                                              │
                                              ▼
                                    (7. Observation Evaluation & Export)
```

---

## 4. Baseline Configuration

The default baseline configuration is defined in [`configs/config.yaml`](file:///e:/ABAGA/ewmvc_dbscan/configs/config.yaml):

```yaml
data:
  perception_file: data/raw/perception_log.csv
  telemetry_file: data/raw/telemetry_log.csv

sensor:
  rotation_deg: 180.0

dbscan:
  eps_m: 1.0
  min_samples: 4

preprocessing:
  min_confidence: 0.0

output:
  directory: outputs
```

---

## 5. How to Run

### **A. Run Baseline Pipeline ($\epsilon = 1.0\text{ m}, \text{min\_samples} = 4$)**:
```bash
python main.py
```

### **B. Run DBSCAN Parameter Sensitivity Experiment ($24$ Configurations)**:
```bash
python -m src.parameter_experiments
```

### **C. Run DBSCAN Failure Analysis (Surviving Ghost Analysis)**:
```bash
python -m src.failure_analysis
```

---

## 6. Output Files Summary ([`outputs/`](file:///e:/ABAGA/ewmvc_dbscan/outputs))

- [`transformed_observations.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/transformed_observations.csv): Perception log augmented with interpolated vehicle pose and global $(x, y)$ coordinates.
- [`dbscan_observations.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_observations.csv): Observations augmented with assigned DBSCAN `cluster_id` (noise $= -1$).
- [`cone_map_dbscan.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/cone_map_dbscan.csv): $72$ confidence-weighted cone centroids and cluster metrics.
- [`evaluation.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/evaluation.csv): Observation-level clustering diagnostics.
- [`global_observations.png`](file:///e:/ABAGA/ewmvc_dbscan/outputs/global_observations.png): Raw global cone observations and vehicle trajectory plot.
- [`dbscan_map.png`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_map.png): DBSCAN clustering map plot.
- [`dbscan_parameter_experiments.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_parameter_experiments.csv): Parameter sensitivity grid results ($24$ configurations).
- [`baseline_surviving_ghosts.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/baseline_surviving_ghosts.csv): $14$ surviving ghost observations incorporated into baseline clusters.
- [`dbscan_failure_analysis.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_failure_analysis.csv): Detailed failure metrics per surviving ghost.
- [`dbscan_cluster_contamination.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_cluster_contamination.csv): Contamination statistics per contaminated cluster.
- [`dbscan_failure_summary.csv`](file:///e:/ABAGA/ewmvc_dbscan/outputs/dbscan_failure_summary.csv): Aggregate failure analysis summary metrics.
- Sensitivity & Failure Plots:
  - `ghost_rejection_vs_eps.png`
  - `number_of_clusters_vs_eps.png`
  - `noise_fraction_vs_eps.png`
  - `dbscan_parameter_heatmap.png`
  - `surviving_ghosts_map.png`
  - `ghost_distance_to_centroid.png`
  - `ghost_cluster_contamination.png`

---

## 7. Evaluation Disclaimer

> [!IMPORTANT]
> **Observation-Level Diagnostics vs. Landmark Evaluation**
>
> The evaluation metrics in `src/evaluate.py`, `outputs/evaluation.csv`, and `outputs/dbscan_parameter_experiments.csv` measure **observation-level DBSCAN clustering behavior** and **ghost observation filtering**.
> Because the dataset contains frame-level perception labels (`real` vs `ghost`) but lacks ground-truth cone map landmark IDs and ground-truth coordinates, **landmark precision, recall, and F1 scores are NOT calculated**.

---

## 8. Current DBSCAN Structural Limitations

Empirical failure analysis on the 14 surviving ghost observations identified three key structural limitations of conventional Euclidean DBSCAN:

1. **Spatial Proximity Adsorption**:
   All 14 surviving ghost observations lie within $\approx 0.53\text{ m}$ (mean nearest-real distance) of a dense landmark cluster. Because standard Euclidean DBSCAN relies purely on unweighted spatial distance ($d \le \epsilon = 1.0\text{ m}$), any false positive that occurs spatially close to a real landmark is unconditionally absorbed into the cluster.

2. **Confidence Indiscrimination**:
   The mean detection confidence of surviving ghosts (`0.3829`) is substantially lower than real cone observations (`0.8775`). Standard DBSCAN assigns binary weight (1 vs 0) to spatial points, ignoring perception detection confidence and sensor range uncertainty.

3. **Temporal Blindness & Single-Frame Transients**:
   Every surviving ghost appears as an isolated, transient single-frame anomaly ($1$ timestamp), whereas valid cone landmarks persist across $100+$ frames ($>73.9\text{ s}$ time span). Standard DBSCAN operates on static point clouds, lacking multi-view temporal continuity and persistence scoring.
