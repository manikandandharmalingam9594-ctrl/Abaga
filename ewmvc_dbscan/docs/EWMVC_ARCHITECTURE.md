# Entropy-Weighted Multi-View Consensus (EWMVC) Mapping Architecture

## Implementation Status

- **Uncertainty Propagation**: IMPLEMENTED (`src/uncertainty.py`)
- **Mahalanobis Graph**: IMPLEMENTED (`src/graph_builder.py`)
- **Entropy**: IMPLEMENTED (`src/entropy.py`)
- **Geometry**: IMPLEMENTED (`src/geometry.py`)
- **Persistence**: IMPLEMENTED (`src/persistence.py`)
- **Diversity**: IMPLEMENTED (`src/diversity.py`)
- **Consensus**: IMPLEMENTED (`src/consensus.py`)
- **Landmark Estimation**: IMPLEMENTED (`src/landmark.py`)
- **Track Reconstruction**: IMPLEMENTED (`src/track_recon.py`)

> [!NOTE]
> **Dataset Covariance Disclaimer**: The dataset does NOT contain measured sensor or pose covariance values. Therefore, all uncertainty parameters (`sigma_sensor_x`, `sigma_sensor_y`, `sigma_pose_x`, `sigma_pose_y`, `sigma_pose_yaw`) are **configured design assumptions** defined in `configs/config.yaml`.

---

## 1. Problem Statement
Autonomous racing vehicles operating in Formula Student environments construct global track boundary maps (left blue cones, right yellow cones, orange track limit cones) using noisy perception observations and vehicle state telemetry. 

While spatial clustering algorithms (such as DBSCAN) can aggregate high-frequency 2D observations into landmark estimates, they suffer from structural limitations when handling false detections (ghost observations) and sensor measurement uncertainties. EWMVC is designed to resolve these limitations by combining spatial uncertainty, multi-view viewpoint diversity, temporal persistence, and information-theoretic entropy into a robust consensus scoring framework.

---

## 2. DBSCAN Baseline & Verified Dataset Results

The current verified fixed-epsilon DBSCAN baseline pipeline operates under the following setup:
- **Sensor Mounting Rotation**: $180.0^\circ$
- **DBSCAN Epsilon ($\epsilon$)**: $1.0\text{ m}$
- **Min Samples**: $4$

### **Verified Dataset Metrics (Ground Truth Verification)**:
> [!NOTE]
> The following metrics represent **VERIFIED DATASET RESULTS** on the actual log files (`perception_log.csv` and `telemetry_log.csv`):

| Metric | Verified Dataset Value | Description |
| :--- | :---: | :--- |
| **Total Observations** | `9,774` | Total perception observations in raw log |
| **Clustered Observations** | `9,708` | Observations assigned to non-noise clusters |
| **Noise Observations** | `66` | Observations labeled as DBSCAN noise (`-1`) |
| **Noise Fraction** | `0.006753` (`0.68%`) | Percentage of noise observations |
| **Total Ghost Observations** | `80` | Total perception false positives (`label == 'ghost'`) |
| **Ghosts Rejected as Noise** | `66` | Ghost observations correctly filtered by DBSCAN |
| **Ghosts Inside Clusters** | `14` | Ghost observations incorrectly absorbed into clusters |
| **Ghost Rejection Rate** | `0.825000` (`82.50%`) | Percentage of ghost observations rejected |
| **Number of Clusters** | `72` | Total cone landmark clusters mapped |
| **Automated Test Suite** | `24 / 24 Passed` | Pytest baseline regression suite (`pytest -v`) |

---

## 3. Baseline Limitations Discovered Experimentally (Motivation for EWMVC)

Empirical failure analysis on the 14 surviving ghost observations revealed three core structural failure modes of conventional Euclidean DBSCAN:

1. **Spatial Proximity Adsorption**:
   All 14 surviving ghost observations lie within $\approx 0.5333\text{ m}$ (mean distance to nearest real cone observation) of a dense valid cone cluster. Because standard DBSCAN relies purely on unweighted Euclidean distance ($d \le \epsilon = 1.0\text{ m}$), any false positive occurring near a real landmark is unconditionally absorbed.
2. **Confidence Indiscrimination**:
   Surviving ghosts have a significantly lower mean perception confidence (`0.3829`) compared to real cone observations (`0.8775`). Standard DBSCAN applies binary spatial thresholding and ignores detection confidence and sensor range noise.
3. **Temporal Blindness & Single-Frame Transients**:
   All 14 surviving ghosts are single-frame perception anomalies (1 timestamp frame), whereas valid cone landmarks persist across $>129.6$ perception frames over a mean time span of $73.94\text{ s}$. Standard DBSCAN operates on static point clouds without temporal continuity or multi-view persistence scoring.

---

## 4. EWMVC Proposed Pipeline Architecture

```
                       [Raw Perception CSV]         [Raw Telemetry CSV]
                                │                            │
                                └─────────────┬──────────────┘
                                              ▼
                                    (1. Data Loader & Schema Check)
                                              │
                                              ▼
                                    (2. Telemetry Pose Interpolation)
                                              │
                                              ▼
                                    (3. Sensor -> Vehicle -> Global SE(2))
                                              │
                                              ▼
                                    (4. Covariance Propagation)
                                              │
                                              ▼
                                    (5. Mahalanobis Compatibility Graph)
                                              │
                                              ▼
                                    (6. Candidate Landmark Component Extraction)
                                              │
                                              ▼
                                ┌─────────────┴─────────────┐
                                │ Multi-View Analysis Suite │
                                ├───────────────────────────┤
                                │ - Viewpoint Entropy (H)   │
                                │ - Geometric Consistency   │
                                │ - Temporal Persistence    │
                                │ - Viewpoint Diversity     │
                                └─────────────┬─────────────┘
                                              ▼
                                    (7. Weighted Consensus Scoring)
                                              │
                                              ▼
                                    (8. Ghost Rejection Thresholding)
                                              │
                                              ▼
                                    (9. Inverse-Covariance Landmark Mean)
                                              │
                                              ▼
                                    (10. Track Boundary Reconstruction)
```

---

## 5. Detailed Module & Interface Specifications

> [!IMPORTANT]
> **PROPOSED EWMVC DESIGN PARAMETERS vs. VERIFIED DATASET RESULTS**
> All mathematical formulations, covariance models, entropy weights, and thresholds below represent **PROPOSED DESIGN SPECIFICATIONS** for future implementation. They are NOT experimentally validated values.

### **Module 1: Uncertainty Estimation (`src/uncertainty.py`)**
- **Purpose**: Propagate perception range/bearing sensor noise and vehicle pose uncertainty into a 2D global spatial covariance matrix $\Sigma_i \in \mathbb{R}^{2 \times 2}$ for each observation $i$.
- **Inputs**: `rel_x_sensor`, `rel_y_sensor`, `vehicle_x`, `vehicle_y`, `vehicle_yaw`, perception confidence.
- **Outputs**: `cov_xx`, `cov_xy`, `cov_yy` (per-observation covariance matrix elements).
- **Available Dataset Fields**: `rel_x_sensor`, `rel_y_sensor`, `vehicle_x`, `vehicle_y`, `vehicle_yaw`.
- **Configured/Assumed Parameters** *(Design Proposal)*:
  - Sensor range noise variance: $\sigma_r^2 = (0.05\text{ m} + 0.02 \cdot r)^2$
  - Sensor bearing noise variance: $\sigma_\theta^2 = (1.5^\circ \cdot \frac{\pi}{180})^2$
  - Vehicle pose position variance: $\sigma_{x,\text{veh}}^2 = \sigma_{y,\text{veh}}^2 = 0.04\text{ m}^2$
  - Vehicle heading variance: $\sigma_{\psi,\text{veh}}^2 = (1.0^\circ \cdot \frac{\pi}{180})^2$
- **Mathematical Formulations**:
  For observation $i$ transformed via $SE(2)$:
  $$\mathbf{p}_g = \mathbf{p}_{\text{veh}} + \mathbf{R}(\psi) \mathbf{R}(\theta_{\text{sensor}}) \mathbf{p}_{\text{sensor}}$$
  Jacobians $\mathbf{J}_{\text{pose}} = \frac{\partial \mathbf{p}_g}{\partial \mathbf{x}_{\text{veh}}}$ and $\mathbf{J}_{\text{sensor}} = \frac{\partial \mathbf{p}_g}{\partial \mathbf{z}_{\text{sensor}}}$ yield:
  $$\mathbf{\Sigma}_{i} = \mathbf{J}_{\text{pose}} \mathbf{\Sigma}_{\text{veh}} \mathbf{J}_{\text{pose}}^T + \mathbf{J}_{\text{sensor}} \mathbf{\Sigma}_{\text{sensor}} \mathbf{J}_{\text{sensor}}^T$$
- **Dependencies**: `numpy`, `pandas`.
- **Failure Cases**: Singular covariance matrices (near-zero range), non-positive definite $\mathbf{\Sigma}_i$.
- **Future Unit Tests**: Test linear propagation at $\psi=0$ and $\psi=\pi/2$; test symmetry and positive-definiteness of $\mathbf{\Sigma}_i$.

---

### **Module 2: Mahalanobis Compatibility & Graph Building (`src/graph_builder.py`) - IMPLEMENTED**
- **Purpose**: Compute pairwise statistical distances between global observations using their propagated spatial covariance matrices and extract connected-component candidate landmarks.
- **Inputs**: `global_x`, `global_y`, `cov_xx`, `cov_xy`, `cov_yy` (from `outputs/ewmvc_observations_with_covariance.csv`).
- **Outputs**:
  - `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `mahalanobis_graph_degree`).
  - `outputs/ewmvc_graph_summary.csv` (`total_observations`, `total_edges`, `number_of_components`, `singleton_components`, `largest_component_size`, `average_component_size`).
  - `outputs/ewmvc_candidate_components.png` (Visual plot titled `"EWMVC Candidate Landmark Components"`).
- **Mathematical Formulations**:
  Pairwise Mahalanobis distance squared between 2D observations $i$ and $j$:
  $$d_{M}^2(i, j) = (\mathbf{p}_i - \mathbf{p}_j)^T \left(\mathbf{\Sigma}_i + \mathbf{\Sigma}_j\right)^{-1} (\mathbf{p}_i - \mathbf{p}_j)$$
  where $\mathbf{p}_i = [x_i, y_i]^T$ and $\mathbf{\Sigma}_{ij} = \mathbf{\Sigma}_i + \mathbf{\Sigma}_j$.
  An undirected edge $(i, j)$ exists if $d_{M}^2(i, j) \le \chi_{2, 0.95}^2$.

- **Configured/Theoretical Parameters**:
  - **Chi-Square 2-DOF Threshold**: $\chi^2_{2, 0.95} = 5.991$ (Theoretical 95% confidence gating threshold for 2 degrees of freedom).
  - **Candidate Spatial Pre-filter Radius**: $R_{\max} = 1.5\text{ m}$ (Computational optimization using `scipy.spatial.cKDTree` to avoid $O(N^2)$ pairwise checks over $N=9,774$ observations).
  > [!IMPORTANT]
  > **Computational Pre-Filter vs. Decision Rule Distinction**:
  > $R_{\max} = 1.5\text{ m}$ is ONLY a spatial pre-filter optimization to rapidly eliminate pairs that are physically too far apart to be statistically compatible. The final compatibility decision rule is strictly determined by Mahalanobis distance $d^2_M(i, j) \le 5.991$.

- **Numerical Stability & Singular Matrix Handling**:
  - Checks finiteness of positions and covariance matrices.
  - Verifies symmetry ($\text{cov\_xy}_i = \text{cov\_yx}_i$).
  - Evaluates determinant $\det(\mathbf{\Sigma}_{ij}) = a c - b^2$. If $\det \le 1e-12$ or non-positive-definite, falls back to pseudo-inverse `np.linalg.pinv` or rejects the edge, preventing silent numerical crashes.

- **Connected Component Candidate Extraction**:
  - Extracted using `scipy.sparse.csgraph.connected_components`.
  - Singletons (components of size 1) are preserved with degree 0 and unique `candidate_id`s for downstream EWMVC consensus scoring.
  - Graph construction strictly operates without using perception `label` (`real`/`ghost`) or DBSCAN `cluster_id`.

- **Computational Complexity**:
  - Naive All-Pairs: $O(N^2) \approx 47.7\times 10^6$ evaluations.
  - KD-Tree Pre-filtered Strategy: $O(N \log N)$ search, evaluating $\approx 5.6 \times 10^5$ candidate pairs.

- **Verified Measured Dataset Results ($N=9,774$ observations)**:
  | Metric | Value |
  | :--- | :---: |
  | **Total Observations** | `9,774` |
  | **Total Graph Edges** | `562,275` |
  | **Number of Candidate Components** | `141` |
  | **Singleton Components** | `67` |
  | **Largest Component Size** | `176` |
  | **Average Component Size** | `69.3191` |

- **Current Limitations**:
  - Connected components extracted from the Mahalanobis graph represent unrefined candidate landmarks. Because statistical compatibility bridges close observations, candidate component count ($141$) does NOT equal true landmark count ($72$). Multi-view consensus scoring, viewpoint entropy, and persistence filtering (Steps 20C+) are required to filter out transient ghost components and refine landmarks.

- **Automated Tests**: 10 comprehensive unit tests implemented in `tests/test_graph_builder.py`.

---

### **Module 3: Viewpoint Entropy (`src/entropy.py`) - IMPLEMENTED**
- **Purpose**: Calculate the normalized angular distribution entropy of vehicle observation viewpoints relative to each candidate landmark component.
- **Inputs**: `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `global_x`, `global_y`, `vehicle_x`, `vehicle_y`).
- **Outputs**:
  - `outputs/ewmvc_candidate_entropy.csv` (`candidate_id`, `candidate_x`, `candidate_y`, `num_observations`, `unique_viewpoint_bins`, `viewpoint_entropy`, `bin_0` .. `bin_7`).
  - `outputs/ewmvc_entropy_distribution.png` (Histogram of candidate entropy values).
  - `outputs/ewmvc_entropy_vs_component_size.png` (Scatter plot of entropy vs component size).

- **Mathematical Formulations**:
  1. **Provisional Candidate Centroid**:
     For candidate component $k$ with $N_k$ observations:
     $$\hat{c}_{k,x} = \frac{1}{N_k} \sum_{i \in k} x_{g,i}, \quad \hat{c}_{k,y} = \frac{1}{N_k} \sum_{i \in k} y_{g,i}$$
     > [!NOTE]
     > The mean position $(\hat{c}_{k,x}, \hat{c}_{k,y})$ is **ONLY a provisional reference** for computing viewpoint vectors, not the final estimated landmark coordinate.

  2. **Observation Viewpoint Angle**:
     For observation $i \in k$ observed from vehicle pose $(x_{v,i}, y_{v,i})$:
     $$\theta_{\text{view}, i} = \operatorname{atan2}\left(\hat{c}_{k,y} - y_{v,i}, \, \hat{c}_{k,x} - x_{v,i}\right) \in [-\pi, \pi)$$

  3. **8 Uniform Angular Viewpoint Bins**:
     Discretize the $360^\circ$ ($2\pi$ rad) angular domain into $N_{\text{bins}} = 8$ uniform sectors of width $\Delta \theta = 45^\circ$ ($\pi/4$ rad):
     $$b_i = \left\lfloor \frac{\theta_{\text{view}, i} + \pi}{\pi / 4} \right\rfloor \bmod 8$$

  4. **Shannon Entropy & Normalization**:
     Let $p_b = \frac{n_b}{N_k}$ be the proportion of observations in bin $b \in \{0, \dots, 7\}$:
     $$H_k = -\sum_{b: p_b > 0} p_b \ln(p_b)$$
     Normalized Viewpoint Entropy:
     $$H_{\text{norm}, k} = \frac{H_k}{\ln(8)} \in [0.0, 1.0]$$
     (Clamped numerically to $[0.0, 1.0]$. Singletons and single-bin candidates yield $H_{\text{norm}, k} = 0.0$).

  > [!IMPORTANT]
  > **Entropy is NOT a Standalone Ghost Rejection Rule**:
  > Viewpoint entropy measures observation angular diversity. Low entropy does NOT automatically prove a component is a ghost (e.g., valid cones viewed from a single straight trajectory also have low entropy). Entropy is strictly one feature for downstream multi-view consensus scoring.

- **Verified Measured Dataset Results ($141$ Candidate Components)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total Candidate Components** | `141` |
  | **Minimum Viewpoint Entropy** | `0.000000` |
  | **Maximum Viewpoint Entropy** | `0.552839` |
  | **Mean Viewpoint Entropy** | `0.165095` |
  | **Median Viewpoint Entropy** | `0.088922` |
  | **Components with Entropy $\approx 0$** | `69` |
  | **Components with Entropy $> 0.5$** | `2` |

- **Unique Viewpoint Bins Distribution**:
  | Occupied Angular Bins | Component Count | Description |
  | :---: | :---: | :--- |
  | **1 Bin** | `69` | Singletons ($67$) & single-viewpoint observations ($2$) |
  | **2 Bins** | `20` | Observations spanning two adjacent $45^\circ$ sectors |
  | **3 Bins** | `45` | Observations spanning three $45^\circ$ sectors |
  | **4 Bins** | `7` | Observations spanning four $45^\circ$ sectors |
  | **5--8 Bins** | `0` | No candidate component currently spans $>4$ sectors |

- **Automated Tests**: 6 comprehensive unit tests implemented in `tests/test_entropy.py`.

---

### **Module 4: Geometric Consistency (`src/geometry.py`) - IMPLEMENTED**
- **Purpose**: Compute RMS spatial dispersion and geometry-consistency scores for candidate landmark components based on observation spatial concentration.
- **Inputs**: `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `global_x`, `global_y`).
- **Outputs**:
  - `outputs/ewmvc_candidate_geometry.csv` (`candidate_id`, `num_observations`, `candidate_x`, `candidate_y`, `sigma_spatial_m`, `median_radius_m`, `p95_radius_m`, `max_radius_m`, `geometry_score`).
  - `outputs/ewmvc_geometry_score_distribution.png` (Histogram of geometry scores).
  - `outputs/ewmvc_geometry_vs_component_size.png` (Scatter plot of geometry score vs component size).

- **Mathematical Formulations**:
  1. **Provisional Component Center**:
     For candidate component $k$ with $N_k$ observations:
     $$\mu_{x, k} = \frac{1}{N_k} \sum_{i \in k} x_{g,i}, \quad \mu_{y, k} = \frac{1}{N_k} \sum_{i \in k} y_{g,i}$$
     > [!NOTE]
     > The mean position $(\mu_{x, k}, \mu_{y, k})$ is **ONLY a provisional reference centroid** for distance calculations, not the final estimated landmark coordinate.

  2. **Observation Radius & RMS Spatial Dispersion**:
     For observation $i \in k$:
     $$r_i = \sqrt{(x_{g,i} - \mu_{x,k})^2 + (y_{g,i} - \mu_{y,k})^2}$$
     $$\sigma_{\text{spatial}, k} = \sqrt{\frac{1}{N_k} \sum_{i \in k} r_i^2}$$

  3. **Geometry Consistency Score**:
     $$w_{\text{geo}, k} = \exp\left(-\frac{\sigma_{\text{spatial}, k}^2}{2 \sigma_0^2}\right) \in (0, 1]$$
     (Where $\sigma_{\text{spatial}} = 0 \implies w_{\text{geo}} = 1.0$, singletons yield $w_{\text{geo}} = 1.0$).

  > [!IMPORTANT]
  > **Design Scale Parameter Disclaimer**:
  > $\sigma_0 = 0.75\text{ m}$ is a **configured design parameter** (defined in `configs/config.yaml`), NOT an empirically optimized classifier threshold. It evaluates spatial concentration without accepting/rejecting landmarks.

- **Verified Measured Dataset Results ($141$ Candidate Components)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total Candidate Components** | `141` |
  | **Minimum Geometry Score** | `0.858700` |
  | **Maximum Geometry Score** | `1.000000` |
  | **Mean Geometry Score** | `0.970102` |
  | **Median Geometry Score** | `0.951484` |
  | **Minimum $\sigma_{\text{spatial}}$** | `0.000000 m` |
  | **Maximum $\sigma_{\text{spatial}}$** | `0.413978 m` |
  | **Mean $\sigma_{\text{spatial}}$** | `0.134442 m` |
  | **Median $\sigma_{\text{spatial}}$** | `0.236535 m` |

- **Threshold Distribution Extrema Counts**:
  | Geometry Score Range | Component Count |
  | :---: | :---: |
  | **$w_{\text{geo}} > 0.5$** | `141 / 141` |
  | **$w_{\text{geo}} > 0.8$** | `141 / 141` |
  | **$w_{\text{geo}} > 0.9$** | `140 / 141` |

- **Automated Tests**: 6 comprehensive unit tests implemented in `tests/test_geometry.py`.

---

### **Module 5: Temporal Persistence (`src/persistence.py`) - IMPLEMENTED**
- **Purpose**: Calculate temporal duration and persistence scores for candidate landmark components based on multi-frame perception observation timestamps.
- **Inputs**: `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `timestamp`).
- **Outputs**:
  - `outputs/ewmvc_candidate_persistence.csv` (`candidate_id`, `observation_count`, `num_unique_timestamps`, `t_min`, `t_max`, `duration_s`, `persistence_score`).
  - `outputs/ewmvc_persistence_distribution.png` (Histogram of persistence scores).
  - `outputs/ewmvc_persistence_vs_component_size.png` (Scatter plot of persistence score vs component size).

- **Mathematical Formulations**:
  1. **Time Span & Duration**:
     For candidate component $k$:
     $$t_{\min, k} = \min_{i \in k}(t_i), \quad t_{\max, k} = \max_{i \in k}(t_i)$$
     $$\text{duration\_s}_k = t_{\max, k} - t_{\min, k}$$

  2. **Unique Timestamps vs. Observation Count**:
     $$N_{\text{unique\_t}, k} = |\{t_i \mid i \in k\}|$$
     > [!IMPORTANT]
     > **Multiple Observations Per Timestamp**: The perception log contains multiple observations per timestamp frame. Therefore, `observation_count` $\neq$ `num_unique_timestamps`. Unique timestamps measure true multi-frame temporal presence.

  3. **Persistence Score**:
     $$\text{Score}_{\text{persistence}, k} = \min\left(1.0, \frac{\text{duration\_s}_k}{T_{\text{expected}}}\right) \in [0.0, 1.0]$$
     (Where $T_{\text{expected}} = 10.0\text{ s}$. Singletons yield $\text{duration\_s} = 0 \implies \text{Score}_{\text{persistence}} = 0.0$).

  > [!IMPORTANT]
  > **Design Scale Parameter Disclaimer**:
  > $T_{\text{expected}} = 10.0\text{ s}$ is a **configured design parameter** (defined in `configs/config.yaml`), NOT an empirically optimized threshold. Persistence is a feature for downstream consensus scoring, not an independent decision rule.

- **Verified Measured Dataset Results ($141$ Candidate Components)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total Candidate Components** | `141` |
  | **Minimum Duration** | `0.000000 s` |
  | **Maximum Duration** | `119.900000 s` |
  | **Mean Duration** | `37.417021 s` |
  | **Median Duration** | `65.000000 s` |
  | **Minimum Persistence Score** | `0.000000` |
  | **Maximum Persistence Score** | `1.000000` |
  | **Mean Persistence Score** | `0.519220` |
  | **Median Persistence Score** | `1.000000` |

- **Score Extrema Counts**:
  | Persistence Score Range | Component Count | Description |
  | :---: | :---: | :--- |
  | **Score == 0.0** | `67` | Transient single-frame / singleton candidates |
  | **Score > 0.5** | `73` | Candidates persisting $>5.0\text{ s}$ |
  | **Score >= 1.0** | `73` | Candidates persisting $\ge 10.0\text{ s}$ |

- **Unique Timestamps Distribution**:
  | Unique Timestamps | Component Count |
  | :---: | :---: |
  | **1 Unique Timestamp** | `67` |
  | **$\ge 2$ Unique Timestamps** | `74` |

- **Automated Tests**: 6 comprehensive unit tests implemented in `tests/test_persistence.py`.

---

### **Module 6: Viewpoint Diversity (`src/diversity.py`) - IMPLEMENTED**
- **Purpose**: Calculate vehicle displacement baseline features and viewpoint diversity scores for candidate landmark components based on unique vehicle observation poses.
- **Inputs**: `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `timestamp`, `vehicle_x`, `vehicle_y`).
- **Outputs**:
  - `outputs/ewmvc_candidate_diversity.csv` (`candidate_id`, `observation_count`, `num_unique_timestamps`, `max_baseline_m`, `mean_pairwise_baseline_m`, `median_pairwise_baseline_m`, `min_pairwise_baseline_m`, `viewpoint_diversity`).
  - `outputs/ewmvc_diversity_distribution.png` (Histogram of diversity scores).
  - `outputs/ewmvc_diversity_vs_component_size.png` (Scatter plot of diversity score vs component size).
  - `outputs/ewmvc_diversity_vs_entropy.png` (Scatter plot of diversity score vs viewpoint entropy).

- **Mathematical Formulations**:
  1. **Unique Viewpoint Deduplication**:
     Multiple perception observations at the same timestamp originate from the same vehicle pose $(x_{v, t}, y_{v, t})$. Vehicle positions are deduplicated per timestamp frame.
  
  2. **Pairwise Vehicle Displacement Baseline**:
     For unique vehicle poses $i, j$ associated with candidate component $k$:
     $$d_{ij} = \sqrt{(x_{v,i} - x_{v,j})^2 + (y_{v,i} - y_{v,j})^2}$$
     $$\text{max\_baseline\_m}_k = \max_{i, j} d_{ij}$$

  3. **Viewpoint Diversity Score**:
     $$\text{Score}_{\text{diversity}, k} = \min\left(1.0, \frac{\text{max\_baseline\_m}_k}{D_{\text{sat}}}\right) \in [0.0, 1.0]$$
     (Where $D_{\text{sat}} = 5.0\text{ m}$. Singletons yield $\text{max\_baseline\_m} = 0 \implies \text{Score}_{\text{diversity}} = 0.0$).

  > [!IMPORTANT]
  > **Design Saturation Scale Disclaimer & Viewpoint Distinctions**:
  > - $D_{\text{sat}} = 5.0\text{ m}$ is a **configured design parameter** (defined in `configs/config.yaml`), NOT an empirically optimized threshold.
  > - **Diversity vs. Entropy**: Viewpoint diversity measures physical vehicle displacement distance ($m$). Viewpoint entropy measures angular sector distribution. Both features remain independent inputs to downstream consensus scoring.
  > - Duplicate timestamps are deduplicated so multiple observations at a single vehicle pose do NOT create artificial viewpoint diversity.

- **Verified Measured Dataset Results ($141$ Candidate Components)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total Candidate Components** | `141` |
  | **Minimum Max Baseline** | `0.000000 m` |
  | **Maximum Max Baseline** | `26.544151 m` |
  | **Mean Max Baseline** | `11.258179 m` |
  | **Median Max Baseline** | `18.351956 m` |
  | **Minimum Mean Pairwise Baseline** | `0.000000 m` |
  | **Maximum Mean Pairwise Baseline** | `9.216800 m` |
  | **Mean Mean Pairwise Baseline** | `3.935231 m` |
  | **Median Mean Pairwise Baseline** | `6.634038 m` |
  | **Minimum Diversity Score** | `0.000000` |
  | **Maximum Diversity Score** | `1.000000` |
  | **Mean Diversity Score** | `0.518473` |
  | **Median Diversity Score** | `1.000000` |

- **Score Extrema Counts**:
  | Diversity Score Range | Component Count | Description |
  | :---: | :---: | :--- |
  | **Score == 0.0** | `67` | Singletons & single-viewpoint candidates |
  | **Score > 0.5** | `73` | Candidates with vehicle displacement $>2.5\text{ m}$ |
  | **Score >= 1.0** | `73` | Candidates reaching saturation displacement $\ge 5.0\text{ m}$ |

- **Automated Tests**: 7 comprehensive unit tests implemented in `tests/test_diversity.py`.

---

### **Module 7: Weighted Consensus Scoring (`src/consensus.py`) - IMPLEMENTED**
- **Purpose**: Aggregate four independent multi-view evidence features (Entropy $H$, Geometry $G$, Persistence $P$, Diversity $D$) into a unified candidate reliability score $C \in [0.0, 1.0]$.
- **Inputs**: `outputs/ewmvc_candidate_entropy.csv`, `outputs/ewmvc_candidate_geometry.csv`, `outputs/ewmvc_candidate_persistence.csv`, `outputs/ewmvc_candidate_diversity.csv`.
- **Outputs**:
  - `outputs/ewmvc_consensus.csv` (`candidate_id`, features, feature contributions, `consensus_score`, `accepted` flag).
  - `outputs/ewmvc_feature_correlation.csv` (Pearson correlation matrix).
  - `outputs/ewmvc_consensus_ablation.csv` (Leave-one-feature-out ablation diagnostics).
  - `outputs/ewmvc_consensus_distribution.png` (Histogram of consensus scores with $0.60$ threshold marker).
  - `outputs/ewmvc_consensus_vs_entropy.png`, `ewmvc_consensus_vs_geometry.png`, `ewmvc_consensus_vs_persistence.png`, `ewmvc_consensus_vs_diversity.png` (Feature scatter plots).
  - `outputs/ewmvc_feature_correlation.png` (Correlation heatmap).

- **Mathematical Formulations**:
  1. **Linear Evidence Combination**:
     $$C_k = w_H \cdot H_k + w_G \cdot G_k + w_D \cdot D_k + w_P \cdot P_k \in [0.0, 1.0]$$

  2. **Feature Contributions**:
     $$\text{contrib}_H = w_H \cdot H, \quad \text{contrib}_G = w_G \cdot G, \quad \text{contrib}_D = w_D \cdot D, \quad \text{contrib}_P = w_P \cdot P$$

  3. **Diagnostic Acceptance Flag**:
     $$\text{accepted}_k = \left(C_k \ge \tau_{\text{acceptance}}\right)$$

- **Configured Design Parameters**:
  - **Proposed Weights**: $w_H = 0.30, w_G = 0.30, w_D = 0.20, w_P = 0.20$ ($\sum w = 1.0$).
  - **Diagnostic Threshold**: $\tau_{\text{acceptance}} = 0.60$.
  > [!IMPORTANT]
  > **Design Parameter & Terminology Disclaimer**:
  > - Weights and threshold are **configured design parameters** (defined in `configs/config.yaml`), NOT empirically optimized against ground truth labels.
  > - Accepted components are termed **"consensus-supported candidate landmarks"**, NOT confirmed ground truth cones.

- **Verified Measured Dataset Results ($141$ Candidate Components)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total Candidate Components** | `141` |
  | **Minimum Consensus Score** | `0.300000` |
  | **Maximum Consensus Score** | `0.848187` |
  | **Mean Consensus Score** | `0.548098` |
  | **Median Consensus Score** | `0.709138` |
  | **Accepted Candidates ($\tau = 0.60$)** | `72` (`51.06%`) |
  | **Rejected Candidates ($\tau = 0.60$)** | `69` (`48.94%`) |

- **Feature Pearson Correlation Matrix**:
  | Feature Pair | Pearson Correlation ($r$) | Insight |
  | :--- | :---: | :--- |
  | **(Persistence, Diversity)** | `+0.979677` | Extremely high correlation; vehicle displacement strongly aligns with time duration. |
  | **(Entropy, Diversity)** | `+0.863465` | High correlation; physical vehicle movement creates multi-angle viewpoints. |
  | **(Entropy, Persistence)** | `+0.863062` | High correlation; longer observation time enables multi-angle views. |
  | **(Geometry, Persistence)** | `-0.959343` | Negative correlation; singletons ($N=1$) have perfect $G=1.0$ but $P=0.0$. |
  | **(Geometry, Diversity)** | `-0.917059` | Negative correlation; singletons have $G=1.0$ but $D=0.0$. |
  | **(Geometry, Entropy)** | `-0.792262` | Negative correlation; single-view singletons have $G=1.0$ but $H=0.0$. |

- **Diagnostic Leave-One-Out Ablation Analysis**:
  | Ablated Feature | Remaining Features | Mean Score | Median Score | Accepted Count ($\tau=0.60$) | Accepted % |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | **Entropy** | Geometry, Diversity, Persistence | `0.712242` | `0.971374` | `74` | `52.48%` |
  | **Geometry** | Entropy, Diversity, Persistence | `0.367239` | `0.609538` | `72` | `51.06%` |
  | **Persistence** | Entropy, Geometry, Diversity | `0.555317` | `0.636422` | `73` | `51.77%` |
  | **Diversity** | Entropy, Geometry, Persistence | `0.555504` | `0.636422` | `72` | `51.06%` |

- **Automated Tests**: 7 comprehensive unit tests implemented in `tests/test_consensus.py`.

---

### **Module 8: Inverse-Covariance Landmark Estimation (`src/landmark.py`) - IMPLEMENTED**
- **Purpose**: Estimate single representative 2D global position $\hat{\mathbf{p}}_k$ and 2D spatial covariance matrix $\mathbf{\Sigma}_{\text{landmark}, k}$ for each consensus-supported candidate component (`accepted == True`).
- **Inputs**: `outputs/ewmvc_candidate_components.csv` (`candidate_id`, `global_x`, `global_y`, `cov_xx`, `cov_xy`, `cov_yy`) and `outputs/ewmvc_consensus.csv` (`candidate_id`, `accepted`, `consensus_score`, etc.).
- **Outputs**:
  - `outputs/ewmvc_landmarks.csv` (`candidate_id`, `landmark_x`, `landmark_y`, `landmark_cov_xx`, `landmark_cov_xy`, `landmark_cov_yy`, `landmark_sigma_x`, `landmark_sigma_y`, `consensus_score`, `weighted_shift_m`, and diagnostics).
  - `outputs/ewmvc_landmark_map.png` (Consensus-supported landmark map overlaid on vehicle trajectory and perception cloud).
  - `outputs/ewmvc_landmark_uncertainty.png` (Landmark positions with 1-sigma uncertainty ellipses).

- **Mathematical Formulations**:
  1. **Precision-Weighted Inverse-Covariance Fusion**:
     For observation $i \in k$ with 2D position $\mathbf{p}_i = [x_i, y_i]^T$ and covariance matrix $\mathbf{\Sigma}_i$:
     $$\mathbf{W}_i = \mathbf{\Sigma}_i^{-1}$$
     $$\mathbf{W}_k = \sum_{i \in k} \mathbf{W}_i$$
     $$\mathbf{b}_k = \sum_{i \in k} \mathbf{W}_i \mathbf{p}_i$$
     $$\hat{\mathbf{p}}_k = \mathbf{W}_k^{-1} \mathbf{b}_k = \left(\sum_{i \in k} \mathbf{\Sigma}_i^{-1}\right)^{-1} \sum_{i \in k} \left(\mathbf{\Sigma}_i^{-1} \mathbf{p}_i\right)$$

  2. **Fused Landmark Covariance**:
     $$\mathbf{\Sigma}_{\text{landmark}, k} = \mathbf{W}_k^{-1} = \left(\sum_{i \in k} \mathbf{\Sigma}_i^{-1}\right)^{-1}$$
     $$\sigma_{\text{landmark}, x} = \sqrt{\mathbf{\Sigma}_{\text{landmark}, k}[0, 0]}, \quad \sigma_{\text{landmark}, y} = \sqrt{\mathbf{\Sigma}_{\text{landmark}, k}[1, 1]}$$

  3. **Diagnostic Weighted Centroid Shift**:
     Euclidean shift relative to ordinary arithmetic mean centroid $(\bar{x}_k, \bar{y}_k)$:
     $$\text{weighted\_shift\_m}_k = \sqrt{(\hat{p}_{k,x} - \bar{x}_k)^2 + (\hat{p}_{k,y} - \bar{y}_k)^2}$$

  > [!IMPORTANT]
  > **Inverse-Covariance Fusion & Validation Disclaimer**:
  > - Observations with smaller uncertainty (higher precision $\mathbf{\Sigma}_i^{-1}$) exert proportionally higher weight on the estimated landmark location.
  > - Singletons ($N=1$) naturally yield landmark position $\hat{\mathbf{p}} = \mathbf{p}_1$ and landmark covariance $\mathbf{\Sigma}_{\text{landmark}} = \mathbf{\Sigma}_1$.
  > - Estimated landmark positions represent **consensus-supported landmark estimates**, NOT ground-truth accuracy validation (no ground-truth cone identities exist).

- **Verified Measured Dataset Results ($72$ Consensus-Supported Landmarks)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Consensus-Supported Landmarks** | `72` |
  | **Landmark X Bounds [m]** | `[-43.4996, 43.5023]` |
  | **Landmark Y Bounds [m]** | `[-28.5472, 28.5188]` |
  | **Mean $\sigma_{\text{landmark}, x}$** | `0.012521 m` |
  | **Median $\sigma_{\text{landmark}, x}$** | `0.012616 m` |
  | **Max $\sigma_{\text{landmark}, x}$** | `0.015781 m` |
  | **Mean $\sigma_{\text{landmark}, y}$** | `0.014159 m` |
  | **Median $\sigma_{\text{landmark}, y}$** | `0.014959 m` |
  | **Max $\sigma_{\text{landmark}, y}$** | `0.018414 m` |
  | **Mean Weighted Centroid Shift** | `0.010684 m` |
  | **Median Weighted Centroid Shift** | `0.008183 m` |
  | **Max Weighted Centroid Shift** | `0.036731 m` |

- **Inter-Landmark Distance Diagnostics**:
  | Distance Metric | Measured Value |
  | :--- | :---: |
  | **Minimum Inter-Landmark Distance** | `3.7348 m` |
  | **Median Inter-Landmark Distance** | `44.6280 m` |
  | **Maximum Inter-Landmark Distance** | `87.0019 m` |

- **Automated Tests**: 7 comprehensive unit tests implemented in `tests/test_landmark.py`.

---

### **Module 9: Track Boundary Reconstruction (`src/track_recon.py`) - IMPLEMENTED**
- **Purpose**: Reconstruct left (blue) and right (yellow) track boundary splines and optional track centerline from EWMVC consensus-supported landmark estimates and vehicle telemetry trajectory poses without hallucinating non-existent cones.
- **Inputs**: `outputs/ewmvc_landmarks.csv` (`candidate_id`, `landmark_x`, `landmark_y`, `consensus_score`) and `data/raw/telemetry_log.csv` (`timestamp`, `x`, `y`, `yaw_rad`).
- **Outputs**:
  - `outputs/ewmvc_track_landmarks.csv` (`candidate_id`, `landmark_x`, `landmark_y`, `consensus_score`, `s_m`, `l_m`, `side`, `trajectory_timestamp`).
  - `outputs/ewmvc_left_boundary.csv` (`s_m`, `x_m`, `y_m`).
  - `outputs/ewmvc_right_boundary.csv` (`s_m`, `x_m`, `y_m`).
  - `outputs/ewmvc_centerline.csv` (`s_m`, `x_m`, `y_m`).
  - `outputs/ewmvc_track_reconstruction.png` (Visual plot titled `"EWMVC Track Boundary Reconstruction"`).
  - `outputs/ewmvc_track_coordinates.png` (Scatter plot of track coordinates $s_{\text{m}}$ vs $l_{\text{m}}$).

- **Mathematical Formulations**:
  1. **Trajectory-Relative Coordinates**:
     For each landmark $\mathbf{p}_l = [x_l, y_l]^T$, find the closest pose on the vehicle trajectory $\mathbf{p}_v = [x_v, y_v]^T$ with heading $\psi_{\text{veh}}$:
     $$\mathbf{r} = \begin{bmatrix} x_l - x_v \\ y_l - y_v \end{bmatrix}$$
     $$s = r_x \cos(\psi_{\text{veh}}) + r_y \sin(\psi_{\text{veh}})$$
     $$l = -r_x \sin(\psi_{\text{veh}}) + r_y \cos(\psi_{\text{veh}})$$
     - $s > 0$: Landmark is ahead of vehicle at closest pose.
     - $s < 0$: Landmark is behind vehicle at closest pose.
     - $l > 0$: Landmark is on vehicle's left side.
     - $l < 0$: Landmark is on vehicle's right side.

  2. **Side Classification Rule**:
     - $|l| < d_{\text{min\_lateral}}$ ($0.5\text{ m}$): Classified as `ambiguous`.
     - $l \ge +d_{\text{min\_lateral}}$: Classified as `left`.
     - $l \le -d_{\text{min\_lateral}}$: Classified as `right`.

  3. **No-Extrapolation Parametric Spline Fitting**:
     - For 0 landmarks: Returns empty DataFrame.
     - For 1 landmark: Returns single point DataFrame.
     - For 2 landmarks: Fits linear interpolation between $s_{\min}$ and $s_{\max}$.
     - For $\ge 3$ landmarks: Fits parametric cubic splines $x(s)$ and $y(s)$ using `scipy.interpolate.CubicSpline` evaluated over $N_{\text{samples}}=200$ uniform points within $[s_{\min}, s_{\max}]$.

  4. **Centerline Calculation**:
     Calculates centerline $\mathbf{p}_{\text{center}}(s) = \frac{1}{2} \left(\mathbf{p}_{\text{left}}(s) + \mathbf{p}_{\text{right}}(s)\right)$ evaluated strictly over the common longitudinal range $[\max(s_{\text{left,min}}, s_{\text{right,min}}), \min(s_{\text{left,max}}, s_{\text{right,max}})]$.

  > [!IMPORTANT]
  > **Strict No-Hallucination Rule & Configured Design Parameters Disclaimer**:
  > - Splines represent continuous spatial approximations of the track boundaries through available landmark estimates; they do NOT generate or hallucinate synthetic cone detections.
  > - All filtering bounds (`min_forward_distance_m=0.0`, `max_forward_distance_m=60.0`, `min_lateral_distance_m=0.5`, `max_lateral_distance_m=15.0`) are **configured design parameters** in `configs/config.yaml`, NOT empirically tuned metrics using labels.
  > - Ground-truth labels (`real`/`ghost`) are strictly NOT used.

- **Verified Measured Dataset Results ($72$ Consensus-Supported Landmarks)**:
  | Metric | Measured Value |
  | :--- | :---: |
  | **Total EWMVC Landmarks Loaded** | `72` |
  | **Landmarks Used for Reconstruction** | `72` |
  | **Left Track Landmarks ($l > 0$)** | `36` |
  | **Right Track Landmarks ($l < 0$)** | `36` |
  | **Ambiguous Landmarks ($|l| < 0.5\text{ m}$)** | `0` |
  | **Left Boundary Longitudinal Range** | `[0.0463 m, 202.1126 m]` |
  | **Right Boundary Longitudinal Range** | `[-0.0050 m, 202.8488 m]` |
  | **Common Centerline Longitudinal Range** | `[0.0463 m, 202.1126 m]` |

- **Automated Tests**: 10 comprehensive unit tests implemented in `tests/test_track_recon.py`.


---

## 6. Proposed Configuration Extensions (`configs/config.yaml`)

```yaml
# Existing Baseline Configuration (Preserved)
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

# PROPOSED EWMVC CONFIGURATION EXTENSIONS (Design Proposals)
ewmvc:
  uncertainty:
    sensor_range_var_base: 0.0025    # (0.05 m)^2
    sensor_range_var_scale: 0.02
    sensor_bearing_var: 0.000685     # (1.5 deg in rad)^2
    vehicle_pos_var: 0.04            # (0.2 m)^2
    vehicle_heading_var: 0.000305    # (1.0 deg in rad)^2

  graph:
    chi2_threshold: 5.991            # Chi-square 2-DOF, alpha=0.95
    max_spatial_distance_m: 1.5

  entropy:
    num_angular_bins: 8              # 45 deg per bin

  geometry:
    reference_radius_m: 0.15

  persistence:
    target_frame_count: 10
    time_scale_sec: 1.0

  diversity:
    target_baseline_displacement_m: 5.0

  consensus:
    weights:
      entropy: 0.30
      geometry: 0.30
      diversity: 0.20
      persistence: 0.20
    acceptance_threshold: 0.45
```

---

## 7. Baseline Preservation & Execution Commands

The existing DBSCAN baseline pipeline and test suites remain **100% untouched**:

```bash
# 1. Run Fixed-Epsilon DBSCAN Baseline
python main.py

# 2. Run DBSCAN Parameter Sensitivity Grid (24 Configs)
python -m src.parameter_experiments

# 3. Run Baseline Failure Analysis (Surviving Ghosts)
python -m src.failure_analysis

# 4. Run Automated Pytest Regression Suite (24/24 Passed)
python -m pytest -v
```

---

## 8. Verification Confirmation

- **Architecture Document Created**: `docs/EWMVC_ARCHITECTURE.md`
- **Modules & Interfaces Designed**: 9 future modules specified with input/output, math, configuration, and failure cases.
- **Baseline Preserved**: All 24 pytest tests pass cleanly; `main.py` baseline output remains identical ($72$ clusters, $82.50\%$ ghost rejection rate).
- **EWMVC Algorithm Implementation**: **ZERO EWMVC algorithms were implemented in Python during this step.**
