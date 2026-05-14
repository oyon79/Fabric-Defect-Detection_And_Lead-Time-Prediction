# 🎨 Unified Fabric Defect Detection & Lead Time Prediction Framework

[![Framework](https://img.shields.io/badge/Framework-YOLO%2B%20Random%20Forest-blue.svg)](https://github.com/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research%20Purpose-purple.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Enabled-orange.svg)](https://developer.nvidia.com/cuda-zone)

> **Smart Manufacturing Integration**: Closing the gap between computer vision defect detection and production planning systems through real-time Defect Density Score (D_s) feedback.

---

## 🔗 Download Full Trained Project

> **⚠️ IMPORTANT**: The GitHub repository contains only the lightweight source code. Large files (trained models, datasets, YOLO weights) are hosted on Google Drive to keep the repo small and clone-friendly.

### 📦 Complete Project Package

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download%20Full%20Project-34A853?style=for-the-badge&logo=google-drive)](https://drive.google.com/drive/folders/1vCdl2v5ODnb-S5Mi_Cfh2Nv1rqKaRU5G?usp=drive_link)

**Google Drive contains everything you need:**
- ✅ Trained YOLO classifier weights (`best.pt`)
- ✅ Complete Fabric Defects Dataset (all defect classes)
- ✅ Pre-trained YOLO model weights (`yolo11n.pt`, `yolov8n.pt`)
- ✅ Training outputs (`runs/classify/train/`)
- ✅ All trained assets and configuration files
- ✅ Complete project with all dependencies resolved

---

## 🚀 Setup Instructions

### Option A: Lightweight GitHub Version (Source Code Only)

For developers who want to **train their own models** or integrate the framework:

```bash
# Clone the repository
git clone https://github.com/oyon79/Fabric-Defect-Detection_And_Lead-Time-Prediction.git
cd Fabric-Defect-Detection_And_Lead-Time-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the framework
python integration.py
```

> **Note:** You will need to train your own YOLO classifier and prepare your dataset.

---

### Option B: Full Trained Project from Google Drive (Recommended)

For **immediate testing** with pre-trained models and complete dataset:

1. **Download from Google Drive** using the button above
2. **Extract** the downloaded folder
3. **Copy** all contents to your project directory
4. **Run:**

```bash
cd Fabric-Defect-Detection_And_Lead-Time-Prediction
pip install -r requirements.txt
python integration.py
```

---

## Project Overview

### What This Framework Does

This framework creates a **closed feedback loop** between quality inspection and production planning:

1. A camera captures images of fabric on a production line
2. A YOLO-based detector identifies defects in real-time
3. The Metadata Bridge converts detection results into a single **Defect Density Score (D_s)**
4. The Lead Time Predictor uses D_s to dynamically adjust production estimates
5. Planners receive accurate, real-time lead time predictions

### Key Innovation

Traditional manufacturing separates the vision system (defect detection) from the planning system (lead time estimation). This creates an **integration gap** where planners make decisions based on historical averages rather than actual factory conditions. This framework eliminates that gap by feeding real-time defect data directly into the prediction model.

---

## Problem Statement

### The Integration Gap in Smart Manufacturing

In traditional smart manufacturing environments:

```
┌─────────────┐      ┌─────────────┐
│   Vision     │      │  Production │
│   System     │      │  Planning   │
│ (Defect      │      │  System     │
│  Detection)  │  X   │ (Lead Time  │
│              │      │  Estimation)│
└─────────────┘      └─────────────┘
     ↑                    ↑
     │                    │
     │    Integration     │
     │       GAP          │
     └────────────────────┘
```

**Problems caused by this gap:**
- Planners assume constant defect rates (ignoring quality variations)
- Lead time estimates become inaccurate when fabric quality changes
- No real-time feedback mechanism between quality and planning
- Reactive rather than proactive production scheduling

### Impact

When defect rates vary (e.g., 5% to 20%), but planners use a fixed assumption (e.g., 10%), results are:
- **Under-prediction**: Orders take longer than estimated (missed deadlines)
- **Over-prediction**: Excess buffer time allocated (inefficient resource use)

---

## Solution Architecture

### Closing the Loop

This framework implements a continuous feedback architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED FRAMEWORK                             │
│                                                                     │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│   │   Camera /   │───▶│ defect_detection │───▶│ metadata_bridge │  │
│   │   Images     │    │     (YOLO)       │    │  (calculates    │  │
│   └──────────────┘    └──────────────────┘    │    D_s)         │  │
│                                               └────────┬────────┘  │
│                                                        │           │
│                                                        ▼           │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│   │  Production  │◀───│lead_time_predictor│◀───│  D_s value      │  │
│   │  Planner     │    │ (adjusts estimate)│    │  (real-time)    │  │
│   └──────────────┘    └──────────────────┘    └─────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Function |
|-----------|----------|
| **defect_detection.py** | YOLO-based classifier detects defect types in images |
| **metadata_bridge.py** | Converts raw detections into a scalar D_s metric |
| **lead_time_predictor.py** | Random Forest model predicts lead time from D_s |
| **integration.py** | Orchestrates the complete pipeline |
| **config.py** | Centralized hyperparameters |

---

## File Structure

```
my-project/
├── config.py                      # All hyperparameters (do not edit inline)
├── defect_detection.py            # YOLO defect detection + industrial preprocessing
├── metadata_bridge.py             # Defect Density Score (D_s) calculation
├── lead_time_predictor.py         # Lead time prediction model
├── integration.py                 # Main orchestration script
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # (Optional) YOLOv8 nano weights
├── yolo11n.pt                     # (Optional) YOLOv11 nano weights
├── runs/
│   └── classify/
│       └── train/
│           └── weights/
│               └── best.pt        # Your trained classifier (train separately)
└── Fabric Defects Dataset/        # Training data (structure varies by use)
    ├── defect free/
    ├── hole/
    ├── horizontal/
    ├── vertical/
    ├── lines/
    └── stain/
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` specifies:
```
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

### Train Your Classifier (Required for Full Detection)

This framework expects a trained YOLO classifier. To train on your fabric defect dataset:

1. **Prepare your dataset** in YOLO classification format:
   ```
   Fabric Defects Dataset/
   ├── class_1/
   │   ├── image001.jpg
   │   ├── image002.jpg
   │   └── ...
   ├── class_2/
   │   └── ...
   ```

2. **Train the model**:
   ```bash
   cd "Fabric Defects Dataset"
   yolo train model=yolo11n.pt data=. --epochs=50 imgsz=640
   ```

3. **Update the path** in `defect_detection.py`:
   ```python
   CLASSIFIER_PATH = "runs/classify/train/weights/best.pt"
   ```

### Dataset Structure Reference

The framework recognizes these fabric defect classes:
- `Vertical` - Vertical defects
- `defect free` - No defect
- `hole` - Hole/puncture defects
- `horizontal` - Horizontal defects
- `lines` - Line pattern defects
- `stain` - Stain/contamination defects

---

## Component Details

### config.py

Centralized configuration for all hyperparameters. **Do not modify constants inline** - edit `config.py` for system-wide changes.

#### Industrial Preprocessing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BLUR_KERNEL_SIZE` | 15 | Motion blur kernel size in pixels |
| `NOISE_VARIANCE` | 0.01 | Gaussian sensor noise variance |
| `LOW_LIGHT_FACTOR` | 0.7 | Gamma correction and darkening factor |

#### Integration Pipeline Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PREDICTION_INTERVAL` | 30 | Update lead time prediction every N frames |
| `RANDOM_STATE` | 42 | Seed for reproducibility across all components |

#### YOLO Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_MODEL` | "yolo11n.pt" | Model file to use (n=small, s=medium, m=large) |
| `CONFIDENCE_THRESHOLD` | 0.25 | Minimum confidence for detection |
| `IOU_THRESHOLD` | 0.45 | Intersection-over-Union threshold for NMS |

#### ML Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYNTHETIC_SAMPLES` | 1000 | Number of synthetic samples for training |
| `TRAIN_TEST_SPLIT` | 0.8 | Training set proportion (80% train, 20% test) |

---

### defect_detection.py

Contains the hybrid defect detection pipeline combining YOLO classification with image processing for area estimation.

#### IndustrialPreprocessor

Simulates hostile factory conditions to ensure robust detection in real environments:

```
Input Frame → Gaussian Noise → Motion Blur → Low Light Simulation → Output Frame
```

**Why this matters:** Real factories have poor lighting, electrical interference, and fast-moving fabric. Without preprocessing simulation, models trained on clean data fail in production.

**Methods:**
- `add_gaussian_noise(frame)` - Adds random sensor/electrical noise
- `apply_motion_blur(frame)` - Simulates blur from fast-moving fabric (configurable angle)
- `simulate_low_light(frame)` - Darkens frames using gamma correction
- `apply(frame)` - Runs all three in sequence

#### DefectAreaEstimator

Estimates defect bounding box area using classifier confidence as a guide:

**Base defect sizes (as fraction of frame):**
| Defect Type | Base Size | Rationale |
|-------------|-----------|-----------|
| hole | 6% | Small, concentrated |
| horizontal | 12% | Wide, thin |
| vertical | 8% | Tall, thin |
| lines | 20% | Multiple line patterns |
| stain | 15% | Irregular stain areas |

**Area scaling logic:**
- High classifier confidence (0.9+) = full base size
- Medium confidence (0.6-0.9) = 70-90% of base
- Low confidence (0.3-0.6) = 40-70% of base
- Very low (<0.3) = 15% of base

#### FabricDefectDetector

The main detection class that wraps YOLO classification with disambiguation logic.

**Detection Flow:**
```
Input Frame
    ↓
[Optional] Apply Industrial Preprocessing
    ↓
Run YOLO Classifier
    ↓
Get Top-1 Prediction + Confidence
    ↓
[Apply] Disambiguation Rules (image statistics)
    ↓
[If "defect free" or conf < 0.3] → Return []
    ↓
Estimate Bounding Box Area
    ↓
Return Detection Dict
```

**Detection Output Format:**
```python
{
    'class_id': int,           # Numeric class index
    'class_name': str,         # "hole", "stain", etc.
    'confidence': float,       # Combined confidence (0-1)
    'bbox': (x1, y1, x2, y2),  # Bounding box coordinates
    'area_ratio': float,       # Defect area / frame area
    'type_confidence': float,  # YOLO classifier confidence
    'area_confidence': float   # Area estimation confidence
}
```

**Disambiguation Rules:**

The classifier sometimes confuses similar defect types. This module corrects common errors using image statistics:

| Rule | Trigger | Correction |
|------|---------|------------|
| 1 | Classifier says "horizontal" + image is bright (max≥80) + low texture (std<3) | → "hole" |
| 2 | Classifier says "hole" + image is dark (max<50) + low texture (std<2.5) | → "horizontal" |
| 3 | Classifier says "vertical" + low confidence (<0.55) + bright + low texture | → "hole" |

---

### metadata_bridge.py

Converts raw YOLO detections into the **Defect Density Score (D_s)** - a single scalar metric that represents fabric quality.

#### The D_s Formula

```
D_s = Σ (Defect Bounding Box Areas) / Total Frame Area
```

**Interpretation:**
| D_s Value | Meaning |
|-----------|---------|
| 0.0 | No defects detected |
| 0.05 | 5% of frame covered by defects |
| 0.10 | 10% of frame covered by defects |
| 0.50+ | Severe defect coverage |

**Example Calculation:**
```
Frame: 1920 × 1080 = 2,073,600 pixels
Defect 1 bbox: (100,100) to (300,300) = 40,000 pixels
Defect 2 bbox: (500,400) to (700,600) = 40,000 pixels
Defect 3 bbox: (1000,800) to (1150,950) = 22,500 pixels

Total defect area: 102,500 pixels
D_s = 102,500 / 2,073,600 ≈ 0.0494 (4.94%)
```

#### Rolling Density

Single-frame D_s values can be noisy. The rolling average reduces fluctuation:

```
Rolling D_s = mean(D_s[frame-N : frame])  for last N frames
```

Default window: 10 frames (configurable via `rolling_window` parameter)

#### MetadataBridge Class API

```python
bridge = MetadataBridge(rolling_window=10)

# Process new detections
density = bridge.update(detections, frame_shape)

# Query methods
current_ds = bridge.get_current_ds()      # Most recent value
rolling_ds = bridge.get_rolling_ds()     # Smoothed average
stats = bridge.get_stats()               # Full statistics dict
```

**Stats Dictionary:**
```python
{
    'current_ds': float,           # Latest D_s value
    'rolling_ds': float,          # Smoothed D_s
    'mean_ds': float,              # Historical mean
    'max_ds': float,               # Historical maximum
    'total_detections': int,       # Total defects found
    'frames_processed': int       # Total frames analyzed
}
```

---

### lead_time_predictor.py

Random Forest regression model that predicts manufacturing lead time from machine parameters and defect density.

#### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `machine_speed` | float | 20-100 m/min | Production line speed |
| `order_quantity` | int | 100-10000 units | Number of units in order |
| `defect_density` | float | 0.0-0.5+ | Current D_s value |

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `lead_time` | float | Predicted time to complete order (hours) |

#### The Learning Relationship

```
Lead Time = (order_quantity / machine_speed) × 0.1 × (1 + defect_density × 2)
```

**Logic:**
- **Higher D_s** → Slower effective production → Longer lead time
- **More units** → Proportionally more time needed
- **Faster machines** → Less time per unit

**Example:**
| Scenario | Speed | Qty | D_s | Lead Time |
|----------|-------|-----|-----|-----------|
| Good quality | 80 m/min | 5000 | 0.05 | ~3.4 hours |
| Bad quality | 80 m/min | 5000 | 0.30 | ~4.8 hours |
| Good quality, big order | 80 m/min | 10000 | 0.05 | ~6.8 hours |

#### LeadTimePredictor API

```python
# Initialize
predictor = LeadTimePredictor(n_estimators=100, max_depth=10)

# Train
predictor.fit(X_train, y_train)  # X: (n_samples, 3), y: (n_samples,)

# Predict single scenario
lead_time = predictor.predict_single(
    machine_speed=60.0,
    order_quantity=2000,
    defect_density=0.15
)

# Predict multiple
lead_times = predictor.predict(X)  # X: (n_samples, 3)

# Update with new data
predictor.update(X_new, y_new)
```

#### Synthetic Data Generation

When historical manufacturing data isn't available, `generate_synthetic_data()` creates realistic training data:

```python
X_train, X_test, y_train, y_test = generate_synthetic_data(
    n_samples=1000,
    random_state=42
)
```

**Data ranges:**
- Machine speeds: Uniform(20, 100) m/min
- Order quantities: Uniform(100, 10000) units
- Defect densities: Uniform(0, 0.5)
- Lead times: Physics-based formula + Gaussian noise

#### Model Evaluation

```python
metrics = evaluate_model(predictor, X_test, y_test)

# Returns:
# {
#     'mae': float,           # Mean Absolute Error (hours)
#     'rmse': float,          # Root Mean Squared Error (hours)
#     'r2': float,            # R² score (0-1)
#     'baseline_mae': float, # Static baseline MAE
#     'mae_improvement': float  # Improvement over baseline
# }
```

**Expected performance:**
- MAE improvement over baseline: 10-30% when D_s varies significantly
- R² score typically 0.85-0.95 on synthetic test data

---

### integration.py

The main orchestration script that ties all components together into a complete real-time processing pipeline.

#### LiveStreamProcessor

Processes a folder of images as if they were frames from a live camera stream.

**Initialization:**
```python
processor = LiveStreamProcessor(
    image_folder="path/to/images",
    machine_speed=60.0,         # m/min
    order_quantity=2000,        # units
    detection_interval=30      # update prediction every N frames
)
```

**Internal setup:**
1. Loads YOLO detector
2. Creates IndustrialPreprocessor
3. Initializes MetadataBridge with rolling window
4. Trains LeadTimePredictor on synthetic data

**Processing Loop (`run()`):**
```
For each image in folder:
    1. Read image as frame
    2. Process through detection pipeline
    3. Update MetadataBridge with detections
    4. Every N frames (detection_interval):
       - Calculate rolling D_s
       - Update lead time prediction
       - Display/print status
    5. [Optional] Show cv2 window with detections
```

**Return value (stats dict):**
```python
{
    'total_frames': int,        # Total images processed
    'detections_total': int,    # Total defects found
    'avg_fps': float            # Average processing FPS
}
```

#### run_benchmark()

Compares the unified model against a static baseline:

| Model | Uses D_s | Behavior |
|-------|----------|----------|
| Unified Model | Yes | Dynamically adjusts prediction as D_s changes |
| Static Baseline | No | Always predicts the mean of training data |

**Benchmark Output:**
```
Unified Model (with real-time D_s):
  MAE:  0.XXX hours
  RMSE: 0.XXX hours
  R2:   0.XXXX

Static Baseline (ignores D_s):
  MAE:  0.XXX hours
  RMSE: 0.XXX hours
  R2:   0.XXXX

IMPROVEMENT FROM INTEGRATION
MAE Reduction:  X.XXX hours
MAE Improvement: XX.X%
```

---

## How It Works

### End-to-End Flow

```
1. IMAGE CAPTURE
   Camera or image folder contains frames of fabric on production line

2. DEFECT DETECTION (defect_detection.py)
   - YOLO classifier identifies defect type
   - Disambiguation rules correct classifier errors
   - Area estimator calculates bounding box size
   - Returns list of detections with class, confidence, bbox

3. DENSITY CALCULATION (metadata_bridge.py)
   - Sum all defect bounding box areas
   - Divide by total frame area
   - D_s = Total Defect Area / Frame Area

4. LEAD TIME PREDICTION (lead_time_predictor.py)
   - Feed D_s + machine_speed + order_quantity to Random Forest
   - Model outputs predicted lead time in hours
   - Higher D_s → Higher predicted lead time

5. DECISION SUPPORT
   - Planners see updated prediction in real-time
   - Production schedules adjust automatically
   - Quality deviations are reflected immediately
```

### The Feedback Loop

```
Fabric Quality ↓    →    D_s ↑    →    Lead Time Prediction ↑
     ↑                                                       │
     │                                                       │
     └───────────────── (Closed Loop) ──────────────────────┘
```

---

## The Integration Loop

```
                    ┌─────────────────┐
                    │  Production     │
                    │  Planner        │
                    │ (Sees lead time)│
                    └────────┬────────┘
                             │
                             ▼
┌──────────┐         ┌─────────────────┐         ┌──────────────────┐
│  Camera  │────────▶│ defect_detection │────────▶│ metadata_bridge  │
│ (Images) │         │     (YOLO)       │         │  (calculates D_s)│
└──────────┘         └─────────────────┘         └────────┬─────────┘
                                                             │
                                                             │ D_s value
                                                             ▼
                                                  ┌──────────────────┐
                                                  │lead_time_predictor│
                                                  │ (adjusts estimate) │
                                                  └──────────────────┘
```

**Traditional approach:**
- Planner estimates 10 hours
- Quality varies (5% → 20% defects)
- Actual time: 15 hours
- Result: Missed deadline

**This framework:**
- Planner estimates 10 hours at 5% defects
- As D_s rises (10% → 15% → 20%)
- Prediction adjusts: 10h → 12h → 15h
- Result: Accurate scheduling, proactive response

---

## Configuration Reference

### Quick Configuration Guide

To adjust system behavior, modify `config.py`:

#### For Faster Detection

```python
# Use smaller model
YOLO_MODEL = "yolo11n.pt"  # nano (fastest)

# Or for even faster
YOLO_MODEL = "yolov8n.pt"  # very small
```

#### For More Responsive Predictions

```python
# Update prediction more frequently
PREDICTION_INTERVAL = 15  # Every 15 frames instead of 30

# Use shorter rolling window for less smoothing
# (In MetadataBridge initialization)
bridge = MetadataBridge(rolling_window=5)
```

#### For More Accurate Predictions

```python
# Use larger model
YOLO_MODEL = "yolo11s.pt"  # small (better accuracy)

# More trees in Random Forest
# (In LeadTimePredictor initialization)
predictor = LeadTimePredictor(n_estimators=200, max_depth=15)
```

---

## Usage

### Run the Complete Framework

```bash
python integration.py
```

**What happens:**
1. Generates synthetic training data
2. Trains the lead time predictor
3. Runs benchmark comparing unified model vs static baseline
4. Processes images from available folder (or synthetic demo frames)
5. Displays real-time D_s and lead time updates

### Run Individual Components

```bash
# Test defect detection only
python defect_detection.py

# Test lead time prediction only
python lead_time_predictor.py

# Test metadata bridge only
python metadata_bridge.py
```

### Custom Image Processing

```python
from integration import LiveStreamProcessor

# Process your own images
processor = LiveStreamProcessor(
    image_folder="path/to/your/fabric/images",
    machine_speed=60.0,      # m/min
    order_quantity=2000,    # units
    detection_interval=30   # update every 30 frames
)

# Run with display
stats = processor.run(display=True, verbose=True)

# Run without display (headless)
stats = processor.run(display=False, verbose=True)

print(f"Processed {stats['total_frames']} frames at {stats['avg_fps']:.1f} FPS")
```

### Programmatic Usage

```python
# Full pipeline usage
from defect_detection import FabricDefectDetector
from metadata_bridge import MetadataBridge
from lead_time_predictor import LeadTimePredictor, generate_synthetic_data

# Setup
detector = FabricDefectDetector()
bridge = MetadataBridge(rolling_window=10)
X_train, X_test, y_train, y_test = generate_synthetic_data()
predictor = LeadTimePredictor()
predictor.fit(X_train, y_train)

# Process single frame
frame = cv2.imread("fabric_image.jpg")
detections = detector.detect(frame, apply_preprocessing=True)
ds = bridge.update(detections, frame.shape)
rolling_ds = bridge.get_rolling_ds()

# Predict lead time
lead_time = predictor.predict_single(
    machine_speed=60.0,
    order_quantity=2000,
    defect_density=rolling_ds
)
print(f"Current D_s: {ds:.4f}, Predicted Lead Time: {lead_time:.2f}h")
```

---

## Key Concepts

| Concept | Definition |
|---------|------------|
| **D_s (Defect Density Score)** | Percentage of frame area covered by defect bounding boxes (0.0 to 1.0+) |
| **Rolling Window** | Average of last N D_s values to reduce single-frame noise |
| **Lead Time** | Hours needed to complete an order |
| **Static Baseline** | Always predicts the same value (mean of training data), ignores D_s |
| **Industrial Preprocessing** | Simulates factory conditions: motion blur, sensor noise, low light |
| **YOLO** | You Only Look Once - real-time object detection model |
| **Integration Gap** | Information loss when vision and planning systems don't share data |
| **Closed Feedback Loop** | System that continuously adjusts based on its own outputs |

---

## Performance Metrics

### Expected Performance

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| Detection FPS | 20-60 FPS | Depends on hardware and image size |
| MAE Improvement | 10-30% | vs static baseline when D_s varies |
| R² Score | 0.85-0.95 | On synthetic test data |
| D_s Range | 0.0-1.0+ | Clamped to 1.0 for display |

### Factors Affecting Performance

**Positive impact:**
- High variation in defect density across samples
- Accurate classifier training
- Appropriate rolling window size

**Negative impact:**
- Very low defect rates (D_s ≈ 0 always)
- Noisy image data
- Incorrect machine speed or order quantity inputs

---

## Troubleshooting

### Common Issues

**1. "No module named 'ultralytics'"**

```bash
pip install -r requirements.txt
```

**2. "Classifier path does not exist"**

Train your YOLO classifier on the fabric defect dataset:
```bash
cd "Fabric Defects Dataset"
yolo train model=yolo11n.pt data=. --epochs=50
```
Then update `CLASSIFIER_PATH` in `defect_detection.py`.

**3. "CUDA out of memory"**

Reduce image size or use smaller YOLO model:
```python
# In config.py
YOLO_MODEL = "yolo11n.pt"  # instead of yolo11s.pt or larger
```

**4. Low detection accuracy**

- Ensure classifier is trained on your specific fabric types
- Adjust `CONFIDENCE_THRESHOLD` in `config.py`
- Check that `FABRIC_CLASSES` mapping matches your dataset

**5. Lead time predictions seem wrong**

- Verify `machine_speed` and `order_quantity` are realistic values
- Check that D_s is in expected range (0.0-0.5 typically)
- Consider adjusting rolling window if predictions are too noisy

### Debug Mode

Enable verbose output to see detailed processing:

```python
processor = LiveStreamProcessor(
    image_folder="path/to/images",
    display=True,
    verbose=True  # Enable detailed logging
)
stats = processor.run()
```

---

## Training Your Own YOLO Model

To use your own fabric defect dataset:

1. **Structure your data** as YOLO classification dataset:
   ```
   dataset/
   ├── class_hole/
   │   ├── img001.jpg
   │   └── ...
   ├── class_stain/
   │   └── ...
   ```

2. **Train** (from dataset parent directory):
   ```bash
   yolo train model=yolo11n.pt data=. --epochs=100 imgsz=640
   ```

3. **Find weights** at:
   ```
   runs/classify/train/weights/best.pt
   ```

4. **Update path** in `defect_detection.py`:
   ```python
   CLASSIFIER_PATH = "runs/classify/train/weights/best.pt"
   ```

5. **Verify class names** match the `FABRIC_CLASSES` dictionary (lines 19-26)

---

## License

This framework is provided as-is for educational and manufacturing research purposes.

---

## References

- YOLO: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Scikit-learn: [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- OpenCV: [Computer Vision Tools](https://opencv.org/)