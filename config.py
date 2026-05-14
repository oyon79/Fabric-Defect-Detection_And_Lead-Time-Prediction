# Hyperparameters for Unified Fabric Defect Detection & Lead Time Prediction Framework

# Industrial Preprocessing
BLUR_KERNEL_SIZE = 15
NOISE_VARIANCE = 0.01
LOW_LIGHT_FACTOR = 0.7

# Integration Pipeline
PREDICTION_INTERVAL = 30  # Update prediction every N frames

# ML Model
RANDOM_STATE = 42

# YOLO Configuration
YOLO_MODEL = "yolo11n.pt"  # YOLOv11 nano variant for speed; use yolo11s/m/l for accuracy
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Synthetic Data Generation
SYNTHETIC_SAMPLES = 1000
TRAIN_TEST_SPLIT = 0.8