"""
Task 4: Integration & Evaluation
===============================
Unified framework that processes a live image stream, detects defects,
calculates D_s in real-time, and updates lead time predictions.

The Integration Gap Addressed:
------------------------------
Traditional smart manufacturing separates the vision system (defect detection)
from the production planning system (lead time estimation). This creates a
information gap where planners make decisions based on estimated defect rates
rather than actual observed quality.

This framework closes the loop by:
1. Continuously monitoring fabric quality via YOLO detection
2. Translating detections into a scalar D_s metric
3. Using D_s to dynamically adjust lead time predictions
4. Enabling proactive production scheduling based on real-time quality data

The result: production plans reflect actual factory conditions, not assumptions.
"""

import os
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional

from defect_detection import FabricDefectDetector, IndustrialPreprocessor
from metadata_bridge import MetadataBridge, extract_defect_density
from lead_time_predictor import LeadTimePredictor, generate_synthetic_data, evaluate_model
from config import PREDICTION_INTERVAL, RANDOM_STATE


class LiveStreamProcessor:
    """
    Processes image folder as a video stream simulation.

    This processor mimics live video processing by iterating through images
    at a configurable frame rate, running detection, and updating predictions.
    """

    def __init__(
        self,
        image_folder: str,
        machine_speed: float = 50.0,
        order_quantity: int = 1000,
        detection_interval: int = PREDICTION_INTERVAL
    ):
        """
        Initialize the live stream processor.

        Args:
            image_folder: Path to folder containing images to process
            machine_speed: Current machine speed (m/min)
            order_quantity: Current order quantity (units)
            detection_interval: Update prediction every N frames
        """
        self.image_folder = Path(image_folder)
        self.machine_speed = machine_speed
        self.order_quantity = order_quantity
        self.detection_interval = detection_interval

        # Initialize components
        self.detector = FabricDefectDetector()
        self.preprocessor = IndustrialPreprocessor()
        self.metadata_bridge = MetadataBridge(rolling_window=10)

        # Lead time predictor (trained on synthetic data)
        X_train, X_test, y_train, y_test = generate_synthetic_data()
        self.predictor = LeadTimePredictor()
        self.predictor.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

        # State tracking
        self.frame_count = 0
        self.current_ds = 0.0
        self.current_lead_time = None
        self.processing_stats = {
            'total_frames': 0,
            'detections_total': 0,
            'avg_fps': 0.0
        }

        # Find image files
        self.image_files = self._collect_images()

    def _collect_images(self) -> list[Path]:
        """Collect all supported image files from folder."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        if self.image_folder.exists():
            for f in self.image_folder.iterdir():
                if f.suffix.lower() in extensions:
                    images.append(f)
        return sorted(images)

    def process_frame(self, frame: np.ndarray) -> tuple[list, float]:
        """
        Process a single frame through the detection pipeline.

        Args:
            frame: BGR image

        Returns:
            (detections, defect_density)
        """
        detections = self.detector.detect(frame, apply_preprocessing=True)
        ds = self.metadata_bridge.update(detections, frame.shape)
        return detections, ds

    def update_prediction(self) -> float:
        """
        Update lead time prediction based on current D_s.

        Returns:
            Predicted lead time in hours
        """
        rolling_ds = self.metadata_bridge.get_rolling_ds()
        self.current_lead_time = self.predictor.predict_single(
            self.machine_speed,
            self.order_quantity,
            rolling_ds
        )
        return self.current_lead_time

    def run(self, display: bool = True, verbose: bool = True) -> dict:
        """
        Run the processing loop over all images.

        Args:
            display: Show live preview window
            verbose: Print status updates

        Returns:
            Final statistics dictionary
        """
        if not self.image_files:
            print(f"No images found in {self.image_folder}")
            return self.processing_stats

        print(f"Processing {len(self.image_files)} images...")
        print(f"Detection interval: every {self.detection_interval} frames")
        print("-" * 50)

        start_time = time.time()
        fps = 0.0

        for i, img_path in enumerate(self.image_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            self.frame_count += 1
            self.processing_stats['total_frames'] = self.frame_count

            # Process frame through detection pipeline
            detections, ds = self.process_frame(frame)
            self.current_ds = ds
            self.processing_stats['detections_total'] += len(detections)

            # Update prediction at configured interval
            if self.frame_count % self.detection_interval == 0:
                lead_time = self.update_prediction()
                rolling_ds = self.metadata_bridge.get_rolling_ds()

                if verbose:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frame {self.frame_count:3d} | D_s: {ds:.4f} | "
                          f"Rolling D_s: {rolling_ds:.4f} | Lead Time: {lead_time:.2f}h | FPS: {fps:.1f}")

            # Display (optional)
            if display:
                # Draw detections on frame
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add info overlay
                info_text = f"D_s: {ds:.4f} | Lead Time: {self.current_lead_time:.2f}h" if self.current_lead_time else f"D_s: {ds:.4f}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Fabric Defect Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elapsed = time.time() - start_time
        self.processing_stats['avg_fps'] = self.frame_count / elapsed if elapsed > 0 else 0

        if display:
            cv2.destroyAllWindows()

        return self.processing_stats


def run_benchmark(unified_predictor: LeadTimePredictor,
                  X_test: np.ndarray, y_test: np.ndarray,
                  baseline_lead_time: float) -> dict:
    """
    Benchmark unified model vs static baseline.

    The static baseline represents traditional manufacturing planning:
    a single predicted lead time based on average historical data, ignoring
    real-time defect information. The unified model incorporates D_s, enabling
    dynamic adjustment to changing quality conditions.

    Args:
        unified_predictor: Trained predictor with D_s feature
        X_test: Test features (includes D_s)
        y_test: True lead times
        baseline_lead_time: Static prediction (mean of training data)

    Returns:
        Benchmark results dictionary
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Unified model predictions
    y_pred_unified = unified_predictor.predict(X_test)

    # Static baseline predictions
    y_pred_baseline = np.full_like(y_test, baseline_lead_time)

    # Calculate metrics
    unified_mae = mean_absolute_error(y_test, y_pred_unified)
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

    unified_rmse = np.sqrt(mean_squared_error(y_test, y_pred_unified))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

    unified_r2 = r2_score(y_test, y_pred_unified)
    baseline_r2 = r2_score(y_test, y_pred_baseline)

    results = {
        'unified': {
            'mae': float(unified_mae),
            'rmse': float(unified_rmse),
            'r2': float(unified_r2)
        },
        'baseline': {
            'mae': float(baseline_mae),
            'rmse': float(baseline_rmse),
            'r2': float(baseline_r2)
        },
        'mae_improvement': float(baseline_mae - unified_mae),
        'mae_improvement_pct': float((baseline_mae - unified_mae) / baseline_mae * 100),
        'rmse_improvement': float(baseline_rmse - unified_rmse)
    }

    return results


def main():
    """
    Main entry point demonstrating the unified framework.
    """
    print("=" * 70)
    print("Unified Fabric Defect Detection & Lead Time Prediction Framework")
    print("=" * 70)

    # Section 1: Train and benchmark the model
    print("\n" + "=" * 70)
    print("SECTION 1: Model Training & Benchmarking")
    print("=" * 70)

    print("\nGenerating synthetic training data...")
    X_train, X_test, y_train, y_test = generate_synthetic_data()

    print("Training unified model (with D_s feature)...")
    unified_model = LeadTimePredictor(n_estimators=100, max_depth=10)
    unified_model.fit(X_train, y_train)

    baseline_lead_time = float(np.mean(y_train))

    print("\nRunning benchmark...")
    benchmark_results = run_benchmark(unified_model, X_test, y_test, baseline_lead_time)

    print("\n" + "-" * 50)
    print("BENCHMARK RESULTS")
    print("-" * 50)
    print(f"\nUnified Model (with real-time D_s):")
    print(f"  MAE:  {benchmark_results['unified']['mae']:.3f} hours")
    print(f"  RMSE: {benchmark_results['unified']['rmse']:.3f} hours")
    print(f"  R2:   {benchmark_results['unified']['r2']:.4f}")

    print(f"\nStatic Baseline (ignores D_s):")
    print(f"  MAE:  {benchmark_results['baseline']['mae']:.3f} hours")
    print(f"  RMSE: {benchmark_results['baseline']['rmse']:.3f} hours")
    print(f"  R2:   {benchmark_results['baseline']['r2']:.4f}")

    print(f"\n" + "=" * 50)
    print("IMPROVEMENT FROM INTEGRATION")
    print("=" * 50)
    print(f"MAE Reduction:  {benchmark_results['mae_improvement']:.3f} hours")
    print(f"MAE Improvement: {benchmark_results['mae_improvement_pct']:.1f}%")
    print(f"\nThe unified model reduces prediction error by incorporating")
    print(f"real-time defect density, closing the integration gap between")
    print(f"vision systems and production planning.")

    # Section 2: Live image processing (if images available)
    print("\n" + "=" * 70)
    print("SECTION 2: Live Image Stream Processing")
    print("=" * 70)

    # Check for sample images folder
    sample_folders = [
        "./sample_fabric_images",
        "./images",
        "./data",
        str(Path.home() / "Desktop" / "sample_images")
    ]

    image_folder = None
    for folder in sample_folders:
        if Path(folder).exists() and any(Path(folder).iterdir()):
            image_folder = folder
            break

    if image_folder:
        print(f"\nProcessing images from: {image_folder}")
        processor = LiveStreamProcessor(
            image_folder=image_folder,
            machine_speed=60.0,
            order_quantity=2000,
            detection_interval=PREDICTION_INTERVAL
        )
        stats = processor.run(display=True, verbose=True)
        print(f"\nProcessing complete: {stats['total_frames']} frames at {stats['avg_fps']:.1f} FPS")
    else:
        print("\nNo sample image folder found. Creating demo with synthetic frames...")
        print("(In production, provide path to fabric defect images)")

        # Demo with synthetic frames
        detector = FabricDefectDetector()
        preprocessor = IndustrialPreprocessor()
        bridge = MetadataBridge()
        predictor = LeadTimePredictor()
        predictor.fit(X_train, y_train)

        print("\nProcessing 30 synthetic frames...")
        for i in range(30):
            synthetic_frame = np.random.randint(80, 200, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect(synthetic_frame, apply_preprocessing=True)
            ds = bridge.update(detections, synthetic_frame.shape)

            if (i + 1) % PREDICTION_INTERVAL == 0:
                rolling_ds = bridge.get_rolling_ds()
                lead_time = predictor.predict_single(60.0, 2000, rolling_ds)
                print(f"  Frame {i+1:2d}: D_s={ds:.4f}, Rolling D_s={rolling_ds:.4f}, Lead Time={lead_time:.2f}h")

        print("\nDemo complete. To process real images, place them in a folder and call:")
        print("  processor = LiveStreamProcessor('path/to/images')")
        print("  processor.run()")

    print("\n" + "=" * 70)
    print("Framework Execution Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()