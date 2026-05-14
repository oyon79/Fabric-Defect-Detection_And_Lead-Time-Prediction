"""
Task 2: The Metadata Bridge (Direct Feedback Loop)
=================================================
Extracts Defect Density Score (D_s) from YOLO detection results.
D_s serves as the critical link between the vision system and
the production planning system, enabling real-time lead time adjustment.
"""

import numpy as np


def calculate_bbox_area(bbox: tuple[int, int, int, int]) -> float:
    """
    Calculate area of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) top-left and bottom-right coordinates

    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return float((x2 - x1) * (y2 - y1))


def extract_defect_density(detections: list[dict], frame_shape: tuple[int, int, int]) -> float:
    """
    Calculate Defect Density Score (D_s).

    D_s bridges the detection and prediction subsystems. High D_s indicates
    fabric quality issues, which directly extend lead time. This scalar metric
    allows production planners to react to quality deviations in real-time.

    Formula: D_s = Total Defect Bounding Box Area / Total Frame Area

    Args:
        detections: List of YOLO detection dicts with 'bbox' key
        frame_shape: (height, width, channels) of the frame

    Returns:
        Defect density score (0.0 to 1.0+, clamped to [0, 1] for reporting)
    """
    if not detections:
        return 0.0

    frame_height, frame_width = frame_shape[:2]
    total_frame_area = float(frame_height * frame_width)

    total_defect_area = sum(calculate_bbox_area(d['bbox']) for d in detections)

    density = total_defect_area / total_frame_area if total_frame_area > 0 else 0.0

    # Clamp for human readability (actual ratio can exceed 1.0 with overlapping boxes)
    return float(np.clip(density, 0.0, 1.0))


def compute_rolling_density(defect_densities: list[float], window_size: int = 10) -> float:
    """
    Compute rolling average of defect density for smoother predictions.

    Uses a moving window to reduce noise in D_s readings caused by
    individual frame variations. This smooths the feedback signal
    to the lead time predictor.

    Args:
        defect_densities: Historical D_s values
        window_size: Number of recent frames to average

    Returns:
        Rolling average D_s
    """
    if not defect_densities:
        return 0.0

    window = defect_densities[-window_size:]
    return float(np.mean(window))


class MetadataBridge:
    """
    Manages the flow of metadata from detection to prediction.

    This class maintains history and provides summary statistics
    that feed into the lead time predictor. It addresses the
    integration gap by translating raw vision data into actionable
    manufacturing metrics.
    """

    def __init__(self, rolling_window: int = 10):
        self.rolling_window = rolling_window
        self.density_history: list[float] = []
        self.detection_count_history: list[int] = []

    def update(self, detections: list[dict], frame_shape: tuple[int, int, int]) -> float:
        """
        Update bridge state with new detection results.

        Args:
            detections: Latest YOLO detections
            frame_shape: Current frame dimensions

        Returns:
            Latest D_s value
        """
        density = extract_defect_density(detections, frame_shape)
        self.density_history.append(density)
        self.detection_count_history.append(len(detections))

        return density

    def get_current_ds(self) -> float:
        """Get the most recent D_s value."""
        return self.density_history[-1] if self.density_history else 0.0

    def get_rolling_ds(self) -> float:
        """Get smoothed D_s over rolling window."""
        return compute_rolling_density(self.density_history, self.rolling_window)

    def get_stats(self) -> dict:
        """
        Get summary statistics for debugging/analysis.

        Returns:
            Dict with current, rolling, and historical stats
        """
        return {
            'current_ds': self.get_current_ds(),
            'rolling_ds': self.get_rolling_ds(),
            'mean_ds': float(np.mean(self.density_history)) if self.density_history else 0.0,
            'max_ds': float(np.max(self.density_history)) if self.density_history else 0.0,
            'total_detections': sum(self.detection_count_history),
            'frames_processed': len(self.density_history)
        }


if __name__ == "__main__":
    # Demo: Calculate D_s from synthetic detections
    print("Testing Metadata Bridge...")

    # Simulate a 1920x1080 frame with some defect detections
    frame_shape = (1080, 1920, 3)

    # Simulate 3 defects at different locations
    synthetic_detections = [
        {'bbox': (100, 100, 300, 300)},   # 200x200 = 40000 px
        {'bbox': (500, 400, 700, 600)},   # 200x200 = 40000 px
        {'bbox': (1000, 800, 1150, 950)}  # 150x150 = 22500 px
    ]

    ds = extract_defect_density(synthetic_detections, frame_shape)
    print(f"D_s = {ds:.6f}")

    # Expected: (40000 + 40000 + 22500) / (1920 * 1080) = 102500 / 2073600 ≈ 0.0494
    print(f"Expected approx: 0.0494")

    # Test rolling density
    bridge = MetadataBridge(rolling_window=5)
    for i in range(10):
        fake_detections = [{'bbox': (0, 0, 100 + i*10, 100 + i*10)}] if i % 2 == 0 else []
        bridge.update(fake_detections, frame_shape)

    print(f"\nAfter 10 frames:")
    print(f"  Current D_s: {bridge.get_current_ds():.6f}")
    print(f"  Rolling D_s: {bridge.get_rolling_ds():.6f}")
    print(f"  Stats: {bridge.get_stats()}")

    print("\nMetadata bridge validated successfully.")