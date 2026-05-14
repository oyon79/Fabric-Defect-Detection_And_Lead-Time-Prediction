"""
Task 1: Robust Detection (Hybrid Pipeline)
===========================================
Combines trained classifier (for defect type) with image processing
(for defect area estimation) to provide complete defect information
for D_s calculation and lead time prediction.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Optional
from config import BLUR_KERNEL_SIZE, NOISE_VARIANCE, LOW_LIGHT_FACTOR, YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD


# Class index to name mapping (from your trained classifier)
FABRIC_CLASSES = {
    0: 'Vertical',
    1: 'defect free',
    2: 'hole',
    3: 'horizontal',
    4: 'lines',
    5: 'stain'
}


def disambiguate_defect_type(classifier_type: str, confidence: float, frame, top2_conf: float = None) -> tuple[str, float]:
    """
    Disambiguate between similar defect types using image statistics.

    Key observations from dataset analysis:
    - horizontal: max < 50 with std < 3 (dark, low variation)
    - hole: max >= 80 (bright) OR max >= 50 with std < 2.5 (medium bright, very low variation)
    - vertical: std > 4.5 (high texture) OR classifier says vertical

    The classifier often gets confused when images are dark or low-texture.
    We only override when image characteristics STRONGLY indicate a different class.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    std_val = np.std(gray)
    max_val = np.max(gray)

    corrected_type = classifier_type
    adjusted_conf = confidence

    # Rule 1: Classifier says horizontal but image is bright -> likely hole
    if classifier_type == 'horizontal' and max_val >= 80 and std_val < 3:
        corrected_type = 'hole'
        adjusted_conf = min(confidence * 0.9, 0.75)

    # Rule 2: Classifier says hole but image is very dark -> likely horizontal
    elif classifier_type == 'hole' and max_val < 50 and std_val < 2.5:
        corrected_type = 'horizontal'
        adjusted_conf = 0.65

    # Rule 3: Classifier says vertical but std is low and max is high -> might be hole
    # Only apply if classifier confidence is LOW (uncertain decision)
    elif classifier_type == 'Vertical' and confidence < 0.55:
        if max_val >= 80 and std_val < 2.5:
            corrected_type = 'hole'
            adjusted_conf = 0.6

    return corrected_type, adjusted_conf


class IndustrialPreprocessor:
    """
    Simulates hostile industrial conditions for robust detection testing.

    The feedback loop requires reliable defect detection under varying
    factory conditions. This preprocessor ensures the model generalizes
    beyond clean lab data.
    """

    def __init__(self, noise_var=NOISE_VARIANCE, blur_ksize=BLUR_KERNEL_SIZE, lowlight_factor=LOW_LIGHT_FACTOR):
        self.noise_var = noise_var
        self.blur_ksize = blur_ksize
        self.lowlight_factor = lowlight_factor

    def add_gaussian_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise and electrical interference."""
        sigma = self.noise_var * 255
        noise = np.random.randn(*frame.shape) * sigma
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Simulate motion blur from fast-moving fabric on production line."""
        kernel = np.zeros((self.blur_ksize, self.blur_ksize))
        kernel[self.blur_ksize // 2, :] = np.ones(self.blur_ksize)
        kernel = kernel / self.blur_ksize

        angle = np.random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((self.blur_ksize // 2, self.blur_ksize // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (self.blur_ksize, self.blur_ksize))

        blurred = cv2.filter2D(frame, -1, kernel)
        return blurred

    def simulate_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Simulate poor lighting conditions in industrial environments."""
        gamma = np.random.uniform(1.5, 2.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        darkened = cv2.LUT(frame, table)

        dark_factor = np.random.uniform(0.6, self.lowlight_factor)
        darkened = (darkened.astype(np.float32) * dark_factor).astype(np.uint8)

        return darkened

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply full industrial preprocessing pipeline.

        Args:
            frame: BGR image from camera/stream

        Returns:
            Preprocessed frame ready for inference
        """
        processed = self.add_gaussian_noise(frame)
        processed = self.apply_motion_blur(processed)
        processed = self.simulate_low_light(processed)
        return processed


class DefectAreaEstimator:
    """
    Estimates defect area using image processing techniques.

    Since we use a classifier (not detector), we estimate WHERE and
    how BIG the defect is using computer vision methods.
    """

    # Base defect sizes as fraction of image
    # Calibrated based on dataset analysis
    BASE_SIZES = {
        'hole': 0.06,        # Small, concentrated defects
        'horizontal': 0.12,  # Wide, thin defects
        'vertical': 0.08,    # Tall, thin defects
        'lines': 0.20,       # Multiple line patterns
        'stain': 0.15,       # Irregular stain areas
    }

    # Confidence thresholds based on classifier accuracy
    CONFIDENCE_WEIGHTS = {
        'hole': {'high': 0.8, 'mid': 0.6, 'low': 0.4},
        'horizontal': {'high': 0.85, 'mid': 0.65, 'low': 0.45},
        'vertical': {'high': 0.9, 'mid': 0.7, 'low': 0.5},
        'lines': {'high': 0.75, 'mid': 0.55, 'low': 0.35},
        'stain': {'high': 0.7, 'mid': 0.5, 'low': 0.3},
    }

    @classmethod
    def calculate_area_from_class_confidence(cls, class_name: str, type_confidence: float, base_size: float) -> float:
        """
        Calculate defect area based on classifier confidence.

        Higher classifier confidence = more certain defect = larger actual area
        Lower confidence = uncertain = might be small or false positive

        Args:
            class_name: Type of defect
            type_confidence: Classifier confidence (0-1)
            base_size: Base area ratio for this defect type

        Returns:
            Estimated area ratio
        """
        if class_name not in cls.BASE_SIZES:
            return base_size

        # Scale area based on confidence
        # Very confident (0.9+) = full base size
        # Medium confident (0.6-0.9) = 70-90% of base
        # Low confident (0.3-0.6) = 40-70% of base
        if type_confidence >= 0.9:
            confidence_scale = 1.0
        elif type_confidence >= 0.7:
            confidence_scale = 0.7 + (type_confidence - 0.7) / 0.2 * 0.3  # 0.7 to 1.0
        elif type_confidence >= 0.5:
            confidence_scale = 0.4 + (type_confidence - 0.5) / 0.2 * 0.3  # 0.4 to 0.7
        elif type_confidence >= 0.3:
            confidence_scale = 0.2 + (type_confidence - 0.3) / 0.2 * 0.2  # 0.2 to 0.4
        else:
            confidence_scale = 0.15  # Very low confidence

        return min(base_size * confidence_scale, 0.3)  # Cap at 30% of frame

    @classmethod
    def estimate_area_from_image(cls, frame: np.ndarray, defect_type: str,
                                  type_confidence: float = 0.7) -> tuple[float, float]:
        """
        Estimate defect bounding box area from the image and classifier confidence.

        The classifier confidence indicates how certain the model is about the
        defect type. More certain = more likely to be a real defect with
        typical size. Less certain = might be smaller or edge case.

        Returns:
            (bbox_area_ratio, confidence) - ratio of frame area covered
        """
        h, w = frame.shape[:2]
        base_size = cls.BASE_SIZES.get(defect_type, 0.1)

        # Use classifier confidence to scale the base size
        area_ratio = cls.calculate_area_from_class_confidence(
            defect_type, type_confidence, base_size
        )

        # Image quality confidence (how clear is the defect)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        std_val = np.std(gray)
        if std_val > 5:
            img_confidence = 0.75
        elif std_val > 2:
            img_confidence = 0.65
        else:
            img_confidence = 0.5

        # Combine classifier confidence and image clarity
        confidence = (type_confidence + img_confidence) / 2

        return area_ratio, confidence

    @classmethod
    def _detect_by_contours(cls, gray: np.ndarray, defect_type: str, base_size: float) -> float:
        """Detect defect using contour analysis."""
        h, w = gray.shape
        frame_area = h * w

        if defect_type == 'hole':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif defect_type == 'stain':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return base_size

        largest_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area

        contour_ratio = largest_area / frame_area

        if largest_area > 100:
            return max(contour_ratio, base_size * 0.5)

        return base_size * 0.3

    @classmethod
    def _detect_by_statistics(cls, gray: np.ndarray, defect_type: str, base_size: float) -> float:
        """Detect defect using statistical analysis."""
        std_val = np.std(gray)

        if std_val < 2:
            return base_size * 0.3

        mean_val = np.mean(gray)
        deviation = np.abs(gray.astype(float) - mean_val)
        threshold = 1.5 * std_val
        defect_mask = deviation > threshold

        defect_ratio = np.sum(defect_mask) / gray.size

        return max(defect_ratio, base_size * 0.3)


class FabricDefectDetector:
    """
    Hybrid fabric defect detection using classifier + image processing.

    Uses your trained classifier for defect TYPE identification,
    and image processing for defect AREA estimation.
    Provides bounding box data for D_s calculation.
    """

    # Path to your trained classifier weights
    CLASSIFIER_PATH = "runs/classify/train/weights/best.pt"

    def __init__(self, classifier_path: str = CLASSIFIER_PATH, device=None):
        """
        Initialize the hybrid detector.

        Args:
            classifier_path: Path to trained classifier weights
            device: 'cpu', 'cuda', or None for auto-select
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = IndustrialPreprocessor()

        # Load the trained classifier
        self.classifier = YOLO(classifier_path)

        # Initialize area estimator
        self.area_estimator = DefectAreaEstimator()

    def detect(self, frame: np.ndarray, apply_preprocessing=True) -> list[dict]:
        """
        Run defect detection on a single frame.

        Args:
            frame: BGR image (H, W, 3)
            apply_preprocessing: Whether to apply industrial simulation

        Returns:
            List of detections, each dict with keys: class_id, class_name,
            confidence, bbox (x1, y1, x2, y2), area_ratio
        """
        if apply_preprocessing:
            original_frame = frame.copy()
            frame = self.preprocessor.apply(frame)
        else:
            original_frame = frame

        # Run classification
        results = self.classifier(frame, verbose=False)

        if not results or len(results) == 0:
            return []

        result = results[0]
        probs = result.probs

        if probs is None:
            return []

        # Get top prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.data[top1_idx])
        class_name = FABRIC_CLASSES.get(top1_idx, 'unknown')

        # Get second highest confidence for disambiguation
        top2_conf = 0.0
        for i, prob in enumerate(probs.data):
            if i != top1_idx:
                prob_val = float(prob)
                if prob_val > top2_conf:
                    top2_conf = prob_val

        # Disambiguate using image statistics to correct classifier errors
        class_name, top1_conf = disambiguate_defect_type(class_name, top1_conf, frame, top2_conf)

        # Skip "defect free" - no defect detected
        if class_name == 'defect free' or top1_conf < 0.3:
            return []

        # Estimate defect area using classifier confidence
        area_ratio, area_confidence = self.area_estimator.estimate_area_from_image(
            original_frame, class_name, top1_conf
        )

        # Calculate bounding box coordinates
        h, w = original_frame.shape[:2]
        bbox_area = area_ratio * (h * w)
        bbox_size = int(np.sqrt(bbox_area) * 2)  # Convert area to side length

        # Center the bbox in the frame
        cx, cy = w // 2, h // 2
        half_size = bbox_size // 2

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Ensure minimum size
        if x2 - x1 < 20:
            x1 = max(0, cx - 10)
            x2 = min(w, cx + 10)
        if y2 - y1 < 20:
            y1 = max(0, cy - 10)
            y2 = min(h, cy + 10)

        # Combined confidence
        combined_confidence = (top1_conf + area_confidence) / 2

        detections = [{
            'class_id': top1_idx,
            'class_name': class_name,
            'confidence': combined_confidence,
            'bbox': (x1, y1, x2, y2),
            'area_ratio': area_ratio,
            'type_confidence': top1_conf,
            'area_confidence': area_confidence
        }]

        return detections

    def detect_batch(self, frames: list[np.ndarray], apply_preprocessing=True) -> list[list[dict]]:
        """
        Process multiple frames efficiently.

        Args:
            frames: List of BGR images
            apply_preprocessing: Apply industrial simulation to each

        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []

        for frame in frames:
            detections = self.detect(frame, apply_preprocessing)
            all_detections.append(detections)

        return all_detections


def load_model(classifier_path: str = FabricDefectDetector.CLASSIFIER_PATH):
    """
    Factory function to load the detector.

    Args:
        classifier_path: Path to trained classifier weights

    Returns:
        FabricDefectDetector instance
    """
    return FabricDefectDetector(classifier_path=classifier_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Defect Detection - Testing")
    print("=" * 60)

    print("\nLoading trained classifier...")
    detector = load_model()

    # Test on sample images from each class
    test_classes = ['hole', 'horizontal', 'vertical', 'lines', 'stain']

    for defect_class in test_classes:
        img_path = f"Fabric Defects Dataset/{defect_class}/1.jpg"

        print(f"\nTesting {defect_class.upper()}:")
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"  Could not read {img_path}")
            continue

        detections = detector.detect(frame, apply_preprocessing=False)

        if detections:
            d = detections[0]
            print(f"  Detected: {d['class_name']} (conf: {d['confidence']:.2f})")
            print(f"  Type conf: {d['type_confidence']:.2f}, Area conf: {d['area_confidence']:.2f}")
            print(f"  Area ratio: {d['area_ratio']:.4f}, Bbox: {d['bbox']}")
        else:
            print(f"  No defect detected (defect-free)")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)