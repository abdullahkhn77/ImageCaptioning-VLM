"""
Drift detection for input data and model performance.
Uses Alibi Detect for data drift detection.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import torch
from PIL import Image
from alibi_detect import KSDrift, MMDDrift
from alibi_detect.saving import save_detector, load_detector
import logging

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift in input images."""
    
    def __init__(self, reference_data: np.ndarray, detector_type: str = 'ks'):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (baseline)
            detector_type: 'ks' for Kolmogorov-Smirnov or 'mmd' for Maximum Mean Discrepancy
        """
        self.reference_data = reference_data
        self.detector_type = detector_type
        
        if detector_type == 'ks':
            self.detector = KSDrift(reference_data, p_val=0.05)
        elif detector_type == 'mmd':
            self.detector = MMDDrift(reference_data, backend='pytorch')
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def detect_drift(self, new_data: np.ndarray) -> Dict:
        """
        Detect drift in new data.
        
        Returns:
            Dictionary with drift detection results
        """
        predictions = self.detector.predict(new_data)
        
        return {
            'drift_detected': predictions['data']['is_drift'],
            'p_value': predictions['data'].get('p_val', None),
            'distance': predictions['data'].get('distance', None),
            'threshold': predictions['data'].get('threshold', None)
        }
    
    def save(self, path: str):
        """Save detector to disk."""
        save_detector(self.detector, path)
        logger.info(f"Saved drift detector to {path}")
    
    @classmethod
    def load(cls, path: str, reference_data: np.ndarray):
        """Load detector from disk."""
        detector = load_detector(path)
        instance = cls.__new__(cls)
        instance.detector = detector
        instance.reference_data = reference_data
        instance.detector_type = 'ks'  # Default, adjust if needed
        return instance


class PerformanceDriftDetector:
    """Detect performance drift in model predictions."""
    
    def __init__(self, baseline_metrics: Dict[str, float], threshold: float = 0.1):
        """
        Initialize performance drift detector.
        
        Args:
            baseline_metrics: Baseline performance metrics
            threshold: Relative change threshold for drift detection
        """
        self.baseline_metrics = baseline_metrics
        self.threshold = threshold
    
    def detect_drift(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Detect performance drift.
        
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                relative_change = abs(current_value - baseline_value) / baseline_value
                
                drift_detected = relative_change > self.threshold
                
                drift_results[metric_name] = {
                    'drift_detected': drift_detected,
                    'baseline': baseline_value,
                    'current': current_value,
                    'relative_change': relative_change,
                    'threshold': self.threshold
                }
        
        return drift_results


def extract_image_features(images: List[Image.Image], model, processor) -> np.ndarray:
    """Extract features from images for drift detection."""
    features = []
    
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            # Extract image features (adjust based on model architecture)
            outputs = model.get_image_features(**inputs)
            features.append(outputs.cpu().numpy().flatten())
    
    return np.array(features)


def monitor_drift(reference_features: np.ndarray, 
                  new_features: np.ndarray,
                  detector_type: str = 'ks') -> Dict:
    """Monitor data drift between reference and new data."""
    detector = DataDriftDetector(reference_features, detector_type)
    results = detector.detect_drift(new_features)
    
    if results['drift_detected']:
        logger.warning("Data drift detected!")
    else:
        logger.info("No data drift detected")
    
    return results

