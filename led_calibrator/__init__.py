"""
LED_Calibrator
==============

Professional image quality tuning tool specialised in LED white uniformity
calibration and calibration-region alignment.

Public API
----------
UniformityAnalyzer   – analyse white uniformity per LED module
LEDCalibrator        – compute and apply RGB gain/offset corrections
RegionMatcher        – align calibration regions to physical LED modules
Visualizer           – render uniformity heat-maps and correction maps
"""

from .models import LEDModule, CalibrationData, UniformityResult
from .uniformity_analyzer import UniformityAnalyzer
from .calibrator import LEDCalibrator
from .region_matcher import RegionMatcher
from .visualizer import Visualizer

__all__ = [
    "LEDModule",
    "CalibrationData",
    "UniformityResult",
    "UniformityAnalyzer",
    "LEDCalibrator",
    "RegionMatcher",
    "Visualizer",
]
