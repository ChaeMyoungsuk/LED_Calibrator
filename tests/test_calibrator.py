"""Tests for led_calibrator.calibrator."""

import numpy as np
import pytest

from led_calibrator.models import LEDModule, UniformityResult
from led_calibrator.calibrator import LEDCalibrator
from led_calibrator.uniformity_analyzer import UniformityAnalyzer


def _make_result(row, col, rgb, region=(0, 0, 100, 100)):
    mean_rgb = np.array(rgb, dtype=np.float64)
    lum = float(np.dot(mean_rgb / 255.0, [0.2126, 0.7152, 0.0722]) * 255.0)
    return UniformityResult(
        module=LEDModule(row=row, col=col, region=region),
        mean_rgb=mean_rgb,
        std_rgb=np.zeros(3),
        luminance=lum,
    )


class TestLEDCalibratorComputeCalibration:
    def test_uniform_input_gives_unit_gain(self):
        """When all modules have the same colour, no correction is needed."""
        results = [
            _make_result(0, 0, (200.0, 200.0, 200.0)),
            _make_result(0, 1, (200.0, 200.0, 200.0)),
            _make_result(1, 0, (200.0, 200.0, 200.0)),
            _make_result(1, 1, (200.0, 200.0, 200.0)),
        ]
        cal = LEDCalibrator()
        calibrations = cal.compute_calibration(results)
        for c in calibrations:
            np.testing.assert_array_almost_equal(c.gain_rgb, [1.0, 1.0, 1.0])

    def test_gain_corrects_towards_target(self):
        """A dim module should receive a gain > 1."""
        results = [
            _make_result(0, 0, (200.0, 200.0, 200.0)),
            _make_result(0, 1, (100.0, 100.0, 100.0)),  # half-brightness
        ]
        cal = LEDCalibrator()
        calibrations = cal.compute_calibration(results)
        # Panel mean = 150, dim module gain should be ~1.5
        dim_cal = calibrations[1]
        assert dim_cal.gain_rgb[0] > 1.0

    def test_zero_mean_gets_unity_gain(self):
        """Modules with zero measured value should not get infinite gain."""
        results = [
            _make_result(0, 0, (0.0, 0.0, 0.0)),
            _make_result(0, 1, (100.0, 100.0, 100.0)),
        ]
        cal = LEDCalibrator()
        calibrations = cal.compute_calibration(results)
        zero_cal = calibrations[0]
        # Gain should be clamped (not inf/nan)
        assert np.all(np.isfinite(zero_cal.gain_rgb))
        assert np.all(zero_cal.gain_rgb <= 10.0)

    def test_target_luminance_scales_gain(self):
        """Providing target_luminance=255 should push bright modules to 255."""
        results = [_make_result(0, 0, (200.0, 200.0, 200.0))]
        cal = LEDCalibrator(target_luminance=255.0)
        calibrations = cal.compute_calibration(results)
        assert calibrations[0].gain_rgb[0] > 1.0

    def test_output_count_matches_input(self):
        results = [_make_result(r, c, (180.0, 180.0, 180.0)) for r in range(3) for c in range(4)]
        cal = LEDCalibrator()
        calibrations = cal.compute_calibration(results)
        assert len(calibrations) == 12


class TestLEDCalibratorApplyCalibration:
    def _build_non_uniform_setup(self):
        """Return (image, analyzer, calibrations) for a 1×2 panel."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100, :] = 200
        image[:, 100:, :] = 100
        analyzer = UniformityAnalyzer(rows=1, cols=2)
        results = analyzer.analyze(image)
        calibrator = LEDCalibrator()
        calibrations = calibrator.compute_calibration(results)
        return image, analyzer, calibrations

    def test_apply_returns_uint8(self):
        image, analyzer, calibrations = self._build_non_uniform_setup()
        cal = LEDCalibrator()
        out = cal.apply_calibration(image, calibrations)
        assert out.dtype == np.uint8

    def test_apply_same_shape(self):
        image, analyzer, calibrations = self._build_non_uniform_setup()
        cal = LEDCalibrator()
        out = cal.apply_calibration(image, calibrations)
        assert out.shape == image.shape

    def test_calibration_improves_uniformity(self):
        image, analyzer, calibrations = self._build_non_uniform_setup()
        calibrator = LEDCalibrator()
        calibrated = calibrator.apply_calibration(image, calibrations)
        before = analyzer.analyze(image)
        after = analyzer.analyze(calibrated)
        ratio_before = analyzer.compute_uniformity_ratio(before)
        ratio_after = analyzer.compute_uniformity_ratio(after)
        assert ratio_after >= ratio_before

    def test_uniform_image_unchanged_after_calibration(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        analyzer = UniformityAnalyzer(rows=2, cols=2)
        results = analyzer.analyze(image)
        cal = LEDCalibrator()
        calibrations = cal.compute_calibration(results)
        out = cal.apply_calibration(image, calibrations)
        # Pixels should be very close (small floating-point rounding possible)
        np.testing.assert_array_equal(out, image)

    def test_uniformity_improvement_method(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100, :] = 200
        image[:, 100:, :] = 100
        analyzer = UniformityAnalyzer(rows=1, cols=2)
        results = analyzer.analyze(image)
        calibrator = LEDCalibrator()
        calibrations = calibrator.compute_calibration(results)
        calibrated = calibrator.apply_calibration(image, calibrations)
        improvement = calibrator.uniformity_improvement(results, calibrated, analyzer)
        assert improvement >= 0.0
