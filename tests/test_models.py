"""Tests for led_calibrator.models."""

import numpy as np
import pytest

from led_calibrator.models import LEDModule, UniformityResult, CalibrationData


class TestLEDModule:
    def test_id_format(self):
        m = LEDModule(row=1, col=2, region=(0, 0, 100, 50))
        assert m.id == "module_r001_c002"

    def test_width_height(self):
        m = LEDModule(row=0, col=0, region=(10, 20, 110, 70))
        assert m.width == 100
        assert m.height == 50

    def test_zero_size_region(self):
        m = LEDModule(row=0, col=0, region=(5, 5, 5, 5))
        assert m.width == 0
        assert m.height == 0


class TestUniformityResult:
    def _make_result(self, rgb=(200.0, 190.0, 180.0)):
        module = LEDModule(row=0, col=0, region=(0, 0, 100, 100))
        mean_rgb = np.array(rgb, dtype=np.float64)
        std_rgb = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        # luminance = dot([r,g,b]/255, [0.2126, 0.7152, 0.0722]) * 255
        lum = float(np.dot(mean_rgb / 255.0, [0.2126, 0.7152, 0.0722]) * 255.0)
        return UniformityResult(module=module, mean_rgb=mean_rgb, std_rgb=std_rgb, luminance=lum)

    def test_channel_accessors(self):
        res = self._make_result((200.0, 190.0, 180.0))
        assert res.mean_r == pytest.approx(200.0)
        assert res.mean_g == pytest.approx(190.0)
        assert res.mean_b == pytest.approx(180.0)


class TestCalibrationData:
    def _make_cal(self, gain=(1.0, 1.2, 0.9), offset=(0.0, 0.0, 0.0)):
        module = LEDModule(row=0, col=0, region=(0, 0, 100, 100))
        return CalibrationData(
            module=module,
            gain_rgb=np.array(gain, dtype=np.float64),
            offset_rgb=np.array(offset, dtype=np.float64),
            target_luminance=200.0,
        )

    def test_apply_gain_only(self):
        cal = self._make_cal(gain=(1.0, 1.0, 1.0))
        pixel = np.array([100.0, 150.0, 200.0])
        result = cal.apply(pixel)
        np.testing.assert_array_almost_equal(result, [100.0, 150.0, 200.0])

    def test_apply_gain_scales(self):
        cal = self._make_cal(gain=(2.0, 0.5, 1.0))
        pixel = np.array([50.0, 200.0, 100.0])
        result = cal.apply(pixel)
        np.testing.assert_array_almost_equal(result, [100.0, 100.0, 100.0])

    def test_apply_clamps_to_255(self):
        cal = self._make_cal(gain=(3.0, 1.0, 1.0))
        pixel = np.array([200.0, 100.0, 100.0])
        result = cal.apply(pixel)
        assert result[0] == pytest.approx(255.0)

    def test_apply_clamps_to_zero(self):
        cal = self._make_cal(gain=(1.0, 1.0, 1.0), offset=(-50.0, 0.0, 0.0))
        pixel = np.array([20.0, 100.0, 100.0])
        result = cal.apply(pixel)
        assert result[0] == pytest.approx(0.0)

    def test_gain_accessors(self):
        cal = self._make_cal(gain=(1.1, 1.2, 1.3))
        assert cal.gain_r == pytest.approx(1.1)
        assert cal.gain_g == pytest.approx(1.2)
        assert cal.gain_b == pytest.approx(1.3)
