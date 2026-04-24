"""Tests for led_calibrator.uniformity_analyzer."""

import numpy as np
import pytest

from led_calibrator.uniformity_analyzer import UniformityAnalyzer, _luminance


class TestLuminanceHelper:
    def test_pure_white(self):
        rgb = np.array([255.0, 255.0, 255.0])
        assert _luminance(rgb) == pytest.approx(255.0, abs=0.5)

    def test_pure_black(self):
        rgb = np.array([0.0, 0.0, 0.0])
        assert _luminance(rgb) == pytest.approx(0.0)

    def test_known_value(self):
        # Confirm BT.709 coefficients
        rgb = np.array([255.0, 0.0, 0.0])
        expected = 0.2126 * 255.0
        assert _luminance(rgb) == pytest.approx(expected, rel=1e-4)


class TestUniformityAnalyzer:
    def _uniform_image(self, rows, cols, value=200, h=400, w=600):
        """Return a uniformly lit image."""
        return np.full((h, w, 3), value, dtype=np.uint8)

    def test_correct_number_of_results(self):
        analyzer = UniformityAnalyzer(rows=4, cols=6)
        image = self._uniform_image(4, 6)
        results = analyzer.analyze(image)
        assert len(results) == 24

    def test_uniform_image_gives_perfect_uniformity(self):
        analyzer = UniformityAnalyzer(rows=3, cols=4)
        image = self._uniform_image(3, 4, value=180)
        results = analyzer.analyze(image)
        ratio = analyzer.compute_uniformity_ratio(results)
        assert ratio == pytest.approx(1.0, abs=1e-4)

    def test_mean_rgb_matches_pixel_value(self):
        analyzer = UniformityAnalyzer(rows=2, cols=2)
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        results = analyzer.analyze(image)
        for r in results:
            np.testing.assert_array_almost_equal(r.mean_rgb, [128.0, 128.0, 128.0], decimal=1)

    def test_invalid_rows_cols(self):
        with pytest.raises(ValueError):
            UniformityAnalyzer(rows=0, cols=3)
        with pytest.raises(ValueError):
            UniformityAnalyzer(rows=3, cols=0)

    def test_invalid_image_shape(self):
        analyzer = UniformityAnalyzer(rows=2, cols=2)
        with pytest.raises(ValueError):
            analyzer.analyze(np.zeros((100, 100)))  # missing channel dim

    def test_padding_is_stripped(self):
        analyzer = UniformityAnalyzer(rows=1, cols=1, image_padding=(10, 10, 10, 10))
        # 200×200 with 10px padding on each side → 180×180 effective region
        image = np.full((200, 200, 3), 200, dtype=np.uint8)
        # Set padding region to a different value
        image[:10, :, :] = 0
        image[-10:, :, :] = 0
        image[:, :10, :] = 0
        image[:, -10:, :] = 0
        results = analyzer.analyze(image)
        # After stripping, the effective region is all 200
        assert results[0].mean_r == pytest.approx(200.0, abs=1.0)

    def test_luminance_map_shape(self):
        analyzer = UniformityAnalyzer(rows=3, cols=5)
        image = self._uniform_image(3, 5)
        results = analyzer.analyze(image)
        lmap = analyzer.luminance_map(results)
        assert lmap.shape == (3, 5)

    def test_non_uniform_image_ratio_less_than_one(self):
        analyzer = UniformityAnalyzer(rows=1, cols=2)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100, :] = 200   # left module bright
        image[:, 100:, :] = 100   # right module dim
        results = analyzer.analyze(image)
        ratio = analyzer.compute_uniformity_ratio(results)
        assert ratio < 1.0
        assert ratio == pytest.approx(0.5, abs=0.05)

    def test_black_image_uniformity_ratio(self):
        analyzer = UniformityAnalyzer(rows=2, cols=2)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        results = analyzer.analyze(image)
        ratio = analyzer.compute_uniformity_ratio(results)
        # All luminances are 0 → ratio should be 1.0 (perfectly uniform black)
        assert ratio == pytest.approx(1.0)

    def test_row_col_indices(self):
        analyzer = UniformityAnalyzer(rows=2, cols=3)
        image = self._uniform_image(2, 3)
        results = analyzer.analyze(image)
        positions = [(r.module.row, r.module.col) for r in results]
        expected = [(r, c) for r in range(2) for c in range(3)]
        assert positions == expected
