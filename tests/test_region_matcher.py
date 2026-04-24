"""Tests for led_calibrator.region_matcher."""

import numpy as np
import pytest

from led_calibrator.region_matcher import RegionMatcher, _centre


class TestCentreHelper:
    def test_simple(self):
        c = _centre((0, 0, 100, 100))
        np.testing.assert_array_almost_equal(c, [50.0, 50.0])

    def test_offset_region(self):
        c = _centre((10, 20, 30, 60))
        np.testing.assert_array_almost_equal(c, [20.0, 40.0])


class TestRegionMatcher:
    def test_ideal_modules_count(self):
        matcher = RegionMatcher(rows=3, cols=4, image_size=(400, 300))
        assert len(matcher.ideal_modules()) == 12

    def test_ideal_modules_tile_image(self):
        w, h = 600, 400
        matcher = RegionMatcher(rows=2, cols=3, image_size=(w, h))
        modules = matcher.ideal_modules()
        # Check that all module regions are within image bounds
        for m in modules:
            x0, y0, x1, y1 = m.region
            assert x0 >= 0 and y0 >= 0
            assert x1 <= w and y1 <= h

    def test_perfect_detection_gives_identity_match(self):
        """Detected regions at ideal positions should all match."""
        w, h = 300, 200
        matcher = RegionMatcher(rows=2, cols=3, image_size=(w, h))
        ideal = matcher.ideal_modules()
        detected = [m.region for m in ideal]
        mapping = matcher.match(detected)
        assert all(v is not None for v in mapping.values())

    def test_no_detections_gives_none_mapping(self):
        matcher = RegionMatcher(rows=2, cols=2, image_size=(200, 200))
        mapping = matcher.match([])
        assert all(v is None for v in mapping.values())

    def test_far_detection_is_unmatched(self):
        """A detected region that is far from any ideal module should not match."""
        matcher = RegionMatcher(rows=1, cols=1, image_size=(100, 100), max_distance_fraction=0.1)
        # Ideal centre is at (50, 50); put detection far away
        far_region = (800, 800, 900, 900)
        mapping = matcher.match([far_region])
        assert list(mapping.values())[0] is None

    def test_align_modules_replaces_matched(self):
        """align_modules should replace matched regions with detected ones."""
        w, h = 200, 100
        matcher = RegionMatcher(rows=1, cols=2, image_size=(w, h))
        # Slightly shift the ideal regions
        ideal = matcher.ideal_modules()
        shifted = [(r[0] + 2, r[1] + 2, r[2] + 2, r[3] + 2) for r in [m.region for m in ideal]]
        aligned = matcher.align_modules(shifted)
        for orig, al, det in zip(ideal, aligned, shifted):
            assert al.region == det  # replaced by detected

    def test_alignment_error_is_zero_for_ideal(self):
        matcher = RegionMatcher(rows=2, cols=2, image_size=(200, 200))
        ideal = [m.region for m in matcher.ideal_modules()]
        error = matcher.compute_alignment_error(ideal)
        assert error == pytest.approx(0.0, abs=1.0)

    def test_alignment_error_non_zero_for_shifted(self):
        matcher = RegionMatcher(rows=2, cols=2, image_size=(200, 200))
        ideal = matcher.ideal_modules()
        shifted = [(r[0] + 5, r[1] + 5, r[2] + 5, r[3] + 5) for r in [m.region for m in ideal]]
        error = matcher.compute_alignment_error(shifted)
        # Each centre is shifted by sqrt(25+25) ≈ 7.07
        assert error == pytest.approx(np.sqrt(50), rel=0.05)

    def test_invalid_rows_cols(self):
        with pytest.raises(ValueError):
            RegionMatcher(rows=0, cols=2, image_size=(200, 200))
        with pytest.raises(ValueError):
            RegionMatcher(rows=2, cols=0, image_size=(200, 200))

    def test_mapping_keys_are_module_ids(self):
        matcher = RegionMatcher(rows=2, cols=2, image_size=(200, 200))
        ideal = matcher.ideal_modules()
        mapping = matcher.match([m.region for m in ideal])
        for m in ideal:
            assert m.id in mapping
