"""Calibration region matching – align detected regions to LED module positions.

In real-world deployments the captured image may be slightly rotated, skewed,
or otherwise misaligned with the physical LED module grid.  The
:class:`RegionMatcher` takes a list of *detected* bounding boxes (from e.g.
contour detection or an external measurement fixture) and matches each one to
the closest *expected* module position in the ideal grid, returning a mapping
that can be used to correct the module regions before calibration is performed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import LEDModule


def _centre(region: Tuple[int, int, int, int]) -> np.ndarray:
    """Return the (x, y) centre of a bounding-box ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = region
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)


class RegionMatcher:
    """Match detected calibration regions to the ideal LED module grid.

    The *ideal* grid is constructed from the panel dimensions and image size,
    exactly as :class:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer`
    does internally.  Detected regions are then matched to ideal module
    positions using a nearest-centroid approach.

    Parameters
    ----------
    rows:
        Number of LED module rows.
    cols:
        Number of LED module columns.
    image_size:
        ``(width, height)`` of the captured image in pixels.
    max_distance_fraction:
        Maximum allowable distance (as a fraction of the module diagonal)
        between a detected region centre and its matched ideal centre.
        Detections that exceed this threshold are treated as *unmatched*.
        Defaults to ``0.5``.

    Examples
    --------
    >>> matcher = RegionMatcher(rows=2, cols=3, image_size=(300, 200))
    >>> ideal = matcher.ideal_modules()
    >>> len(ideal)
    6
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        image_size: Tuple[int, int],
        max_distance_fraction: float = 0.5,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.rows = rows
        self.cols = cols
        self.image_size = image_size  # (width, height)
        self.max_distance_fraction = max_distance_fraction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ideal_modules(self) -> List[LEDModule]:
        """Return the ideal (evenly tiled) LED module list.

        Returns
        -------
        list[LEDModule]
            Ordered row-major, shape ``rows × cols``.
        """
        w, h = self.image_size
        tile_w = w / self.cols
        tile_h = h / self.rows
        modules: List[LEDModule] = []
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = int(round(c * tile_w))
                y0 = int(round(r * tile_h))
                x1 = int(round((c + 1) * tile_w))
                y1 = int(round((r + 1) * tile_h))
                modules.append(LEDModule(row=r, col=c, region=(x0, y0, x1, y1)))
        return modules

    def match(
        self,
        detected_regions: List[Tuple[int, int, int, int]],
    ) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        """Match each detected region to the nearest ideal module.

        Parameters
        ----------
        detected_regions:
            List of ``(x0, y0, x1, y1)`` bounding boxes for the detected
            calibration patches (e.g. from camera image analysis).

        Returns
        -------
        dict[str, tuple | None]
            Maps ``module.id`` → detected region (or ``None`` if no suitable
            detection was found for that module).
        """
        ideal = self.ideal_modules()
        # Pre-compute ideal centres and the threshold distance
        ideal_centres = np.stack([_centre(m.region) for m in ideal])  # (N, 2)
        diag = self._module_diagonal()
        threshold = diag * self.max_distance_fraction

        # For each ideal module find the closest detected region
        detected_centres = (
            np.stack([_centre(r) for r in detected_regions])
            if detected_regions
            else np.empty((0, 2), dtype=np.float64)
        )

        matched: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
        for idx, module in enumerate(ideal):
            if len(detected_centres) == 0:
                matched[module.id] = None
                continue
            diffs = detected_centres - ideal_centres[idx]  # (M, 2)
            dists = np.linalg.norm(diffs, axis=1)          # (M,)
            best = int(np.argmin(dists))
            if dists[best] <= threshold:
                matched[module.id] = detected_regions[best]
            else:
                matched[module.id] = None
        return matched

    def align_modules(
        self,
        detected_regions: List[Tuple[int, int, int, int]],
    ) -> List[LEDModule]:
        """Return a corrected module list using detected region positions.

        For modules that could be matched, the region is replaced with the
        detected bounding box.  For unmatched modules the ideal region is kept.

        Parameters
        ----------
        detected_regions:
            List of ``(x0, y0, x1, y1)`` bounding boxes.

        Returns
        -------
        list[LEDModule]
            Corrected module list, same length and ordering as
            :meth:`ideal_modules`.
        """
        mapping = self.match(detected_regions)
        aligned: List[LEDModule] = []
        for module in self.ideal_modules():
            region = mapping.get(module.id)
            if region is not None:
                aligned.append(LEDModule(row=module.row, col=module.col, region=region))
            else:
                aligned.append(module)
        return aligned

    def compute_alignment_error(
        self,
        detected_regions: List[Tuple[int, int, int, int]],
    ) -> float:
        """Return the mean centre-to-centre alignment error in pixels.

        Only matched pairs are included.  Returns ``0.0`` if there are no
        matched pairs.

        Parameters
        ----------
        detected_regions:
            List of detected ``(x0, y0, x1, y1)`` bounding boxes.

        Returns
        -------
        float
            Mean pixel distance between ideal and matched detected centres.
        """
        mapping = self.match(detected_regions)
        ideal = {m.id: m for m in self.ideal_modules()}
        errors: List[float] = []
        for mid, det_region in mapping.items():
            if det_region is None:
                continue
            ic = _centre(ideal[mid].region)
            dc = _centre(det_region)
            errors.append(float(np.linalg.norm(dc - ic)))
        return float(np.mean(errors)) if errors else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _module_diagonal(self) -> float:
        """Return the diagonal length of one ideal module tile in pixels."""
        w, h = self.image_size
        tw = w / self.cols
        th = h / self.rows
        return float(np.sqrt(tw ** 2 + th ** 2))
