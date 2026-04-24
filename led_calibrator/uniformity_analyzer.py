"""White uniformity analysis for individual LED modules.

The :class:`UniformityAnalyzer` divides a captured white-field image into a
grid of LED module regions and computes per-module colour statistics that are
then used by :class:`~led_calibrator.calibrator.LEDCalibrator` to calculate
correction coefficients.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .models import LEDModule, UniformityResult


# sRGB → CIE-Y luminance coefficients (ITU-R BT.709)
_LUM_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _luminance(rgb: np.ndarray) -> float:
    """Return the linear luminance estimate from an ``(R, G, B)`` triplet.

    Values are assumed to be in the range ``[0, 255]``.
    """
    return float(np.dot(rgb / 255.0, _LUM_COEFF) * 255.0)


class UniformityAnalyzer:
    """Analyse the white uniformity of an LED panel from a captured image.

    The panel is modelled as a rectangular grid of *rows* × *cols* LED modules.
    Each module occupies an equal-sized tile of the input image (after optional
    padding is removed).

    Parameters
    ----------
    rows:
        Number of LED module rows in the panel.
    cols:
        Number of LED module columns in the panel.
    image_padding:
        Optional ``(top, bottom, left, right)`` pixel padding to strip from the
        image before partitioning into module tiles.  Defaults to no padding.

    Examples
    --------
    >>> import numpy as np
    >>> analyzer = UniformityAnalyzer(rows=4, cols=6)
    >>> image = np.full((400, 600, 3), 200, dtype=np.uint8)
    >>> results = analyzer.analyze(image)
    >>> len(results)
    24
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        image_padding: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.rows = rows
        self.cols = cols
        self.image_padding: Tuple[int, int, int, int] = (
            image_padding if image_padding is not None else (0, 0, 0, 0)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, image: np.ndarray) -> List[UniformityResult]:
        """Compute per-module uniformity statistics.

        Parameters
        ----------
        image:
            ``uint8`` or ``float`` NumPy array of shape ``(H, W, 3)``
            representing an RGB white-field capture of the LED panel.

        Returns
        -------
        list[UniformityResult]
            One :class:`~led_calibrator.models.UniformityResult` per module,
            ordered row-major (left-to-right, top-to-bottom).
        """
        img = self._preprocess(image)
        modules = self._build_module_grid(img.shape)
        results: List[UniformityResult] = []
        for module in modules:
            x0, y0, x1, y1 = module.region
            tile = img[y0:y1, x0:x1].astype(np.float64)
            mean_rgb = tile.reshape(-1, 3).mean(axis=0)
            std_rgb = tile.reshape(-1, 3).std(axis=0)
            lum = _luminance(mean_rgb)
            results.append(
                UniformityResult(
                    module=module,
                    mean_rgb=mean_rgb,
                    std_rgb=std_rgb,
                    luminance=lum,
                )
            )
        return results

    def compute_uniformity_ratio(
        self, results: List[UniformityResult]
    ) -> float:
        """Return the panel-level white uniformity ratio.

        The ratio is defined as::

            uniformity = min_luminance / max_luminance

        A value of ``1.0`` means perfect uniformity; lower values indicate
        greater variation across modules.

        Parameters
        ----------
        results:
            Output of :meth:`analyze`.

        Returns
        -------
        float
            Uniformity ratio in ``[0, 1]``.
        """
        luminances = np.array([r.luminance for r in results], dtype=np.float64)
        max_lum = luminances.max()
        if max_lum == 0.0:
            return 1.0
        return float(luminances.min() / max_lum)

    def luminance_map(
        self, results: List[UniformityResult]
    ) -> np.ndarray:
        """Return a ``(rows, cols)`` array of per-module luminance values.

        Parameters
        ----------
        results:
            Output of :meth:`analyze`.

        Returns
        -------
        np.ndarray
            Shape ``(self.rows, self.cols)``, dtype ``float64``.
        """
        lum_map = np.zeros((self.rows, self.cols), dtype=np.float64)
        for r in results:
            lum_map[r.module.row, r.module.col] = r.luminance
        return lum_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Strip padding and normalise image dtype."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected an (H, W, 3) array, got shape {image.shape}"
            )
        img = np.asarray(image, dtype=np.float64)
        top, bottom, left, right = self.image_padding
        h, w = img.shape[:2]
        y0 = top
        y1 = h - bottom if bottom > 0 else h
        x0 = left
        x1 = w - right if right > 0 else w
        return img[y0:y1, x0:x1]

    def _build_module_grid(
        self, shape: Tuple[int, ...]
    ) -> List[LEDModule]:
        """Create the LED module grid from the (cropped) image dimensions."""
        h, w = shape[:2]
        tile_h = h / self.rows
        tile_w = w / self.cols
        modules: List[LEDModule] = []
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = int(round(c * tile_w))
                y0 = int(round(r * tile_h))
                x1 = int(round((c + 1) * tile_w))
                y1 = int(round((r + 1) * tile_h))
                modules.append(LEDModule(row=r, col=c, region=(x0, y0, x1, y1)))
        return modules
