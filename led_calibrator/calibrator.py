"""LED calibration – compute and apply per-module RGB corrections.

The :class:`LEDCalibrator` takes the uniformity measurements produced by
:class:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer` and calculates
per-module gain + offset coefficients that drive every module toward a common
luminance and white-point target.  The coefficients can then be applied to an
arbitrary input image to produce a uniformity-corrected output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .models import LEDModule, UniformityResult, CalibrationData


class LEDCalibrator:
    """Compute and apply RGB gain/offset corrections for LED uniformity.

    The correction model is::

        corrected = clip(gain * raw + offset, 0, 255)

    Gain and offset are chosen per channel so that the *corrected* mean of
    each module equals the *target* value.  For the luminance channel the
    target is the panel-level mean (or a user-supplied value); for R and G
    and B the target is derived from the target white point.

    Parameters
    ----------
    target_white_point:
        Desired ``(R, G, B)`` target for full-white in the range ``[0, 255]``.
        Defaults to ``(255, 255, 255)`` (pure white).
    target_luminance:
        If given, overrides the luminance target used when computing gains.
        When ``None`` (default) the panel-average luminance is used.

    Examples
    --------
    >>> import numpy as np
    >>> from led_calibrator import UniformityAnalyzer, LEDCalibrator
    >>> analyzer = UniformityAnalyzer(rows=2, cols=2)
    >>> image = np.random.randint(180, 220, (200, 200, 3), dtype=np.uint8)
    >>> results = analyzer.analyze(image)
    >>> calibrator = LEDCalibrator()
    >>> cal_data = calibrator.compute_calibration(results)
    >>> len(cal_data)
    4
    """

    def __init__(
        self,
        target_white_point: Tuple[float, float, float] = (255.0, 255.0, 255.0),
        target_luminance: Optional[float] = None,
    ) -> None:
        self.target_white_point = np.array(target_white_point, dtype=np.float64)
        self.target_luminance = target_luminance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_calibration(
        self, results: List[UniformityResult]
    ) -> List[CalibrationData]:
        """Calculate per-module correction coefficients.

        The gain for each RGB channel is::

            gain_ch = target_ch / mean_ch        (if mean_ch > 0, else 1.0)

        The offset is always set to ``0.0`` (pure multiplicative correction).
        For modules where the measured value is zero the gain is clamped to
        ``1.0`` to avoid division by zero.

        Parameters
        ----------
        results:
            Output of
            :meth:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer.analyze`.

        Returns
        -------
        list[CalibrationData]
            One :class:`~led_calibrator.models.CalibrationData` per module.
        """
        target = self._resolve_target(results)
        calibrations: List[CalibrationData] = []
        for r in results:
            safe_mean = np.where(r.mean_rgb > 0.0, r.mean_rgb, 1.0)
            gain = np.where(r.mean_rgb > 0.0, target / safe_mean, 1.0)
            # Clamp gain to a sane range to avoid extreme corrections
            gain = np.clip(gain, 0.1, 10.0)
            offset = np.zeros(3, dtype=np.float64)
            calibrations.append(
                CalibrationData(
                    module=r.module,
                    gain_rgb=gain,
                    offset_rgb=offset,
                    target_luminance=float(np.dot(target / 255.0, np.array([0.2126, 0.7152, 0.0722])) * 255.0),
                )
            )
        return calibrations

    def apply_calibration(
        self,
        image: np.ndarray,
        calibrations: List[CalibrationData],
    ) -> np.ndarray:
        """Apply pre-computed corrections to *image*.

        Each module's region in the image is multiplied by the corresponding
        gain (and offset added) independently.

        Parameters
        ----------
        image:
            ``uint8`` NumPy array of shape ``(H, W, 3)``.
        calibrations:
            Output of :meth:`compute_calibration`.

        Returns
        -------
        np.ndarray
            Corrected image of the same shape, dtype ``uint8``.
        """
        out = image.astype(np.float64).copy()
        for cal in calibrations:
            x0, y0, x1, y1 = cal.module.region
            out[y0:y1, x0:x1] = cal.apply(out[y0:y1, x0:x1])
        return np.clip(out, 0, 255).astype(np.uint8)

    def uniformity_improvement(
        self,
        before: List[UniformityResult],
        after_image: np.ndarray,
        analyzer: "UniformityAnalyzer",  # type: ignore[name-defined]
    ) -> float:
        """Return the percentage improvement in uniformity ratio after calibration.

        Parameters
        ----------
        before:
            Uniformity results from the *uncalibrated* image.
        after_image:
            The calibrated image (output of :meth:`apply_calibration`).
        analyzer:
            The same :class:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer`
            used for the original measurement.

        Returns
        -------
        float
            Improvement in percentage points (can be negative if calibration
            made things worse).
        """
        from .uniformity_analyzer import UniformityAnalyzer  # local import to avoid circular

        after = analyzer.analyze(after_image)
        ratio_before = analyzer.compute_uniformity_ratio(before)
        ratio_after = analyzer.compute_uniformity_ratio(after)
        return (ratio_after - ratio_before) * 100.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_target(self, results: List[UniformityResult]) -> np.ndarray:
        """Determine the per-channel calibration target.

        If ``target_luminance`` was supplied at construction the target is
        scaled to achieve that luminance; otherwise the panel-mean per-channel
        value is used as target (so that the overall brightness is preserved
        while equalising variation).
        """
        mean_values = np.stack([r.mean_rgb for r in results])  # (N, 3)
        panel_mean = mean_values.mean(axis=0)  # (3,)

        if self.target_luminance is not None:
            # Scale panel mean to achieve the requested luminance
            lum_coeff = np.array([0.2126, 0.7152, 0.0722])
            current_lum = float(np.dot(panel_mean / 255.0, lum_coeff) * 255.0)
            if current_lum > 0.0:
                scale = self.target_luminance / current_lum
                target = np.clip(panel_mean * scale, 0.0, 255.0)
            else:
                target = panel_mean.copy()
        else:
            target = panel_mean.copy()

        return target
