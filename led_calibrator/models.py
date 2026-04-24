"""Data models used throughout LED_Calibrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class LEDModule:
    """Represents a single LED module on the display panel.

    Parameters
    ----------
    row:
        Zero-based row index of the module in the panel grid.
    col:
        Zero-based column index of the module in the panel grid.
    region:
        Pixel bounding box ``(x_min, y_min, x_max, y_max)`` of the module
        within the captured image.
    """

    row: int
    col: int
    region: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    @property
    def id(self) -> str:
        return f"module_r{self.row:03d}_c{self.col:03d}"

    @property
    def width(self) -> int:
        return self.region[2] - self.region[0]

    @property
    def height(self) -> int:
        return self.region[3] - self.region[1]


@dataclass
class UniformityResult:
    """Holds raw uniformity measurements for one LED module.

    Parameters
    ----------
    module:
        The :class:`LEDModule` these measurements belong to.
    mean_rgb:
        Average ``(R, G, B)`` values measured inside the module region
        (each component in the range ``[0.0, 255.0]``).
    std_rgb:
        Standard deviations of ``(R, G, B)`` inside the module region.
    luminance:
        Estimated luminance (CIE Y) of the module, derived from ``mean_rgb``.
    """

    module: LEDModule
    mean_rgb: np.ndarray   # shape (3,)
    std_rgb: np.ndarray    # shape (3,)
    luminance: float

    @property
    def mean_r(self) -> float:
        return float(self.mean_rgb[0])

    @property
    def mean_g(self) -> float:
        return float(self.mean_rgb[1])

    @property
    def mean_b(self) -> float:
        return float(self.mean_rgb[2])


@dataclass
class CalibrationData:
    """Stores the per-module RGB correction coefficients.

    The correction model is a simple gain + offset applied per channel::

        corrected_value = gain * raw_value + offset

    Parameters
    ----------
    module:
        The :class:`LEDModule` these corrections apply to.
    gain_rgb:
        Per-channel gain factors ``(gain_R, gain_G, gain_B)``.
        A value of ``1.0`` means no correction.
    offset_rgb:
        Per-channel additive offsets ``(offset_R, offset_G, offset_B)``.
        A value of ``0.0`` means no correction.
    target_luminance:
        The target luminance that the gain/offset were computed to reach.
    """

    module: LEDModule
    gain_rgb: np.ndarray    # shape (3,)
    offset_rgb: np.ndarray  # shape (3,)
    target_luminance: float

    def apply(self, pixel: np.ndarray) -> np.ndarray:
        """Apply the correction to *pixel* ``(R, G, B)`` values.

        Parameters
        ----------
        pixel:
            Array of shape ``(..., 3)`` with values in ``[0, 255]``.

        Returns
        -------
        np.ndarray
            Corrected pixel values, clipped to ``[0, 255]``.
        """
        corrected = pixel * self.gain_rgb + self.offset_rgb
        return np.clip(corrected, 0.0, 255.0)

    @property
    def gain_r(self) -> float:
        return float(self.gain_rgb[0])

    @property
    def gain_g(self) -> float:
        return float(self.gain_rgb[1])

    @property
    def gain_b(self) -> float:
        return float(self.gain_rgb[2])
