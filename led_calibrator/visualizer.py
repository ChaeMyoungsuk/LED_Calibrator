"""Visualisation helpers for LED uniformity and calibration results.

The :class:`Visualizer` can produce:

* **Luminance heat-map** – false-colour map of per-module brightness.
* **Gain map** – per-module gain correction factors for each RGB channel.
* **Before/after comparison** – side-by-side comparison of raw and calibrated
  images overlaid with module grid lines.

All methods return Matplotlib :class:`~matplotlib.figure.Figure` objects so
that callers can either save them to disk or display them interactively.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

from .models import CalibrationData, UniformityResult


def _require_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


class Visualizer:
    """Generate uniformity and calibration visualisations.

    Parameters
    ----------
    rows:
        Number of LED module rows in the panel.
    cols:
        Number of LED module columns in the panel.
    figsize:
        Default ``(width_in, height_in)`` for generated figures.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        figsize: Tuple[float, float] = (10.0, 6.0),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_luminance_map(
        self,
        results: List[UniformityResult],
        title: str = "Module Luminance Map",
        cmap: str = "hot",
    ) -> "Figure":
        """Plot a heat-map of per-module luminance values.

        Parameters
        ----------
        results:
            Output of
            :meth:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer.analyze`.
        title:
            Figure title.
        cmap:
            Matplotlib colour-map name.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _require_matplotlib()
        lum_map = np.zeros((self.rows, self.cols), dtype=np.float64)
        for r in results:
            lum_map[r.module.row, r.module.col] = r.luminance

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(lum_map, cmap=cmap, vmin=0, vmax=255)
        plt.colorbar(im, ax=ax, label="Luminance (0–255)")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        self._annotate_grid(ax, lum_map, fmt="{:.1f}")
        fig.tight_layout()
        return fig

    def plot_gain_map(
        self,
        calibrations: List[CalibrationData],
        channel: int = 1,
        title: Optional[str] = None,
        cmap: str = "RdYlGn",
    ) -> "Figure":
        """Plot a heat-map of per-module gain values for one RGB channel.

        Parameters
        ----------
        calibrations:
            Output of
            :meth:`~led_calibrator.calibrator.LEDCalibrator.compute_calibration`.
        channel:
            ``0`` = R, ``1`` = G, ``2`` = B.
        title:
            Figure title.  Defaults to ``"Gain Map – <channel>"`` .
        cmap:
            Matplotlib colour-map name.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _require_matplotlib()
        ch_names = ["R", "G", "B"]
        if title is None:
            title = f"Gain Map – {ch_names[channel]} channel"
        gain_map = np.ones((self.rows, self.cols), dtype=np.float64)
        for cal in calibrations:
            gain_map[cal.module.row, cal.module.col] = cal.gain_rgb[channel]

        fig, ax = plt.subplots(figsize=self.figsize)
        vmax = max(gain_map.max(), 2.0)
        im = ax.imshow(gain_map, cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Gain factor")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        self._annotate_grid(ax, gain_map, fmt="{:.2f}")
        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        original: np.ndarray,
        calibrated: np.ndarray,
        calibrations: Optional[List[CalibrationData]] = None,
        title: str = "Before / After Calibration",
    ) -> "Figure":
        """Show a side-by-side before/after comparison.

        Parameters
        ----------
        original:
            The raw (uncalibrated) ``uint8`` image.
        calibrated:
            The corrected ``uint8`` image.
        calibrations:
            If given, module grid lines are drawn on both images.
        title:
            Overall figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _require_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        for ax, img, label in zip(
            axes, [original, calibrated], ["Original", "Calibrated"]
        ):
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
            if calibrations:
                self._draw_module_grid(ax, calibrations)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def plot_rgb_uniformity(
        self,
        results: List[UniformityResult],
        title: str = "Per-Module RGB Mean Values",
    ) -> "Figure":
        """Bar chart of per-module mean R, G, B values.

        Parameters
        ----------
        results:
            Output of
            :meth:`~led_calibrator.uniformity_analyzer.UniformityAnalyzer.analyze`.
        title:
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _require_matplotlib()
        n = len(results)
        indices = np.arange(n)
        r_vals = np.array([res.mean_r for res in results])
        g_vals = np.array([res.mean_g for res in results])
        b_vals = np.array([res.mean_b for res in results])

        fig, ax = plt.subplots(figsize=self.figsize)
        width = 0.25
        ax.bar(indices - width, r_vals, width, color="red",   label="R")
        ax.bar(indices,         g_vals, width, color="green", label="G")
        ax.bar(indices + width, b_vals, width, color="blue",  label="B")
        ax.set_xlabel("Module index (row-major)")
        ax.set_ylabel("Mean value (0–255)")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 255)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotate_grid(
        self, ax: "Axes", data: np.ndarray, fmt: str = "{:.1f}"
    ) -> None:
        """Annotate each cell of *data* with its value."""
        rows, cols = data.shape
        for r in range(rows):
            for c in range(cols):
                ax.text(
                    c, r, fmt.format(data[r, c]),
                    ha="center", va="center",
                    fontsize=max(4, min(8, 80 // max(rows, cols))),
                    color="white",
                )

    def _draw_module_grid(
        self,
        ax: "Axes",
        calibrations: List[CalibrationData],
    ) -> None:
        """Draw module bounding boxes on *ax*."""
        for cal in calibrations:
            x0, y0, x1, y1 = cal.module.region
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=0.5, edgecolor="cyan", facecolor="none",
            )
            ax.add_patch(rect)
