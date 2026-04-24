#!/usr/bin/env python3
"""LED Calibrator – command-line entry point.

Usage
-----
Run with ``--help`` to see all options::

    python main.py --help

Basic workflow::

    # Analyse a white-field capture and produce a calibrated output
    python main.py calibrate \\
        --image white_field.png \\
        --rows 4 --cols 6 \\
        --output calibrated.png \\
        --report report.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _load_image(path: str) -> np.ndarray:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _save_image(array: np.ndarray, path: str) -> None:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    Image.fromarray(array.astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Sub-command: calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run the full calibration pipeline on a white-field image."""
    from led_calibrator import UniformityAnalyzer, LEDCalibrator, Visualizer

    print(f"[LED Calibrator] Loading image: {args.image}")
    image = _load_image(args.image)

    rows: int = args.rows
    cols: int = args.cols
    print(f"[LED Calibrator] Panel grid: {rows} rows × {cols} cols")

    # 1 – Analyse uniformity
    analyzer = UniformityAnalyzer(rows=rows, cols=cols)
    results = analyzer.analyze(image)
    ratio = analyzer.compute_uniformity_ratio(results)
    print(f"[LED Calibrator] Uniformity ratio (before): {ratio:.4f} ({ratio * 100:.2f}%)")

    # 2 – Compute calibration
    calibrator = LEDCalibrator(
        target_luminance=args.target_luminance,
    )
    calibrations = calibrator.compute_calibration(results)

    # 3 – Apply calibration
    calibrated = calibrator.apply_calibration(image, calibrations)

    # 4 – Measure improvement
    after_results = analyzer.analyze(calibrated)
    ratio_after = analyzer.compute_uniformity_ratio(after_results)
    print(f"[LED Calibrator] Uniformity ratio (after):  {ratio_after:.4f} ({ratio_after * 100:.2f}%)")
    improvement = (ratio_after - ratio) * 100.0
    print(f"[LED Calibrator] Improvement: {improvement:+.2f} percentage points")

    # 5 – Save calibrated image
    if args.output:
        _save_image(calibrated, args.output)
        print(f"[LED Calibrator] Calibrated image saved to: {args.output}")

    # 6 – Save visualisation report
    if args.report:
        try:
            import matplotlib
            matplotlib.use("Agg")
            viz = Visualizer(rows=rows, cols=cols)
            fig = viz.plot_luminance_map(results, title="Luminance Map (before)")
            fig.savefig(args.report, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"[LED Calibrator] Report saved to: {args.report}")
        except ImportError:
            print("[LED Calibrator] Warning: matplotlib not installed – skipping report.")

    return 0


# ---------------------------------------------------------------------------
# Sub-command: analyse
# ---------------------------------------------------------------------------

def cmd_analyse(args: argparse.Namespace) -> int:
    """Print per-module uniformity statistics without applying corrections."""
    from led_calibrator import UniformityAnalyzer

    image = _load_image(args.image)
    analyzer = UniformityAnalyzer(rows=args.rows, cols=args.cols)
    results = analyzer.analyze(image)
    ratio = analyzer.compute_uniformity_ratio(results)

    print(f"Panel grid:        {args.rows} × {args.cols}")
    print(f"Uniformity ratio:  {ratio:.4f}  ({ratio * 100:.2f}%)")
    print()
    header = f"{'Module':<20}  {'R':>7}  {'G':>7}  {'B':>7}  {'Luminance':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.module.id:<20}  "
            f"{r.mean_r:7.2f}  {r.mean_g:7.2f}  {r.mean_b:7.2f}  "
            f"{r.luminance:10.2f}"
        )
    return 0


# ---------------------------------------------------------------------------
# Sub-command: match-regions
# ---------------------------------------------------------------------------

def cmd_match_regions(args: argparse.Namespace) -> int:
    """Match detected calibration regions to ideal module positions."""
    from led_calibrator import RegionMatcher

    matcher = RegionMatcher(
        rows=args.rows,
        cols=args.cols,
        image_size=(args.image_width, args.image_height),
    )
    ideal = matcher.ideal_modules()
    print(f"Ideal module grid ({args.rows} × {args.cols}):")
    for m in ideal:
        print(f"  {m.id}: region={m.region}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="led_calibrator",
        description="Professional LED panel white-uniformity calibration tool.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate a white-field image.")
    p_cal.add_argument("--image", required=True, help="Path to the white-field input image.")
    p_cal.add_argument("--rows", type=int, required=True, help="Number of LED module rows.")
    p_cal.add_argument("--cols", type=int, required=True, help="Number of LED module columns.")
    p_cal.add_argument("--output", default=None, help="Path to save the calibrated image.")
    p_cal.add_argument("--report", default=None, help="Path to save the visualisation report (PNG).")
    p_cal.add_argument(
        "--target-luminance",
        type=float,
        default=None,
        dest="target_luminance",
        help="Target luminance (0–255). Defaults to panel average.",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    # analyse
    p_ana = sub.add_parser("analyse", help="Print per-module uniformity statistics.")
    p_ana.add_argument("--image", required=True, help="Path to the white-field input image.")
    p_ana.add_argument("--rows", type=int, required=True, help="Number of LED module rows.")
    p_ana.add_argument("--cols", type=int, required=True, help="Number of LED module columns.")
    p_ana.set_defaults(func=cmd_analyse)

    # match-regions
    p_mr = sub.add_parser("match-regions", help="Show ideal module regions for a given panel.")
    p_mr.add_argument("--rows", type=int, required=True)
    p_mr.add_argument("--cols", type=int, required=True)
    p_mr.add_argument("--image-width", type=int, required=True, dest="image_width")
    p_mr.add_argument("--image-height", type=int, required=True, dest="image_height")
    p_mr.set_defaults(func=cmd_match_regions)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
