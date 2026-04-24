# LED_Calibrator

Professional image-quality tuning program specialised in LED panel white-uniformity calibration and calibration-region alignment.

## Features

| Feature | Description |
|---------|-------------|
| **White Uniformity Analysis** | Divides a captured white-field image into a grid of LED module tiles and measures per-module mean RGB and luminance values. |
| **Calibration** | Computes per-module RGB gain/offset correction coefficients that drive every module toward a common brightness target. |
| **Region Matching** | Aligns detected calibration regions (e.g. from a camera fixture) to the ideal LED module grid using a nearest-centroid algorithm. |
| **Visualisation** | Renders luminance heat-maps, gain maps, and before/after comparisons using Matplotlib. |

## Architecture

```
led_calibrator/
├── __init__.py            – public API
├── models.py              – LEDModule, UniformityResult, CalibrationData
├── uniformity_analyzer.py – white uniformity analysis
├── calibrator.py          – compute & apply RGB corrections
├── region_matcher.py      – match detected regions to ideal grid
└── visualizer.py          – Matplotlib-based visualisation helpers
main.py                    – CLI entry point
tests/                     – pytest test suite
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Command-Line Usage

### Calibrate a white-field image

```bash
python main.py calibrate \
    --image white_field.png \
    --rows 4 --cols 6 \
    --output calibrated.png \
    --report report.png
```

### Print per-module uniformity statistics

```bash
python main.py analyse \
    --image white_field.png \
    --rows 4 --cols 6
```

### Show ideal module regions for a panel

```bash
python main.py match-regions \
    --rows 4 --cols 6 \
    --image-width 1920 --image-height 1080
```

## Python API

```python
import numpy as np
from led_calibrator import UniformityAnalyzer, LEDCalibrator, RegionMatcher, Visualizer

# Load your white-field capture as a NumPy uint8 array (H, W, 3)
image = ...  # e.g. np.asarray(PIL.Image.open("white_field.png").convert("RGB"))

# 1 – Analyse white uniformity
analyzer = UniformityAnalyzer(rows=4, cols=6)
results = analyzer.analyze(image)
print(f"Uniformity ratio: {analyzer.compute_uniformity_ratio(results):.4f}")

# 2 – Compute calibration coefficients
calibrator = LEDCalibrator()
calibrations = calibrator.compute_calibration(results)

# 3 – Apply corrections to produce a uniform output
calibrated_image = calibrator.apply_calibration(image, calibrations)

# 4 – Visualise
viz = Visualizer(rows=4, cols=6)
fig = viz.plot_luminance_map(results)
fig.savefig("luminance_map.png", dpi=150)

fig2 = viz.plot_gain_map(calibrations, channel=1)  # green channel
fig2.savefig("gain_map_green.png", dpi=150)

fig3 = viz.plot_comparison(image, calibrated_image, calibrations)
fig3.savefig("comparison.png", dpi=150)

# 5 – Region matching (for physical alignment)
matcher = RegionMatcher(rows=4, cols=6, image_size=(image.shape[1], image.shape[0]))
ideal_modules = matcher.ideal_modules()
detected_regions = [...]  # (x0, y0, x1, y1) bounding boxes from camera
aligned_modules = matcher.align_modules(detected_regions)
error_px = matcher.compute_alignment_error(detected_regions)
print(f"Mean alignment error: {error_px:.2f} px")
```

## Calibration Model

The correction applied to each module is:

```
corrected_value = clip(gain * raw_value + offset, 0, 255)
```

Gains are computed per RGB channel so that each module's mean equals the panel-average
(or a user-specified target). Gains are clamped to `[0.1, 10.0]` to prevent extreme corrections.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
