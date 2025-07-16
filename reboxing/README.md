# Image Compositing Tool

This tool reads JSON files containing image placement data and composites neural render images onto blank composition images.

## JSON File Structure

Each JSON file contains:
- `proxy_comp_width` and `proxy_comp_height`: Size of the blank composition image
- `neural_render_width` and `neural_render_height`: Size of neural render images (1024x1024)
- `c0`, `c1`, `c2`, `c3`: Corner coordinates for placing the neural render image

Example JSON:
```json
{
  "proxy_comp_width": 2048,
  "proxy_comp_height": 858,
  "neural_render_width": 1024,
  "neural_render_height": 1024,
  "c0": [1211.97, 219.40],
  "c1": [1831.47, 197.91],
  "c2": [1852.96, 817.41],
  "c3": [1233.46, 838.90]
}
```

## Files

- `composite_images.py`: Full implementation with OpenCV and PIL for image processing
- `composite_images_simple.py`: Simplified version using only standard library
- `requirements.txt`: Dependencies for the full implementation

## Installation

For the full implementation with image processing:
```bash
pip install -r requirements.txt
```

## Usage

### Full Implementation (with image processing)

```python
from composite_images import process_folder_batch

# Process all files in a folder
process_folder_batch(
    json_folder="path/to/json/files",
    neural_render_folder="path/to/neural/render/images",
    output_folder="path/to/output",
    blank_image_path="path/to/blank/image.png"  # Optional
)
```

### Command Line Usage

```bash
python composite_images.py \
    --json_folder data/rebox/vmdf02_ep01_pt03_0050 \
    --neural_render_folder path/to/neural/render/images \
    --output_folder output/composited \
    --blank_image path/to/blank/image.png
```

### Simple Implementation (no external dependencies)

```python
from composite_images_simple import process_folder_batch_simple

# Process all files in a folder
process_folder_batch_simple(
    json_folder="path/to/json/files",
    neural_render_folder="path/to/neural/render/images",
    output_folder="path/to/output"
)
```

## How It Works

1. **Reads JSON files**: Extracts composition dimensions and corner coordinates
2. **Loads neural render images**: Assumes same naming pattern as JSON files (e.g., `000001.json` → `000001.png`)
3. **Creates blank image**: Either loads provided blank image or creates white image
4. **Applies perspective transform**: Uses corner coordinates to warp neural render image
5. **Composites images**: Blends warped neural render onto blank image using mask
6. **Saves result**: Outputs composited image with same naming as input

## Key Features

- **Perspective transformation**: Uses OpenCV to warp neural render images to match corner coordinates
- **Masked compositing**: Only composites within the specified quadrilateral area
- **Batch processing**: Processes entire folders of JSON files
- **Error handling**: Skips missing files and provides warnings
- **Flexible input**: Optional blank image template or creates white background

## Example Workflow

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your JSON files with placement data
3. Ensure neural render images match JSON file names
4. Run the compositing function
5. Check output folder for composited images

## Notes

- Neural render images should be 1024x1024 pixels
- JSON files should be named with 6-digit numbers (e.g., `000001.json`)
- Corresponding neural render images should have same names with `.png` extension
- The function creates output directories automatically