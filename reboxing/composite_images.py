import os
import json
import numpy as np
import cv2
import imageio
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def process_single_image(
    json_file: str,
    json_folder: str,
    neural_render_folder: str,
    output_folder: str,
    blank_image_path: Optional[str] = None,
    soften_edges: bool = False
) -> bool:
    """
    Process a single JSON file and composite the corresponding neural render image.
    
    Args:
        json_file: Name of the JSON file to process
        json_folder: Path to folder containing JSON files
        neural_render_folder: Path to folder containing neural render images
        output_folder: Path to save composited images
        blank_image_path: Optional path to a blank image
        soften_edges: Whether to apply edge softening
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        json_path = os.path.join(json_folder, json_file)
        
        # Read JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract dimensions and coordinates
        comp_width = data['proxy_comp_width']
        comp_height = data['proxy_comp_height']
        neural_render_width = data['neural_render_width']
        neural_render_height = data['neural_render_height']
        
        # Extract corner coordinates
        c0 = data['c0']  # [x, y]
        c1 = data['c1']  # [x, y]
        c2 = data['c2']  # [x, y]
        c3 = data['c3']  # [x, y]
        
        # Create or load blank image (BLACK canvas instead of white)
        if blank_image_path and os.path.exists(blank_image_path):
            blank_img = cv2.imread(blank_image_path)
            if blank_img is None:
                raise ValueError(f"Could not load image: {blank_image_path}")
            # Convert BGR to RGB
            blank_img = cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB)
            # Resize to match composition dimensions if needed
            if blank_img.shape[1] != comp_width or blank_img.shape[0] != comp_height:
                blank_img = cv2.resize(blank_img, (comp_width, comp_height), interpolation=cv2.INTER_LINEAR)
        else:
            # Create blank BLACK image (like Nuke's default)
            blank_img = np.zeros((comp_height, comp_width, 3), dtype=np.uint8)
        
        # Load corresponding neural render image
        neural_render_filename = json_file.replace('.json', '.png')
        neural_render_path = os.path.join(neural_render_folder, neural_render_filename)
        
        if not os.path.exists(neural_render_path):
            print(f"Warning: Neural render image not found: {neural_render_path}")
            return False
        
        neural_render_img = cv2.imread(neural_render_path)
        if neural_render_img is None:
            print(f"Warning: Neural render image not found: {neural_render_path}")
            return False
        
        # Convert BGR to RGB
        neural_render_img = cv2.cvtColor(neural_render_img, cv2.COLOR_BGR2RGB)
        
        # Verify neural render image dimensions
        if neural_render_img.shape[1] != neural_render_width or neural_render_img.shape[0] != neural_render_height:
            print(f"Warning: Neural render image size mismatch. Expected {neural_render_width}x{neural_render_height}, got {neural_render_img.shape[1]}x{neural_render_img.shape[0]}")
            neural_render_img = cv2.resize(neural_render_img, (neural_render_width, neural_render_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert to numpy arrays for OpenCV operations
        blank_array = blank_img
        neural_render_array = neural_render_img
        
        # Apply anti-aliasing to neural render image before transformation
        neural_render_array = cv2.GaussianBlur(neural_render_array, (3, 3), 0.5)
        
        # Apply additional edge softening if enabled
        if soften_edges:
            # More aggressive blur for better compositing
            neural_render_array = cv2.GaussianBlur(neural_render_array, (5, 5), 1.0)
            # Apply bilateral filter to preserve edges while smoothing
            neural_render_uint8 = neural_render_array.astype(np.uint8)
            neural_render_array = cv2.bilateralFilter(neural_render_uint8, 9, 75, 75)
        
        # Create perspective transform matrix
        src_points = np.float32([
            [0, 0],  # Top-left
            [neural_render_width, 0],  # Top-right
            [neural_render_width, neural_render_height],  # Bottom-right
            [0, neural_render_height]  # Bottom-left
        ])
        
        # Destination points (coordinates from JSON)
        dst_points = np.float32([c0, c1, c2, c3])
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform to neural render image with INTER_CUBIC for smoother edges
        warped_neural_render = cv2.warpPerspective(
            neural_render_array, 
            transform_matrix, 
            (comp_width, comp_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Create smooth mask for the warped neural render image
        mask = np.zeros((comp_height, comp_width), dtype=np.uint8)
        cv2.fillPoly(mask, [dst_points.astype(np.int32)], 255)
        
        # Apply Gaussian blur to the mask for smoother edges (like Nuke's edge softening)
        mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
        
        # Apply additional mask softening if edge softening is enabled
        if soften_edges:
            # More aggressive mask blur for better compositing
            mask = cv2.GaussianBlur(mask, (11, 11), 2.5)
        
        # Normalize mask to 0-1 range
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        
        # Blend images using the smooth mask (premultiplied alpha compositing like Nuke)
        composite = blank_array * (1 - mask_3d) + warped_neural_render * mask_3d
        
        # Apply additional edge smoothing to the final composite
        composite = cv2.GaussianBlur(composite, (3, 3), 0.3)
        
        # Apply final edge softening if enabled
        if soften_edges:
            # Convert to uint8 for bilateral filtering
            composite_uint8 = composite.astype(np.uint8)
            # Final bilateral filter for better compositing quality
            composite = cv2.bilateralFilter(composite_uint8, 9, 50, 50)
        
        # Convert RGB to BGR for OpenCV saving
        composite_bgr = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_filename = json_file.replace('.json', '.png')
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, composite_bgr)
        
        return True
        
    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")
        return False


def composite_neural_render_images(
    json_folder: str,
    neural_render_folder: str,
    output_folder: str,
    blank_image_path: Optional[str] = None,
    soften_edges: bool = False,
    num_processes: int = None
) -> None:
    """
    Read JSON files containing placement data and composite neural render images onto blank images.
    
    Args:
        json_folder: Path to folder containing JSON files with placement data
        neural_render_folder: Path to folder containing neural render images (1024x1024)
        output_folder: Path to save composited images
        blank_image_path: Optional path to a blank image. If None, creates a blank image
                         based on comp_width/height from JSON
        soften_edges: Whether to apply edge softening
        num_processes: Number of processes to use for parallel processing. If None, uses all available cores.
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get sorted list of JSON files
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
    
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_image,
        json_folder=json_folder,
        neural_render_folder=neural_render_folder,
        output_folder=output_folder,
        blank_image_path=blank_image_path,
        soften_edges=soften_edges
    )
    
    # Process files in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, json_files),
            total=len(json_files),
            desc="Compositing images",
            unit="file"
        ))
    
    # Report results
    successful = sum(results)
    failed = len(json_files) - successful
    print(f"Processing complete: {successful} successful, {failed} failed")


def process_folder_batch(
    json_folder: str,
    neural_render_folder: str,
    output_folder: str,
    blank_image_path: Optional[str] = None,
    soften_edges: bool = False,
    num_processes: int = None
) -> None:
    """
    Process all JSON files in a folder and composite corresponding neural render images.
    
    Args:
        json_folder: Path to folder containing JSON files
        neural_render_folder: Path to folder containing neural render images
        output_folder: Path to save composited images
        blank_image_path: Optional path to blank image template
    """
    print(f"Processing JSON folder: {json_folder}")
    print(f"Neural render folder: {neural_render_folder}")
    print(f"Output folder: {output_folder}")
    
    # Verify folders exist
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"JSON folder not found: {json_folder}")
    
    if not os.path.exists(neural_render_folder):
        raise FileNotFoundError(f"Neural render folder not found: {neural_render_folder}")
    
    if blank_image_path and not os.path.exists(blank_image_path):
        raise FileNotFoundError(f"Blank image not found: {blank_image_path}")
    
    # Process all files
    composite_neural_render_images(
        json_folder=json_folder,
        neural_render_folder=neural_render_folder,
        output_folder=output_folder,
        blank_image_path=blank_image_path,
        soften_edges=soften_edges,
        num_processes=num_processes
    )
    
    print("Processing complete!")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Composite neural render images onto blank images using JSON placement data")
    parser.add_argument("--json_folder", required=True, help="Path to folder containing JSON files")
    parser.add_argument("--nr", required=True, help="Path to folder containing neural render images")
    parser.add_argument("--output_folder", required=True, help="Path to save composited images")
    parser.add_argument("--blank_image", help="Optional path to blank image template")
    parser.add_argument("--soften_edges", action="store_true", help="Enable edge softening for better compositing")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use for parallel processing (default: all available cores)")
    
    args = parser.parse_args()
    
    process_folder_batch(
        json_folder=args.json_folder,
        neural_render_folder=args.nr,
        output_folder=args.output_folder,
        blank_image_path=args.blank_image,
        soften_edges=args.soften_edges,
        num_processes=args.num_processes
    ) 