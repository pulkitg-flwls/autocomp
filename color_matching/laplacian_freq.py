import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Add face parsing imports
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
# Import the existing lip mask generator function
from face_parsing.lip_mask_generator import create_smooth_lip_mask

def create_batch_lip_masks(img_a_bgr, img_b_bgr, face_detector, face_parser, 
                          dilation_kernel=15, smooth_kernel=21, smooth_sigma=5.0):
    """
    Create lip masks for two images in batch for better efficiency.
    
    Args:
        img_a_bgr: First image in BGR format
        img_b_bgr: Second image in BGR format
        face_detector: Face detection model
        face_parser: Face parsing model
        dilation_kernel: Size of dilation kernel
        smooth_kernel: Size of Gaussian blur kernel
        smooth_sigma: Standard deviation for Gaussian blur
    
    Returns:
        tuple: (mask_a, mask_b) where each can be None if no face detected
    """
    # Process both images
    mask_a = create_smooth_lip_mask(img_a_bgr, face_detector, face_parser, 
                                   dilation_kernel, smooth_kernel, smooth_sigma)
    mask_b = create_smooth_lip_mask(img_b_bgr, face_detector, face_parser, 
                                   dilation_kernel, smooth_kernel, smooth_sigma)
    
    return mask_a, mask_b


def initialize_face_models(device='cuda:0', threshold=0.8):
    """Initialize face detector and parser models."""
    face_detector = RetinaFacePredictor(
        threshold=threshold, 
        device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25')
    )
    face_parser = RTNetPredictor(
        device=device, 
        ckpt=None, 
        encoder='rtnet50', 
        decoder='fcn', 
        num_classes=11
    )
    return face_detector, face_parser


def nuke_blur(img, radius_x, radius_y, quality=15):
    """
    Gaussian blur that behaves like Nuke:
      • radius = size knob
      • quality = 'don't let the kernel be > quality px' speed trick
    """
    h, w = img.shape[:2]
    max_r = max(radius_x, radius_y)

    # Down-sample if the requested radius is bigger than 'quality'
    if max_r > quality:
        scale = max_r / quality
        small = cv2.resize(img, None, fx=1/scale, fy=1/scale,
                           interpolation=cv2.INTER_LINEAR)
        sig_x = radius_x / (3 * scale)   # radius → σ
        sig_y = radius_y / (3 * scale)
        
        small = cv2.GaussianBlur(small, (0, 0), sig_x, sig_y)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Full-res blur
    sig_x = radius_x / 3.0
    sig_y = radius_y / 3.0
    return cv2.GaussianBlur(img, (0, 0), sig_x, sig_y)


def build_gaussian_pyramid(img, levels=4, blur_radius=401, blur_quality=15):
    """
    Build Gaussian pyramid for an image using nuke_blur.
    
    Args:
        img: Input image (float32, [0,1])
        levels: Number of pyramid levels
        blur_radius: Blur radius for nuke_blur
        blur_quality: Quality parameter for nuke_blur
    
    Returns:
        list: Gaussian pyramid levels
    """
    pyramid = [img.astype(np.float32)]
    
    for i in range(levels - 1):
        # Use nuke_blur for consistent blurring
        blurred = nuke_blur(pyramid[-1], radius_x=blur_radius, radius_y=blur_radius, quality=blur_quality)
        # Downsample by factor of 2
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
    
    return pyramid


def build_laplacian_pyramid(img, levels=4, blur_radius=401, blur_quality=15):
    """
    Build Laplacian pyramid for an image.
    
    Args:
        img: Input image (float32, [0,1])
        levels: Number of pyramid levels
        blur_radius: Blur radius for nuke_blur
        blur_quality: Quality parameter for nuke_blur
    
    Returns:
        list: Laplacian pyramid levels
    """
    gaussian_pyramid = build_gaussian_pyramid(img, levels, blur_radius, blur_quality)
    laplacian_pyramid = []
    
    for i in range(levels - 1):
        # Upsample the next level
        upsampled = cv2.resize(gaussian_pyramid[i + 1], 
                              (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        # Laplacian = current - upsampled
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)
    
    # Add the top level (smallest Gaussian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        laplacian_pyramid: List of Laplacian pyramid levels
    
    Returns:
        np.ndarray: Reconstructed image
    """
    levels = len(laplacian_pyramid)
    reconstructed = laplacian_pyramid[-1].copy()
    
    for i in range(levels - 2, -1, -1):
        # Upsample current reconstruction
        upsampled = cv2.resize(reconstructed, 
                              (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        # Add Laplacian detail
        reconstructed = upsampled + laplacian_pyramid[i]
    
    return reconstructed


def combine_laplacian_pyramids(pyramid_a, pyramid_b, lip_mask=None, levels=4):
    """
    Combine two Laplacian pyramids using OG for low frequencies and NR for high frequencies.
    
    Args:
        pyramid_a: Laplacian pyramid of OG image
        pyramid_b: Laplacian pyramid of NR image
        lip_mask: Optional lip mask for blending
        levels: Number of pyramid levels
    
    Returns:
        np.ndarray: Combined image
    """
    combined_pyramid = []
    
    for i in range(levels):
        if i < levels - 1:
            # High frequency levels: use NR details
            combined_level = pyramid_b[i]
        else:
            # Low frequency level: use OG base
            combined_level = pyramid_a[i]
        
        # Apply lip mask if provided (for high frequency levels)
        if lip_mask is not None and i < levels - 1:
            # Resize mask to match pyramid level
            mask_resized = cv2.resize(lip_mask, 
                                    (combined_level.shape[1], combined_level.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
            mask_3d = mask_resized[..., np.newaxis] if len(combined_level.shape) == 3 else mask_resized
            
            # Blend: mask * NR + (1-mask) * combined
            combined_level = mask_3d * pyramid_b[i] + (1 - mask_3d) * combined_level
        
        combined_pyramid.append(combined_level)
    
    # Reconstruct from combined pyramid
    return reconstruct_from_laplacian_pyramid(combined_pyramid)


def laplacian_high_low_combo(img_a, img_b, lip_mask=None, levels=4, blur_radius=401, blur_quality=15):
    """
    Combine images using Laplacian pyramid decomposition.
    
    Args:
        img_a: OG image (float32, [0,1])
        img_b: NR image (float32, [0,1])
        lip_mask: Optional lip mask for blending
        levels: Number of pyramid levels
        blur_radius: Blur radius for nuke_blur
        blur_quality: Quality parameter for nuke_blur
    
    Returns:
        tuple: (low_a, high_b, combined)
    """
    # Build Laplacian pyramids
    pyramid_a = build_laplacian_pyramid(img_a, levels, blur_radius, blur_quality)
    pyramid_b = build_laplacian_pyramid(img_b, levels, blur_radius, blur_quality)
    
    # Get low frequency component (base of OG pyramid)
    low_a = pyramid_a[-1]
    
    # Get high frequency component (sum of all detail levels from NR)
    high_b = np.zeros_like(img_b)
    for i in range(levels - 1):
        detail = pyramid_b[i]
        # Resize detail to original size
        detail_resized = cv2.resize(detail, (img_b.shape[1], img_b.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        high_b += detail_resized
    
    # Combine using pyramid reconstruction
    combined = combine_laplacian_pyramids(pyramid_a, pyramid_b, lip_mask, levels)
    
    # Ensure values are in valid range
    combined = np.clip(combined, 0.0, 1.0).astype(np.float32)
    
    return low_a, high_b, combined


def read_image(path: str) -> np.ndarray:
    """Load an image and return it as float32 RGB in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Blend images using Laplacian pyramid decomposition."
    )
    parser.add_argument("--og", required=True, help="First image (provides low frequencies)")
    parser.add_argument("--nr", required=True, help="Second image (provides high frequencies)")
    parser.add_argument("--gt", required=True, help="Ground truth image folder")
    parser.add_argument("--output_dir", default="test/combined.png", help="Output image path")
    parser.add_argument("--vis_dir", default="test/plot.png", help="Output image path")
    parser.add_argument("--title", default="Laplacian Pyramid", help="Output image path")
    parser.add_argument("--dilation-kernel", type=int, default=15, help="Dilation kernel size (default=15)")
    parser.add_argument("--smooth-kernel", type=int, default=21, help="Smoothing kernel size (default=21)")
    parser.add_argument("--smooth-sigma", type=float, default=5.0, help="Gaussian blur sigma (default=5.0)")
    parser.add_argument("--levels", type=int, default=4, help="Number of pyramid levels (default=4)")
    parser.add_argument("--blur-radius", type=int, default=401, help="Blur radius for nuke_blur (default=401)")
    parser.add_argument("--blur-quality", type=int, default=15, help="Quality parameter for nuke_blur (default=15)")
    args = parser.parse_args()
    return args


def process_single_img(args):
    
    og_path, nr_path, gt_path, output_path, vis_path, title, dilation_kernel, smooth_kernel, smooth_sigma, levels, blur_radius, blur_quality = args
    
    # Initialize face models for this process
    face_detector, face_parser = initialize_face_models()
    
    # Read both images first
    img_a = read_image(og_path)
    img_b = read_image(nr_path)
    
    # Convert to BGR for batch lip mask generation
    img_a_bgr = (img_a * 255).astype(np.uint8)
    img_a_bgr = cv2.cvtColor(img_a_bgr, cv2.COLOR_RGB2BGR)
    img_b_bgr = (img_b * 255).astype(np.uint8)
    img_b_bgr = cv2.cvtColor(img_b_bgr, cv2.COLOR_RGB2BGR)
    
    # Generate lip masks in batch
    lip_mask_a, lip_mask_b = create_batch_lip_masks(
        img_a_bgr, img_b_bgr, face_detector, face_parser,
        dilation_kernel, smooth_kernel, smooth_sigma
    )
    
    # Convert masks to float32 in [0,1] if they exist
    if lip_mask_a is not None:
        lip_mask_a = lip_mask_a.astype(np.float32) / 255.0
    if lip_mask_b is not None:
        lip_mask_b = lip_mask_b.astype(np.float32) / 255.0
    
    # Create union of lip masks
    if lip_mask_a is not None and lip_mask_b is not None:
        # Union of the two masks
        lip_mask = np.maximum(lip_mask_a, lip_mask_b)
    elif lip_mask_a is not None:
        lip_mask = lip_mask_a
    elif lip_mask_b is not None:
        lip_mask = lip_mask_b
    else:
        lip_mask = None
        print(f"Warning: No face detected in either {og_path} or {nr_path}, using original combination")
    
    # Read GT image
    img_gt = read_image(gt_path)
    
    # Use Laplacian pyramid combination
    low_a, high_b, combined = laplacian_high_low_combo(img_a, img_b, lip_mask=lip_mask, levels=levels, blur_radius=blur_radius, blur_quality=blur_quality)

    # save
    cv2.imwrite(
        output_path,
        cv2.cvtColor((combined * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # quick visual sanity check - 2 rows, 3 columns
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{title}", fontsize=16)
    
    # Turn off all axes
    for row in ax:
        for a in row:
            a.axis("off")
    
    # First row: OG Img, NR Img, GT Img
    ax[0, 0].imshow(img_a)
    ax[0, 0].set_title("OG Img")
    ax[0, 1].imshow(img_b)
    ax[0, 1].set_title("NR Img")
    ax[0, 2].imshow(img_gt)
    ax[0, 2].set_title("GT Img")
    
    # Second row: Low OG, High NR (Masked), Combine
    ax[1, 0].imshow(low_a)
    ax[1, 0].set_title("Low OG")
    
    # Show high_b with masked lips
    if lip_mask is not None:
        # Create masked version: (1-mask)*high_b + mask*nr
        mask_3d = lip_mask[..., np.newaxis]
        high_b_masked = (1 - mask_3d) * np.clip(high_b, 0, 1) + mask_3d * img_b
        ax[1, 1].imshow(np.clip(high_b_masked, 0, 1))
        ax[1, 1].set_title("High NR (Union Mask)")
    else:
        ax[1, 1].imshow(np.clip(high_b, 0, 1))
        ax[1, 1].set_title("High NR")
    
    ax[1, 2].imshow(np.clip(combined, 0, 1))    
    ax[1, 2].set_title("Combine")
    
    plt.tight_layout()
    
    plt.savefig(vis_path)
    plt.close()


if __name__ == "__main__":
    args = arg_parse()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    og_files = sorted(os.listdir(args.og))
    nr_files = sorted(os.listdir(args.nr))

    assert len(og_files) == len(nr_files), "OG and NR folders must contain the same number of files."

    tasks = []
    for file in og_files:
        og_path = os.path.join(args.og, file)
        nr_path = os.path.join(args.nr, file)
        gt_path = os.path.join(args.gt, file)
        output_path = os.path.join(args.output_dir, file)
        vis_path = os.path.join(args.vis_dir, file)
        tasks.append((og_path, nr_path, gt_path, output_path, vis_path, args.title, 
                     args.dilation_kernel, args.smooth_kernel, args.smooth_sigma, args.levels, 
                     args.blur_radius, args.blur_quality))

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_img, tasks), total=len(tasks), desc="Processing images"):
            pass 