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
      • quality = 'don’t let the kernel be > quality px' speed trick
    """
    h, w = img.shape[:2]
    max_r = max(radius_x, radius_y)

    # Down-sample if the requested radius is bigger than ‘quality’
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



def read_image(path: str) -> np.ndarray:
    """Load an image and return it as float32 RGB in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def low_high_combo(img_a,
                   img_b,
                   lip_mask=None,
                   blur_radius=401,
                   blur_quality=15,
                    ):
    """
    Returns (low_a, high_b, combined).
    All images assumed float32 RGB in [0,1] and same 1024×1024 size.
    If lip_mask is provided, use feather blending strategy.
    init 51.2
    """
    low_a   = nuke_blur(img_a, radius_x=blur_radius,radius_y=blur_radius, quality=blur_quality)
    low_b = nuke_blur(img_a, radius_x=blur_radius,radius_y=blur_radius, quality=blur_quality)
    high_b  = img_b.astype(np.float32) - low_b.astype(np.float32)
    
    if lip_mask is not None:
        # Use feathered mask for smooth blending between lip and non-lip regions
        mask_3d = lip_mask[..., np.newaxis]
        # Blend: mask * NR + (1-mask) * (low_a + high_b)
        combo = mask_3d * img_b + (1 - mask_3d) * np.clip(low_a + high_b, 0.0, 1.0)
        combo = np.clip(combo, 0.0, 1.0).astype(np.float32)
    else:
        # Original combination
        combo = np.clip(low_a + high_b, 0.0, 1.0).astype(np.float32)
    
    return low_a, high_b, combo


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Blend blurred Image-A with high-freq details of Image-B."
    )
    parser.add_argument("--og", required=True, help="First image (provides blur)")
    parser.add_argument("--nr", required=True, help="Second image (provides details)")
    parser.add_argument("--gt", required=True, help="Ground truth image folder")
    parser.add_argument("--output_dir", default="test/combined.png", help="Output image path")
    parser.add_argument("--vis_dir", default="test/plot.png", help="Output image path")
    parser.add_argument("--title", default="dpflow", help="Output image path")
    parser.add_argument("--dilation-kernel", type=int, default=15, help="Dilation kernel size (default=15)")
    parser.add_argument("--smooth-kernel", type=int, default=21, help="Smoothing kernel size (default=21)")
    parser.add_argument("--smooth-sigma", type=float, default=5.0, help="Gaussian blur sigma (default=5.0)")
    args = parser.parse_args()
    return args

def process_single_img(args):
    
    og_path, nr_path, gt_path, output_path, vis_path, title, dilation_kernel, smooth_kernel, smooth_sigma = args
    
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
    
    # Use lip mask in combination
    blur_a, high_b, combined = low_high_combo(img_a, img_b, lip_mask=lip_mask)

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
    
    # Second row: Blur OG, High NR (Masked), Combine
    ax[1, 0].imshow(blur_a)
    ax[1, 0].set_title("Blur OG")
    
    # Show high_b with masked lips: (1-mask)*high_b + mask*nr
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
                     args.dilation_kernel, args.smooth_kernel, args.smooth_sigma))

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_img, tasks), total=len(tasks), desc="Processing images"):
            pass