import cv2
import numpy as np
from ptlflow.utils import flow_utils  # ships with PTLFlow
import matplotlib.pyplot as plt
import argparse

def warp_image(img_path: str,
               flow_path: str,
               *,
               border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """
    Args
    ----
    img_path  : path to RGB frame that you want to re-sample
    flow_path : path to flow.png produced by PTLFlow
                (forward flow: frame_t  →  frame_{t+1})
    border    : remap padding mode (cv2 constant)

    Returns
    -------
    warped_img: the image backward-warped to align with the *next* frame
    """
    # 1. read data ----------------------------------------------------------------
    img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # flow_read → float32 (H,W,2) in pixel units
    flow = flow_utils.flow_read(flow_path)      # x in [:, :, 0], y in [:, :, 1]

    assert flow.shape[:2] == (h, w), "flow/image size mismatch"

    # 2. build absolute sampling grid --------------------------------------------
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x  = (xs + flow[..., 0]).astype(np.float32)
    map_y  = (ys + flow[..., 1]).astype(np.float32)

    # 3. sample & return ----------------------------------------------------------
    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=border)
    return warped

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def compute_mse_image_and_score(gt_img, warp_img,amp=1000):
    
    diff = (gt_img - warp_img) ** 2
    mse_map = np.mean(diff, axis=2)
    mse_map_amp = np.clip(mse_map * amp, 0, 1)
    mse_score = np.mean(mse_map)  # scalar
    return mse_map_amp, mse_score

def main():
    parser = argparse.ArgumentParser(description="Warp Plate A toward Plate B using optical flow.")
    parser.add_argument("--img_a", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--img_b", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--flow", type=str, help="Path to Plate B image (target)")
    parser.add_argument("--out_path", type=str, default="warped_output.png", help="Path to save output plot")
    args = parser.parse_args()

    img_a_to_b = warp_image(args.img_a,args.flow)
    flow_img = read_image(args.flow)

    img_a = read_image(args.img_a)
    img_b = read_image(args.img_b)
    error_img, error_score = compute_mse_image_and_score(img_a,img_a_to_b)
    # Plot and save results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_a)
    axes[0].set_title("Plate A")
    axes[1].imshow(img_b)
    axes[1].set_title("Plate B")
    axes[2].imshow(img_a_to_b)
    axes[2].set_title("A Warped to B")
    axes[3].imshow((error_img*255).astype('uint8'),cmap='hot')
    axes[3].set_title("Optical Flow A→B")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out_path)
    cv2.imwrite('warpy.png',cv2.cvtColor((img_a_to_b),cv2.COLOR_BGR2RGB).astype('uint8'))
    

if __name__ == "__main__":
    main()