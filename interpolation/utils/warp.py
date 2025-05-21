import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def compute_optical_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow
def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB for visualization.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle / 2              # Hue maps direction
    hsv[..., 1] = 255                    # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Brightness = magnitude

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap((img * 255).astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description="Warp Plate A toward Plate B using optical flow.")
    parser.add_argument("--plate_a", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--plate_b", type=str, help="Path to Plate B image (target)")
    parser.add_argument("--out_path", type=str, default="warped_output.png", help="Path to save output plot")
    args = parser.parse_args()

    img_a = read_image(args.plate_a)
    img_b = read_image(args.plate_b)

    flow_ab = compute_optical_flow(img_a, img_b)
    flow_ba = compute_optical_flow(img_b, img_a)

    img_a_to_b = warp_image(img_a, flow_ab)

    flow_rgb = flow_to_rgb(flow_ab)

    # Plot and save results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_a)
    axes[0].set_title("Plate A")
    axes[1].imshow(img_b)
    axes[1].set_title("Plate B")
    axes[2].imshow(img_a_to_b)
    axes[2].set_title("A Warped to B")
    axes[3].imshow(flow_rgb)
    axes[3].set_title("Optical Flow A→B")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"[✓] Saved plot to: {args.out_path}")

if __name__ == "__main__":
    main()