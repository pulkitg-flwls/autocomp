import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap((img * 255).astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.float32) / 255.0

def morph_images(img_a, img_b, flow_ab, flow_ba, t: float) -> np.ndarray:
    flow_t_a = -(1 - t) * t * flow_ab
    flow_t_b = t * (1 - t) * flow_ba

    warped_a = warp_image(img_a, flow_t_a)
    warped_b = warp_image(img_b, flow_t_b)

    morphed = (1 - t) * warped_a + t * warped_b
    return morphed

def plot_and_save(img_a, img_b, morphed, save_path="morphed_plot.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_a)
    axes[0].set_title("Plate A")
    axes[1].imshow(img_b)
    axes[1].set_title("Plate B")
    axes[2].imshow(morphed)
    axes[2].set_title("Morphed Image")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser(description="Bidirectional warp-based image morphing.")
    parser.add_argument("--plate_a", type=str, help="Path to Plate A (start image)")
    parser.add_argument("--plate_b", type=str, help="Path to Plate B (end image)")
    parser.add_argument("--t", type=float, help="Interpolation value (0 ≤ t ≤ 1)")
    parser.add_argument("--out_path", type=str, default="morphed_plot.png", help="Path to save result plot")
    args = parser.parse_args()

    img_a = read_image(args.plate_a)
    img_b = read_image(args.plate_b)

    flow_ab = compute_optical_flow(img_a, img_b)
    flow_ba = compute_optical_flow(img_b, img_a)

    morphed = morph_images(img_a, img_b, flow_ab, flow_ba, args.t)
    plot_and_save(img_a, img_b, morphed, args.out_path)

if __name__ == "__main__":
    main()