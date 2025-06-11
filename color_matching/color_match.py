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
def laplacian_per_channel(img,scale=1.0):
    channels = cv2.split(img)
    laplacians = [cv2.Laplacian(c, cv2.CV_32F, ksize=3) * scale for c in channels]
    return cv2.merge(laplacians)

def color_match(img_a,img_b,blur_ksize=201):
    # blur_a = cv2.GaussianBlur(img_a, (blur_ksize, blur_ksize), 0)
    blur_a = cv2.bilateralFilter(img_a.astype('float32'), d=0, sigmaColor=150, sigmaSpace=150)
    # blur_a = np.ones_like(img_a) * img_a.mean(axis=(0, 1), keepdims=True)
    # High frequency from B via Laplacian
    laplacian_b = laplacian_per_channel(img_b.astype('float32'))
    # Combine
    combined = np.clip(blur_a + laplacian_b, 0, 1)
    return blur_a,combined


def main():
    parser = argparse.ArgumentParser(description="Warp Plate A toward Plate B using optical flow.")
    parser.add_argument("--plate_a", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--plate_b", type=str, help="Path to Plate B image (target)")
    parser.add_argument("--out_path", type=str, default="warped_output.png", help="Path to save output plot")
    args = parser.parse_args()

    img_a = read_image(args.plate_a)
    img_b = read_image(args.plate_b)

    blur_a,combined = color_match(img_a,img_b)
    
    # Plot and save results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_a)
    axes[0].set_title("Plate A")
    axes[1].imshow(img_b)
    axes[1].set_title("Plate B")
    axes[2].imshow(blur_a)
    axes[2].set_title("Blur img")
    axes[3].imshow(combined)
    axes[3].set_title("Combined")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out_path)
    # flip = compute_flip(img_a,img_a_to_b)
    # lpips = compute_lpips(img_a,img_a_to_b)
    # flip_f = compute_flip(img_a,img_a_to_b_f)
    # lpips_f = compute_lpips(img_a,img_a_to_b_f)
    # # fdbd = compute_forward_backward_consistency(flow_ab,flow_ba)
    # print(f'FLIP OpenCV {flip:.5f}')
    # print(f'LPIPS OpenCV {lpips:.5f}')
    # print(f'FLIP FlowFormer {flip_f:.5f}')
    # print(f'LPIPS FlowFormer {lpips_f:.5f}')
    # # print(f'{fdbd:3f}')

if __name__ == "__main__":
    main()