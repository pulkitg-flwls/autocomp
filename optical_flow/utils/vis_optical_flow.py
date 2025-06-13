import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from metrics import compute_lpips,compute_flip, compute_mse, compute_psnr
from flwls_optical_flow import FlowSystem, FlowSystemConfig
from flwls_optical_flow.temp_common_io import FlowSystemInput, FlowSystemOutput
from skimage.metrics import mean_squared_error
import os 
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_mse_image_and_score(gt_img, warp_img,amp=25):
    
    diff = (gt_img - warp_img) ** 2
    mse_map = np.mean(diff, axis=2)
    mse_map_amp = np.clip(mse_map * amp, 0, 1)
    mse_score = np.mean(mse_map)  # scalar
    return mse_map_amp, mse_score

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0).astype('float32')

def parse_args():
    parser = argparse.ArgumentParser(description="Warp Plate A toward Plate B using optical flow.")
    parser.add_argument("--plate_a", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--plate_b", type=str, help="Path to Plate B image (target)")
    parser.add_argument("--warp_img", type=str, help="")
    parser.add_argument("--gt", type=str, help="")
    parser.add_argument("--flow_viz_dir", type=str, help="")
    parser.add_argument("--title", type=str, help="")
    parser.add_argument("--out_path", type=str, default="warped_output.png", help="Path to save output plot")
    args = parser.parse_args()
    return args    

from skimage.metrics import structural_similarity as ssim

def compute_overlay(gt_img, warp_img, alpha=0.5):
    
    return (alpha * gt_img + (1 - alpha) * warp_img).clip(0, 1)

def compute_flow_magnitude(gt_img, warp_img):
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gt_gray, warp_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return mag, np.mean(mag)

def compute_ssim_error_and_score(gt_img, warp_img):
    # gt = gt_img.astype(np.float32) / 255.0
    # warp = warp_img.astype(np.float32) / 255.0
    h, w = gt_img.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1

    ssim_score, ssim_map = ssim(gt_img, warp_img,
                                full=True,
                                win_size=win_size,
                                channel_axis=-1,data_range=1.0)
    return 1 - ssim_map, ssim_score


def red_green_overlay(gt_img, warp_img):
    return np.stack([
        gt_img[..., 0],         # Red from GT
        warp_img[..., 1],       # Green from warped
        # 0.5 * (gt_img[..., 2] + warp_img[..., 2])  # Blue average
        0*(gt_img[...,2])
    ], axis=-1).clip(0, 1)

def highlight_misalignment_mask(gt_img, warp_img, threshold=0.02, highlight_color=(1.0, 0.0, 0.0)):
    """
    Highlights misaligned pixels with solid red.
    threshold: pixel-wise absolute diff threshold
    highlight_color: RGB tuple in [0,1] for the overlay (default red)
    """
    gt = gt_img.astype(np.float32)
    warp = warp_img.astype(np.float32)

    # Per-pixel mean abs diff
    diff = np.mean(np.abs(gt - warp), axis=2)

    # Binary mask of where misalignment exceeds threshold
    mask = diff > threshold

    # Initialize result as a copy of the GT
    result = gt.copy()

    # Apply red wherever mask is True
    for c in range(3):
        result[..., c][mask] = highlight_color[c]

    return result

def process_single_image(args):
    dns_path,nr_path,warp_path,gt_path,out_path,flow_path,title = args
    img_a = read_image(dns_path)
    img_b = read_image(nr_path)
    warp_img = read_image(warp_path)
    gt_img = read_image(gt_path)
    flow_img = read_image(flow_path)
    error_img,mse = compute_mse_image_and_score(gt_img, warp_img)
    ssim_error_img, ssim_score = compute_ssim_error_and_score(gt_img,warp_img)
    # flow_error_img, flow_error = compute_flow_magnitude(gt_img,warp_img)
    overlay_error_img = highlight_misalignment_mask(gt_img,warp_img)
    og_error_img,og_mse = compute_mse_image_and_score(img_a, img_b)
    # Plot and save results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title}", fontsize=16)
    axes[0,0].imshow(img_a)
    axes[0,0].set_title("OG Img")
    axes[0,1].imshow(img_b)
    axes[0,1].set_title("NR Img")
    axes[0,2].imshow(warp_img)
    axes[0,2].set_title("OG Warp NR")
    axes[0,3].imshow(gt_img)
    axes[0,3].set_title("Nuke GT Img")
    axes[1,0].imshow(og_error_img,cmap='hot')
    axes[1,0].set_title(f"MSE Error(OG,NR) (1e-3):{og_mse*1000:.3f}")
    axes[1,1].imshow(error_img, cmap="hot")
    axes[1,1].set_title(f"MSE Error(OG Warp,Nuke) (1e-3):{mse*1000:.3f}")
    # axes[1,2].imshow(ssim_error_img, cmap="hot")
    # axes[1,2].set_title(f"1-SSIM :{ssim_score:.3f}")
    axes[1,2].imshow(flow_img)
    axes[1,2].set_title(f"Flow (OG,NR)")
    axes[1,3].imshow(overlay_error_img, cmap="hot")
    axes[1,3].set_title("Overlay Img thresh=0.02")
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    flip = compute_flip(warp_img,gt_img)
    lpips = compute_lpips(warp_img,gt_img)
    
    psnr = compute_psnr(warp_img,gt_img)
    # fdbd = compute_forward_backward_consistency(flow_ab,flow_ba)
    return {
            'psnr': psnr,
            'mse': mse,
            'flip': flip,
            'lpips': lpips
        }
    

def image_pairs_generator(args):
    
    for imgs in sorted(os.listdir(args.plate_a)):
        dns_path = os.path.join(args.plate_a,imgs)
        nr_path = os.path.join(args.plate_b,imgs)
        warp_path = os.path.join(args.warp_img,imgs)
        gt_path = os.path.join(args.gt,imgs)
        out_path = os.path.join(args.out_path,imgs)
        flow_path = os.path.join(args.flow_viz_dir,imgs)
        yield(dns_path,nr_path,warp_path,gt_path,out_path,flow_path,args.title)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path,exist_ok=True)
    num_workers = min(cpu_count(), 8)
    print("Num Workers:", num_workers)
    batch_size = 50
    results = []
    with Pool(processes=num_workers) as pool:
        gen = image_pairs_generator(args)
        while True:
            batch = list(next(gen, None) for _ in range(batch_size))
            batch = [b for b in batch if b is not None]  # Remove None values

            if not batch:
                break  # Stop when generator is exhausted

            for res in tqdm(pool.imap_unordered(process_single_image, batch), total=len(batch), desc="Processing batch"):
                if res is not None:
                    results.append(res)  # Collect results
    
    psnr = np.mean([d["psnr"] for d in results])
    lpips = np.mean([d["lpips"] for d in results])
    mse = np.mean([d["mse"] for d in results])
    flip = np.mean([d["flip"] for d in results])
   
    print(f'PSNR:{psnr.mean():.3f}, LPIPS:{lpips.mean():.3f}, MSE:{mse.mean():.5f},FLIP:{flip.mean():.4f}')

    