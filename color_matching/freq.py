import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
                   blur_radius=401,
                   blur_quality=15,
                    ):
    """
    Returns (low_a, high_b, combined).
    All images assumed float32 RGB in [0,1] and same 1024×1024 size.
    init 51.2
    """
    low_a   = nuke_blur(img_a, radius_x=blur_radius,radius_y=blur_radius, quality=blur_quality)
    low_b = nuke_blur(img_a, radius_x=blur_radius,radius_y=blur_radius, quality=blur_quality)
    high_b  = img_b.astype(np.float32) - low_b.astype(np.float32)
    combo   = np.clip(low_a + high_b, 0.0, 1.0).astype(np.float32)
    return low_a, high_b, combo


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Blend blurred Image-A with high-freq details of Image-B."
    )
    parser.add_argument("--og", required=True, help="First image (provides blur)")
    parser.add_argument("--nr", required=True, help="Second image (provides details)")
    parser.add_argument("--output_dir", default="test/combined.png", help="Output image path")
    parser.add_argument("--vis_dir", default="test/plot.png", help="Output image path")
    parser.add_argument("--title", default="dpflow", help="Output image path")
    args = parser.parse_args()
    return args

def process_single_img(args):
    
    og_path,nr_path,output_path,vis_path,title = args
    img_a = read_image(og_path)
    img_b = read_image(nr_path)
    # blur_a, high_b,combined = low_high_combo(img_a, img_b, args.ksize)
    blur_a, high_b,combined = low_high_combo(img_a, img_b)

    # save
    cv2.imwrite(
        output_path,
        cv2.cvtColor((combined * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # quick visual sanity check
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"{title}", fontsize=16)
    for a in ax:
        a.axis("off")
    ax[0].imshow(img_a)
    ax[0].set_title("OG Img")
    ax[1].imshow(img_b)
    ax[1].set_title("NR Img")
    ax[2].imshow(blur_a)
    ax[2].set_title("Blur OG")
    ax[3].imshow(np.clip(high_b,0,1))
    ax[3].set_title("High NR")
    ax[4].imshow(np.clip(combined,0,1))    
    ax[4].set_title("Combine")
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
        output_path = os.path.join(args.output_dir, file)
        vis_path = os.path.join(args.vis_dir, file)
        tasks.append((og_path, nr_path, output_path, vis_path, args.title))

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_img, tasks), total=len(tasks), desc="Processing images"):
            pass