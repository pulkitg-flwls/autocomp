import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import json
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def process_single_image(args):
    nuke_path,opencv_path,output_path = args
    
    nuke_img = read_image(nuke_path)
    opencv_img = read_image(opencv_path)
    error_img = np.abs(nuke_img-opencv_img)

    fig, axs = plt.subplots(1, 3, figsize=(6, 3))
    axs[0].imshow(nuke_img)
    axs[0].set_title("Nuke", fontsize=10)
    axs[0].axis('off')

    axs[1].imshow(opencv_img)
    axs[1].set_title("OpenCV OF", fontsize=10)
    axs[1].axis('off')

    axs[2].imshow((error_img*255).astype('uint8'),cmap='hot')
    axs[2].set_title(f"MAE: {error_img.mean():.3f}", fontsize=10)
    axs[2].axis('off')
    
    
    plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.85)
    plt.tight_layout()
    plt.savefig(output_path,bbox_inches='tight', pad_inches=0, dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot triplet images with titles and interpolation values.")
    parser.add_argument('--dir1', required=True, help="Path to original images folder")
    parser.add_argument('--dir2', required=True, help="Path to neural render images folder")
    parser.add_argument('--output_dir', required=True, help="Path to save plotted output images")
    args = parser.parse_args()
    img_pairs = []
    os.makedirs(args.output_dir,exist_ok=True)
    for i,imgs in enumerate(tqdm(os.listdir(args.dir1))):
        nuke_path = os.path.join(args.dir1,imgs)
        opencv_path = os.path.join(args.dir2,imgs)
        output_path = os.path.join(args.output_dir,imgs)
        img_pairs.append((nuke_path,opencv_path,output_path))
        
    num_workers = min(cpu_count(),len(img_pairs))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image,img_pairs)