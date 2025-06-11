import os
import json
import argparse
import cv2
from tqdm import tqdm
from warp import compute_optical_flow,compute_flwls_optical_flow,warp_image, dense_keypoint_flow
from multiprocessing import Pool, cpu_count
from functools import partial

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

# Save an RGB float32 image (0–1) to disk as 8-bit PNG.
def save_image(path, img):
    """Save an RGB float32 image (0–1) to disk as 8-bit PNG."""
    img_uint8 = (img * 255).clip(0, 255).astype('uint8')
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def process_image(file, args):
    og_path = os.path.join(args.og, file)
    nr_path = os.path.join(args.nr, file)
    output_path = os.path.join(args.output_dir, file)

    og_img = read_image(og_path)
    nr_img = read_image(nr_path)

    if args.algo == 'opencv':
        flow_ab, _ = compute_optical_flow(og_img, nr_img)
    elif args.algo == 'dense':
        flow_ab, _ = dense_keypoint_flow(og_img, nr_img)
    else:
        flow_ab, _ = compute_flwls_optical_flow(og_img, nr_img, args.algo)

    result = warp_image(og_img, flow_ab)
    save_image(output_path, result)
    return file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--og", required=True, help="Path to original images folder")
    parser.add_argument("--nr", required=True, help="Path to neural rendered images folder")
    parser.add_argument("--output_dir", required=True, help="Path to output dir")
    parser.add_argument("--algo",type=str, required=True,default='opencv',help="Path to interpolation JSON folder")
    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    if not all(map(os.path.exists, [args.og, args.nr])):
        raise FileNotFoundError("One or more input directories do not exist.")

    # interp_vals = read_interpolation_values(args.interpolation)

    for idx,file in enumerate(tqdm(sorted(os.listdir(args.og)))):
        og_img = read_image(os.path.join(args.og,file))
        nr_img = read_image(os.path.join(args.nr,file))

        # if abs(interp_vals[idx]-0.0) < 1e-4:
        if args.algo == 'opencv':
            flow_ab,flow_ba = compute_optical_flow(og_img,nr_img)
        elif args.algo == 'dense':
            flow_ab,flow_ba = dense_keypoint_flow(og_img,nr_img)
        else:
            flow_ab,flow_ba = compute_flwls_optical_flow(og_img,nr_img,args.algo)
        result = warp_image(og_img, flow_ab)
        output_path = os.path.join(args.output_dir,file)
        save_image(output_path,result)
        del og_img, nr_img, result
    # files = sorted(os.listdir(args.og))
    # num_workers = min(cpu_count(), 1)  # or just cpu_count()

    # with Pool(processes=num_workers) as pool:
        # list(tqdm(pool.imap_unordered(partial(process_image, args=args), files), total=len(files)))

