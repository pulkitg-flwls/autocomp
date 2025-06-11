import os
import json
import argparse
import cv2
from json_update import fix_ramps_np, visualize_ramp_fix
from morph import compute_optical_flow, morph_images
from tqdm import tqdm
from flwls_optical_flow import FlowSystem, FlowSystemConfig
from flwls_optical_flow.temp_common_io import FlowSystemInput, FlowSystemOutput

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

def read_interpolation_values(json_dir):
    json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    values = []

    for filename in json_files:
        path = os.path.join(json_dir, filename)
        with open(path, 'r') as f:
            data = json.load(f)
            val = data.get("interpolation_value", None)
            if val is not None:
                values.append(val)
            else:
                print(f"Warning: No 'interpolation_value' in {filename}")
    return values

def compute_flwls_optical_flow(img1,img2,algo='Raft'):
    optical_flow = FlowSystem(FlowSystemConfig(
        batch_size = 1,
        model_name =algo,
        flow_direction = "both",
        device = "cuda",
    ))
    img1 = (img1*255).astype('uint8')
    img2 = (img2*255).astype('uint8')
    value = [FlowSystemInput(image=img1), FlowSystemInput(image=img2)]
    output = list(optical_flow(value))[0]
    output = output.to_dict()
    forward = output['forward_flow'].transpose(2,1,0)
    backward = output['backward_flow'].transpose(2,1,0)
    return forward,backward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--og", required=True, help="Path to original images folder")
    parser.add_argument("--nr", required=True, help="Path to neural rendered images folder")
    parser.add_argument("--output_dir", required=True, help="Path to output dir")
    parser.add_argument("--interpolation", required=True, help="Path to interpolation JSON folder")
    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    if not all(map(os.path.exists, [args.og, args.nr, args.interpolation])):
        raise FileNotFoundError("One or more input directories do not exist.")

    interp_vals = read_interpolation_values(args.interpolation)
    update_interp_vals = fix_ramps_np(interp_vals)
    visualize_ramp_fix(interp_vals,update_interp_vals)

    for idx,file in enumerate(tqdm(sorted(os.listdir(args.og)))):
        og_img = read_image(os.path.join(args.og,file))
        nr_img = read_image(os.path.join(args.nr,file))

        if abs(update_interp_vals[idx]-0.0) < 1e-4:
            result = og_img
        elif abs(update_interp_vals[idx]-1.0) < 1e-4:
            result = nr_img
        else:
            # print(idx)
            # flow_ab = compute_optical_flow(og_img, nr_img)
            # flow_ba = compute_optical_flow(nr_img, og_img)
            flow_ab,flow_ba = compute_flwls_optical_flow(og_img,nr_img,algo='FlowFormer++')

            result = morph_images(og_img, nr_img, flow_ab, flow_ba, update_interp_vals[idx])
        output_path = os.path.join(args.output_dir,file)
        save_image(output_path,result)
        del og_img, nr_img, result

