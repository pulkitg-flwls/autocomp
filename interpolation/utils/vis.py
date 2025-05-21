import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import json

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Get sorted lists
    files_og = sorted(glob(os.path.join(args.og, "*.png")))
    files_nr = sorted(glob(os.path.join(args.nr, "*.png")))
    files_res = sorted(glob(os.path.join(args.result, "*.png")))
    files_json = sorted(glob(os.path.join(args.json_dir, "*.json")))

    assert len(files_og) == len(files_nr) == len(files_res) == len(files_json), "Mismatch in number of files."

    for idx, (f1, f2, f3, fjson) in enumerate(zip(files_og, files_nr, files_res, files_json), start=1):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        titles = ["Original", "Neural Render", "Result"]
        for ax, file, title in zip(axs, [f1, f2, f3], titles):
            img = mpimg.imread(file)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # Read interpolation value
        with open(fjson, 'r') as jf:
            data = json.load(jf)
        interp_val = data.get("interpolation_value", "N/A")

        fig.suptitle(f"Interpol Val = {interp_val:.2f}", fontsize=12)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.85)

        out_path = os.path.join(args.output_dir, f"{idx:06d}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot triplet images with titles and interpolation values.")
    parser.add_argument('--og', required=True, help="Path to original images folder")
    parser.add_argument('--nr', required=True, help="Path to neural render images folder")
    parser.add_argument('--result', required=True, help="Path to result images folder")
    parser.add_argument('--json_dir', required=True, help="Path to JSON files with interpolation values")
    parser.add_argument('--output_dir', required=True, help="Path to save plotted output images")
    args = parser.parse_args()
    main(args)