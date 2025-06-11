import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from metrics import compute_lpips,compute_flip
from flwls_optical_flow import FlowSystem, FlowSystemConfig
from flwls_optical_flow.temp_common_io import FlowSystemInput
import mediapipe as mp
from scipy.spatial import Delaunay

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def compute_optical_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    forward = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    backward = cv2.calcOpticalFlowFarneback(
        gray2, gray1, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return forward,backward

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

def get_face_keypoints(img: np.ndarray) -> np.ndarray:
    """Extracts facial keypoints using MediaPipe. Returns (N, 2) array of (x, y)."""
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process((img * 255).astype(np.uint8))
        if not results.multi_face_landmarks:
            raise ValueError("No face detected.")
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        coords = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        return coords
def inpaint_flow(flow: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaints each channel of the flow using the provided mask."""
    inpainted = np.zeros_like(flow)
    for i in range(2):  # For dx and dy channels
        # OpenCV inpaint requires 8-bit or 32-bit 1-channel image
        channel = flow[..., i]
        inpainted[..., i] = cv2.inpaint(channel.astype(np.float32), mask.astype(np.uint8), 3, cv2.INPAINT_NS)
    return inpainted

def dense_keypoint_flow(img1: np.ndarray, img2: np.ndarray, selected_indices=None):
    """Creates dense flow based on facial keypoints between img1 and img2."""
    kp1 = get_face_keypoints(img1)
    kp2 = get_face_keypoints(img2)
    
    # Focus on lower face (jawline) - landmarks 0–16 in MediaPipe
    lower_face_indices = (
    list(range(0, 17)) +         # Jawline
    list(range(61, 89)) +        # Outer lips
    list(range(95, 107)) +       # Inner lips
    [97, 98, 99, 100, 164, 393, 152, 148, 172]  # Lower nose + cheek base + chin center
    )  # MediaPipe jaw
    src_jaw = kp1[lower_face_indices]
    dst_jaw = kp2[lower_face_indices]

    # --- FIXED REGION: Anchor points (same in both images) ---
    all_indices = set(range(468))  # MediaPipe total
    anchor_indices = sorted(list(all_indices - set(lower_face_indices)))
    anchor_src = kp1[anchor_indices]
    anchor_dst = kp1[anchor_indices]  # fixed in both source and target

    # --- Combine for triangulation ---
    src_pts = np.vstack([src_jaw, anchor_src])
    dst_pts = np.vstack([dst_jaw, anchor_dst])

    h, w = img1.shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float32)

    tri = Delaunay(src_pts)

    for simplex in tri.simplices:
        src_triangle = np.float32(src_pts[simplex])
        dst_triangle = np.float32(dst_pts[simplex])

        # Create affine transform per triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, src_triangle.astype(np.int32), 1)

        coords = np.column_stack(np.nonzero(mask))
        coords = coords[:, ::-1]  # (y, x) -> (x, y)

        if len(coords) == 0:
            continue

        # Calculate affine transform
        M = cv2.getAffineTransform(src_triangle, dst_triangle)
        dst_coords = cv2.transform(np.array([coords], dtype=np.float32), M)[0]

        displacements = dst_coords - coords
        for (x, y), dxy in zip(coords, displacements):
            if 0 <= y < h and 0 <= x < w:
                flow[y, x] = dxy

    # Fill empty flow regions using interpolation or smooth filtering
    invalid_mask = (np.linalg.norm(flow, axis=2) == 0).astype(np.uint8)
    flow_filled = inpaint_flow(flow, invalid_mask)
    # Get backward flow as inverse displacement at valid pixels
    backward_flow = -flow_filled  # Approximation

    return flow_filled, backward_flow


def main():
    parser = argparse.ArgumentParser(description="Warp Plate A toward Plate B using optical flow.")
    parser.add_argument("--plate_a", type=str, help="Path to Plate A image (source)")
    parser.add_argument("--plate_b", type=str, help="Path to Plate B image (target)")
    parser.add_argument("--out_path", type=str, default="warped_output.png", help="Path to save output plot")
    args = parser.parse_args()

    img_a = read_image(args.plate_a)
    img_b = read_image(args.plate_b)

    flow_ab, flow_ba = compute_optical_flow(img_a, img_b)
    
    flow_ab_f,flow_ba_f = compute_flwls_optical_flow(img_a,img_b,'Raft')
    flow_kab,flow_kba = dense_keypoint_flow(img_a,img_b)
    img_a_to_b = warp_image(img_a, flow_ab)
    img_a_to_b_f = warp_image(img_a, flow_ab_f)

    flow_rgb = flow_to_rgb(flow_ab)
    flow_rgb_f = flow_to_rgb(flow_ab_f)
    flow_rgb_k = flow_to_rgb(flow_kab)
    
    
    img_a_to_b_k = warp_image(img_a,flow_kab)
    # Plot and save results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_a)
    axes[0].set_title("Plate A")
    axes[1].imshow(img_b)
    axes[1].set_title("Plate B")
    axes[2].imshow(img_a_to_b_k)
    axes[2].set_title("A Warped to B")
    axes[3].imshow(flow_rgb_k)
    axes[3].set_title("Optical Flow A→B")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out_path)
    # flip = compute_flip(img_a,img_a_to_b)
    # lpips = compute_lpips(img_a,img_a_to_b)
    # flip_f = compute_flip(img_a,img_a_to_b_f)
    # lpips_f = compute_lpips(img_a,img_a_to_b_f)
    # fdbd = compute_forward_backward_consistency(flow_ab,flow_ba)
    # print(f'FLIP OpenCV {flip:.5f}')
    # print(f'LPIPS OpenCV {lpips:.5f}')
    # print(f'FLIP FlowFormer {flip_f:.5f}')
    # print(f'LPIPS FlowFormer {lpips_f:.5f}')
    # print(f'{fdbd:3f}')

if __name__ == "__main__":
    main()