
import cv2
import numpy as np
import argparse
import json

def auto_canny_threshold(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper

def load_and_resize(path, resize):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if resize and resize != "none":
        h, w = img.shape[:2]
        if resize.startswith("width="):
            target_w = int(resize.split("=")[1])
            scale = target_w / w
        elif resize.startswith("height="):
            target_h = int(resize.split("=")[1])
            scale = target_h / h
        elif resize.startswith("longest="):
            target_long = int(resize.split("=")[1])
            scale = target_long / max(h, w)
        else:
            raise ValueError(f"Invalid resize option: {resize}")
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def get_edges(img, blur, sigma, low, high, dilate_iter):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur)
    if low is None or high is None:
        low, high = auto_canny_threshold(gray, sigma=sigma)
    edges = cv2.Canny(gray, low, high)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return edges

def overlay_edges(left_edges, right_edges):
    h, w = left_edges.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    both = cv2.bitwise_and(left_edges, right_edges)
    left_only = cv2.bitwise_and(left_edges, cv2.bitwise_not(right_edges))
    right_only = cv2.bitwise_and(right_edges, cv2.bitwise_not(left_edges))

    overlay[left_only > 0] = (0, 0, 255)      # Red for left only
    overlay[right_only > 0] = (255, 255, 0)   # Cyan for right only
    overlay[both > 0] = (255, 255, 255)       # White for both

    return overlay, left_only, right_only, both

def main():
    parser = argparse.ArgumentParser(description="Overlay edges of two images")
    parser.add_argument("--left", required=True, help="Path to left image (or combined image if right is 'none')")
    parser.add_argument("--right", default="none", help="Path to right image (use 'none' to split left image)")
    parser.add_argument("--out", required=True, help="Path to output overlay image")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur sigma before edge detection")
    parser.add_argument("--sigma", type=float, default=0.33, help="Sigma for auto Canny threshold")
    parser.add_argument("--low", type=int, default=None, help="Low threshold for Canny")
    parser.add_argument("--high", type=int, default=None, help="High threshold for Canny")
    parser.add_argument("--dilate", type=int, default=0, help="Number of dilation iterations")
    parser.add_argument("--resize", type=str, default=None, help="Resize images: width=, height=, longest=, or none")
    parser.add_argument("--metrics", type=str, default=None, help="Optional path to save JSON metrics")

    args = parser.parse_args()

    # Handle combined image case
    if args.right == "none":
        combined_img = load_and_resize(args.left, args.resize)
        h, w = combined_img.shape[:2]
        mid = w // 2
        left_img = combined_img[:, :mid]
        right_img = combined_img[:, mid:]
        print(f"Split combined image: left={left_img.shape}, right={right_img.shape}")
    else:
        left_img = load_and_resize(args.left, args.resize)
        right_img = load_and_resize(args.right, args.resize)

    left_edges = get_edges(left_img, args.blur, args.sigma, args.low, args.high, args.dilate)
    right_edges = get_edges(right_img, args.blur, args.sigma, args.low, args.high, args.dilate)

    overlay, left_only, right_only, both = overlay_edges(left_edges, right_edges)

    cv2.imwrite(args.out, overlay)
    print(f"Overlay saved to {args.out}")

    if args.metrics:
        total_edges = np.count_nonzero(left_edges) + np.count_nonzero(right_edges)
        overlap_edges = np.count_nonzero(both)
        metrics = {
            "left_edge_count": int(np.count_nonzero(left_edges)),
            "right_edge_count": int(np.count_nonzero(right_edges)),
            "overlap_edge_count": int(overlap_edges),
            "overlap_ratio": float(overlap_edges) / max(1, total_edges / 2)
        }
        with open(args.metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics}")

if __name__ == "__main__":
    main()
