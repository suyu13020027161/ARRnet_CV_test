import cv2
import numpy as np
import os

#Change it to your own image folder path!!!
path='/home/ysu/Downloads/CV_test_dataset_50'

def evaluate_image(path, blur_thresh_low=30.0, blur_thresh_high=100.0, overexpose_thresh=0.05):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Assessing clarity
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_thresh_low:
        sharpness_level = "Blurry"
    elif laplacian_var < blur_thresh_high:
        sharpness_level = "Acceptable"
    else:
        sharpness_level = "Sharp"

    # Assess overexposure
    overexposed_mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
    overexposed_ratio = np.sum(overexposed_mask > 0) / (image.shape[0] * image.shape[1])
    is_overexposed = overexposed_ratio > overexpose_thresh

    print(f"Image: {path}")
    print(f"  Sharpness Value: {laplacian_var:.2f} -> {sharpness_level}")
    print(f"  Overexposed Ratio: {overexposed_ratio:.2%} -> {'Overexposed' if is_overexposed else 'Normal'}")

    return {
        "sharpness_level": sharpness_level,
        "is_overexposed": is_overexposed
    }

if __name__ == "__main__":
    folder = path
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    total = 0
    qualified = 0

    for img_path in all_files:
        result = evaluate_image(img_path)
        if result is not None:
            total += 1
            if result["sharpness_level"] in ("Acceptable", "Sharp") and not result["is_overexposed"]:
                qualified += 1

    print("\n=== Summary ===")
    print(f"Total images evaluated: {total}")
    if total > 0:
        percent = qualified / total * 100
        print(f"Qualified images (Acceptable or Sharp and Normal exposure): {qualified} ({percent:.2f}%)")
    else:
        print("No images found.")

