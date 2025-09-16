import cv2
import numpy as np

def evaluate_image(path, blur_thresh_low=30.0, blur_thresh_high=100.0, overexpose_thresh=0.05):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Assessing clarity (Laplace variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_thresh_low:
        sharpness_level = "Blurry"
    elif laplacian_var < blur_thresh_high:
        sharpness_level = "Acceptable"
    else:
        sharpness_level = "Sharp"

    #Assess whether overexposure has occurred
    overexposed_mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
    overexposed_ratio = np.sum(overexposed_mask > 0) / (image.shape[0] * image.shape[1])
    is_overexposed = overexposed_ratio > overexpose_thresh

    #Result output
    print(f"Image: {path}")
    print(f"  Sharpness Value: {laplacian_var:.2f} -> {sharpness_level}")
    print(f"  Overexposed Ratio: {overexposed_ratio:.2%} -> {'Overexposed' if is_overexposed else 'Normal'}")

    return {
        "sharpness": laplacian_var,
        "sharpness_level": sharpness_level,
        "overexposed_ratio": overexposed_ratio,
        "is_overexposed": is_overexposed
    }


if __name__ == "__main__":
    path = "1.png"
    evaluate_image(path)

