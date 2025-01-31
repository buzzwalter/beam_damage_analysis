import cv2
import numpy as np
import os 

def threshold_differencing(image1, image2, max_difference, max_area, low_intensity):
    """
    Compute threshold differencing between two images.

    Parameters:
        image1: np.array - First image.
        image2: np.array - Second image.
        max_difference: float - Max percent difference for pixel change threshold.
        max_area: float - Max percent of changed pixels to trigger damage detection.
        low_intensity: int - Threshold for low pixel values.

    Returns:
        bool: True if image is damaged based on thresholds, False otherwise.
        int: Count of changed pixels.
    """
    # Convert images to float
    img1_float = image1.astype(np.float32)
    img2_float = image2.astype(np.float32)

    # Set low-intensity pixels to the threshold value -- gets rid of superfluous non-damage related pixel flagging from noise
    img1_float[img1_float < low_intensity] = low_intensity
    img2_float[img2_float < low_intensity] = low_intensity

    # Compute absolute difference and relative difference
    diff = np.abs(img1_float - img2_float)
    relative_diff = diff / img1_float

    # Threshold based on max_difference
    thresholded_diff = (relative_diff > (max_difference / 100.0)).astype(np.uint8) * 255
    
    # Ensure the image is single-channel
    if len(thresholded_diff.shape) > 2:
        thresholded_diff = cv2.cvtColor(thresholded_diff, cv2.COLOR_BGR2GRAY)

    
    # Count non-zero pixels
    changed_pixels = cv2.countNonZero(thresholded_diff)
    total_pixels = image1.size

    # Check if the area of changed pixels exceeds max_area
    if changed_pixels / total_pixels > (max_area / 100.0):
        return True, changed_pixels

    return False, changed_pixels

def process_directory(directory, max_difference, max_area, low_intensity):
    """
    Process images in a directory to analyze differences between consecutive images.

    Parameters:
        directory: str - Path to the directory containing images.
        max_difference: float - Max percent difference for pixel change threshold.
        max_area: float - Max percent of changed pixels to trigger damage detection.
        low_intensity: int - Threshold for low pixel values.

    Returns:
        None
    """
    # Get sorted list of image paths
    image_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )

    if len(image_files) < 2:
        print("Not enough images to process.")
        return

    for i in range(len(image_files) - 1):
        # Load consecutive images
        img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_files[i + 1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Failed to load images: {image_files[i]} or {image_files[i + 1]}")
            continue

        # Perform threshold differencing
        damaged, count = threshold_differencing(img1, img2, max_difference, max_area, low_intensity)

        # Print results
        print(f"Comparing {image_files[i]} and {image_files[i + 1]}:")
        print(f"  Damaged: {damaged}, Changed Pixels: {count}")

if __name__ == "__main__":
    # Parameters
    input_directory = "path/to/image/directory"  # Change this to your directory path
    max_difference = 20.0  # Maximum percent difference threshold
    max_area = 10.0        # Maximum percent area threshold
    low_intensity = 30     # Minimum intensity threshold

    # Process the directory
    process_directory(input_directory, max_difference, max_area, low_intensity)

