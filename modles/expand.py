import cv2
import os
import numpy as np

def apply_dilation(image_path, output_dir, filename, kernel_size=40, iterations=1):
    """Applies dilation to an image with the given parameters.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Path to the output directory.
        filename (str): Filename of the input image.
        kernel_size (int, optional): Size of the kernel for dilation. Defaults to 40.
        iterations (int, optional): Number of iterations for dilation. Defaults to 1.
    """

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)

        output_filename = f"dilated_{filename}"
        cv2.imwrite(os.path.join(output_dir, output_filename), dilated_image)

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_dir = "./Binarization/"
    output_dir = "./expand"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        apply_dilation(image_path, output_dir, filename)
