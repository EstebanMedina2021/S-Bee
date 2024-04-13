import cv2
import os
import numpy as np

def otsu_binarization(image_path, output_dir, filename, kernel_size=3, iterations=1):
  """
  Performs Otsu's thresholding and dilation on an image.

  Args:
      image_path (str): Path to the input image.
      output_dir (str): Path to the output directory for saving the binary image.
      filename (str): Filename of the input image (used for output naming).
      kernel_size (int, optional): Size of the kernel for dilation (defaults to 3).
      iterations (int, optional): Number of iterations for dilation (defaults to 1).

  Returns:
      None
  """

  try:
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      raise FileNotFoundError(f"Image not found: {image_path}")

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)

    # Create output filename with prefix (optional)
    output_filename = f"binary_{filename}"

    # Save the binary image
    cv2.imwrite(os.path.join(output_dir, output_filename), dilated_image)

  except FileNotFoundError as e:
    print(f"Error: {e}")

if __name__ == "__main__":
  input_dir = "./test/"
  output_dir = "./Binarization"

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    otsu_binarization(image_path, output_dir, filename)

