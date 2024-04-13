import cv2
import os


def crop_image(input_img_path, output_img_path, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0):
  """Crops an image based on specified coordinates and saves it.

  Args:
      input_img_path (str): Path to the input image.
      output_img_path (str): Path to save the cropped image.
      crop_top (int, optional): Number of pixels to crop from the top. Defaults to 0.
      crop_bottom (int, optional): Number of pixels to crop from the bottom. Defaults to 0.
      crop_left (int, optional): Number of pixels to crop from the left. Defaults to 0.
      crop_right (int, optional): Number of pixels to crop from the right. Defaults to 0.
  """

  try:
    image = cv2.imread(input_img_path)
    if image is None:
      raise FileNotFoundError(f"Image not found: {input_img_path}")

    height, width, _ = image.shape  # Get image dimensions

    # Adjust cropping coordinates based on image dimensions (optional)
    top = max(0, crop_top)  # Ensure top doesn't go beyond image height
    bottom = min(height, crop_bottom)  # Ensure bottom doesn't exceed image height
    left = max(0, crop_left)  # Ensure left doesn't go beyond image width
    right = min(width, crop_right)  # Ensure right doesn't exceed image width

    cropped_image = image[top:bottom, left:right]
    cv2.imwrite(output_img_path, cropped_image)

  except FileNotFoundError as e:
    print(f"Error: {e}")
  except Exception as e:
    print(f"Error processing image: {input_img_path} - {e}")


if __name__ == "__main__":
  dataset_dir = 'expand'
  output_dir = 'cut'

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  for filename in os.listdir(dataset_dir):
    input_img_path = os.path.join(dataset_dir, filename)
    output_img_path = os.path.join(output_dir, filename)
    crop_image(input_img_path, output_img_path, crop_top=840, crop_bottom=3000, crop_left=0, crop_right=3840)
