from PIL import Image
import os


def resize_image(image_path, output_format="JPEG", size=(3840, 3840)):
    """Resizes an image in-place and saves it with the same filename and specified format.

    Args:
        image_path (str): Path to the image file.
        output_format (str, optional): Output image format (e.g., JPEG, PNG). Defaults to JPEG.
        size (tuple, optional): Target size for resizing (width, height). Defaults to (3840, 3840).

    Returns:
        None
    """

    try:
        with Image.open(image_path) as image:
            resized_image = image.resize(size)
            resized_image.save(image_path, output_format.upper())  # Save in-place with uppercase format

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing image: {image_path} - {e}")


if __name__ == "__main__":
    input_dir = "./expand/"
    for filename in (f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))):
        # Use generator expression to avoid loading all filenames at once
        image_path = os.path.join(input_dir, filename)
        resize_image(image_path)
