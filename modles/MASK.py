import cv2
import os
from tqdm import tqdm

def get_img_list(dir_path, ext=".jpg"):
    """
    Gets a list of image paths with a specified extension within a directory.

    Args:
        dir_path (str): Path to the directory containing images.
        ext (str, optional): The file extension of the images to search for.
            Defaults to ".jpg".

    Returns:
        list: A list of image paths matching the criteria.
    """
    img_list = []
    if os.path.isfile(dir_path) and dir_path.endswith(ext):
        img_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(ext):
                    img_list.append(os.path.join(root, file))
    return img_list

def read_img(image_path1='./cut/', image_path2='./data/'):
    imglist1 = get_img_list(image_path1)
    imglist2 = get_img_list(image_path2)
    for imgpath1, imgpath2 in tqdm(zip(imglist1, imglist2), total=min(len(imglist1), len(imglist2))):
        imgname1 = os.path.split(imgpath1)[1]
        img1 = cv2.imread(imgpath1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(imgpath2, cv2.IMREAD_COLOR)
        if img1 is not None and img2 is not None:
            sift(img1, img2, imgname1)

def sift(img1, img2, imgname1):
  # Check if images have the same shape
  if img1.shape != img2.shape:
    # Resize img1 to match img2 size
    img1_resized = cv2.resize(img1, dsize=(img2.shape[1], img2.shape[0]))
    image = cv2.bitwise_and(img1_resized, img2)
  else:
    # Images have the same shape, perform bitwise AND directly
    image = cv2.bitwise_and(img1, img2)

  # Save the resulting image
  cv2.imwrite(f'./MASK/{imgname1}', image)

if __name__ == '__main__':
    read_img()
