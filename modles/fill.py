import cv2
import os
import matplotlib.pyplot as plt
import numpy as np



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
  if os.path.isfile(dir_path):
    if dir_path.endswith(ext):
      img_list.append(dir_path)
  elif os.path.isdir(dir_path):
    for filename in os.listdir(dir_path):
      full_path = os.path.join(dir_path, filename)
      img_list.extend(get_img_list(full_path, ext))
  return img_list


def read_img(image_dir="./data/"):
  """
  Reads all images with a specific extension from a directory.

  Args:
      image_dir (str, optional): Path to the directory containing images.
          Defaults to "./data/".

  Returns:
      list: A list of loaded images as OpenCV cv2.Mat objects.
  """

  img_list = get_img_list(image_dir)
  loaded_images = []
  for img_path in img_list:
    try:
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      if img is not None:
        imgname = os.path.split(img_path)[1] 
        sift(img, imgname)
        loaded_images.append(img)
      else:
        print(f"Error loading image: {img_path}")
    except cv2.error as e:
      print(f"OpenCV error: {e}")
  return loaded_images

def sift(img,i): 
    top_size,bottom_size,left_size,right_size = (840,840,0,0)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite('./fill/' + (str)(i) , constant)

if __name__ == '__main__':
    images  = read_img()
