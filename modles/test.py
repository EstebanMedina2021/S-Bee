import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt

# Checks if a GPU is available and sets the device for computations (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints/fcn_model.pt', map_location=device) 
model = model.to(device)

# Convert images to tensors and normalize them for the model.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


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


def read_img(image_dir="./fill/"):
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
    if __name__ == '__main__':
        img = cv2.resize(img, (480, 480))

        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        output = model(img)
        output = torch.sigmoid(output)

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)
        ret = np.squeeze(output_np[0, ...])
        plt.imsave('./test/' + (str)(i), ret) # Save image segmentation

if __name__ == '__main__':
    imgall = read_img()
