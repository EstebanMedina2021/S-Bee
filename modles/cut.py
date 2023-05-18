import numpy as np
import cv2
import os


def update(input_img_path, output_img_path):
    image = cv2.imread(input_img_path)
    print(image.shape)
    cropped = image[840:3000, 0:3840]  
    cv2.imwrite(output_img_path, cropped)


dataset_dir = 'expand'
output_dir = 'cut'


image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
                   for x in os.listdir(dataset_dir)]

for path in image_filenames:
    update(path[0], path[1])
