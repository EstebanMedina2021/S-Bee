import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def get_img_list(dir, firelist, ext=None):
    newdir = dir
    if os.path.isfile(dir):  
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)

    return firelist

def read_img():
    image_path = './data/'
    imglist = get_img_list(image_path, [], 'jpg')
    imgall = []
    for imgpath in imglist:
        # print(imgpath)
        imgname = os.path.split(imgpath)[1] 
        print(imgname)
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        sift(img, imgname)
        imgall.append(img)
        # cv2.namedWindow(imgname, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow(imgname, img)
        # print(imgname, img.shape)
    # cv2.waitKey(0)

    return imgall

def sift(img,i):
    # img = cv2.imread('./Snapshot020.jpg')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    top_size,bottom_size,left_size,right_size = (840,840,0,0)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
    # constant = cv2.resize(constant,(1920,1920))
    # plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    cv2.imwrite('./fill/' + (str)(i) , constant)
    # plt.imsave('./img/Snapshot001.jpg',constant)
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    imgall = read_img()
