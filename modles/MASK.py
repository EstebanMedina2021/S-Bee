import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


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
    image_path1 = './cut/'
    image_path2 = './data/'
    imglist1 = get_img_list(image_path1, [], 'jpg')
    imglist2 = get_img_list(image_path2, [], 'jpg')
    imgall1 = []
    imgall2 = []
    # print(imglist1)
    for i in range(412):
        # print(imgpath)
        imgpath1 = imglist1[i]
        imgpath2 = imglist2[i]
        imgname1 = os.path.split(imgpath1)[1]  
        imgname2 = os.path.split(imgpath2)[1]
        print(imgname1)
        print(imgname2)
        img1 = cv2.imread(imgpath1, cv2.IMREAD_COLOR)
        # img1 = cv2.resize(img1,(480,480))
        img2 = cv2.imread(imgpath2, cv2.IMREAD_COLOR)
        sift(img1, img2,imgname1)
        # imgall.append(img)




def sift(img1,img2,imgname1):
   
    # img1 = cv2.imread(yuantu)
    # img2 = cv2.imread(masktu, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite(imgname1,img2)
    # img2= cv2.imread(imgname1)
    alpha = 0.5
    meta = 1 - alpha
    gamma = 0
    #cv2.imshow('img1', img1)
    #cv2.imshow('img2', img2)
    # image = cv2.addWeighted(img1,alpha,img2,meta,gamma)
    # image = cv2.add(img1, img2)
    # img2 = cv2.bitwise_not(img2)
    #
    #
    #
    # cv2.imshow('img2', img2)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = cv2.bitwise_and(img1, img2)

    # cv2.imshow('image', image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite('./MASK/'+imgname1,image)
if __name__ == '__main__':
    imgall = read_img()
