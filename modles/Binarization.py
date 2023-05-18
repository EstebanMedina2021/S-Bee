import numpy as np
import cv2
import os

i = 0
indir='./test/'
outdir='./Binarization'
def img_handle(img,outdir,i):

    image=cv2.imread(indir + '/' + img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t, rst = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 

    # dst = cv2.dilate(rst,kernel=1,anchor=None,iterations=None)




    cv2.imwrite(outdir + '/' + str(i),rst)

imlist=os.listdir(indir)
for img in imlist:
    imgname = os.path.split(img)[1]
    print(imgname)
    img_handle(img,outdir,imgname)
