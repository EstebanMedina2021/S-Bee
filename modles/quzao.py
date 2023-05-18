import numpy as np
import cv2
import os
i = 0
indir='./54/'
outdir='./55'
def img_handle(img,outdir,i):
    image=cv2.imread(indir + img)
    kernel=np.ones((3,3),np.uint8)
    erosion=cv2.erode(image,kernel)

    cv2.imwrite(outdir + '/' + str(i),erosion)

imlist=os.listdir(indir)
for img in imlist:
    imgname = os.path.split(img)[1]
    print(imgname)
    img_handle(img,outdir,imgname)