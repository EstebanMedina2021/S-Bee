import numpy as np
import cv2
import os
i = 0
indir='./Binarization/'
outdir='./expand'
def img_handle(img,outdir,i):
    image=cv2.imread(indir + img)
    kernel=np.ones((40,40),np.uint8)
    dilation=cv2.dilate(image,kernel)

    cv2.imwrite(outdir + '/' + str(i),dilation)

imlist=os.listdir(indir)
for img in imlist:
    imgname = os.path.split(img)[1]
    print(imgname)
    img_handle(img,outdir,imgname)