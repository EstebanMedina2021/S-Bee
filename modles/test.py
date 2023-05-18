#######t.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
# from BagData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = torch.load('checkpoints/fcn_model.pt')  # 加载模型
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_img_list(dir, firelist, ext=None):
    newdir = dir
    if os.path.isfile(dir):  # 如果是文件
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  # 如果是目录
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)

    return firelist

def read_img():
    image_path = './fill/'
    imglist = get_img_list(image_path, [], 'jpg')
    imgall = []
    for imgpath in imglist:
        # print(imgpath)
        imgname = os.path.split(imgpath)[1]  # 分离文件路径和文件名后获取文件名（包括了后缀名）
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
    if __name__ == '__main__':
        # img_name = r'./test/0.jpg'  # 预测的图片
        # imgA = cv2.imread(img_name)
        # print(imgA.shape)
        # x, y = imgA.shape[0:2]
        # imgA = cv2.resize(imgA, (int(y / 20), int(x / 20)))
        print(img.shape)
        img = cv2.resize(img, (480, 480))

        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        output = model(img)
        output = torch.sigmoid(output)

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        print(output_np.shape)  # (1, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)
        print(output_np.shape)  # (1,160, 160)
        ret = np.squeeze(output_np[0, ...])

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(ret, 'gray')
        # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
        plt.imsave('./test/' + (str)(i), ret)
        # plt.show()
    # plt.pause(3)
    # cv2.imwrite('./result/' + (str)(i), ret)
    # plt.imsave('./img/Snapshot001.jpg',constant)
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    imgall = read_img()