import os
from PIL import Image


def image_processing():
    #  待处理图片路径下的所有文件名字
    all_file_names = os.listdir('./expand/')
    for file_name in all_file_names:
        #  待处理图片路径
        img_path = Image.open(f'./expand/{file_name}')
        #  resize图片大小，入口参数为一个tuple，新的图片的大小
        img_size = img_path.resize((3840, 3840))
        #  处理图片后存储路径，以及存储格式
        img_size.save(f'./expand/{file_name}', 'JPEG')


if __name__ == '__main__':
    image_processing()