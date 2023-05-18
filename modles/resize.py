import os
from PIL import Image


def image_processing():
  
    all_file_names = os.listdir('./expand/')
    for file_name in all_file_names:
        
        img_path = Image.open(f'./expand/{file_name}')
        
        img_size = img_path.resize((3840, 3840))
        
        img_size.save(f'./expand/{file_name}', 'JPEG')


if __name__ == '__main__':
    image_processing()
