
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
   
    mode = "video"
  
    crop            = False
    count           = False
   
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
   
    dir_origin_path = "MASK/"
    
    dir_save_path   = "img_out/"
    
   
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
       
        while True:
            img = 'direct_predict/'
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="" and video_save_path==0:
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError
        fps = 0.0
        while(True):
            t1 = time.time()
            
            ref, frame = capture.read()
            if not ref:
                break
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
           
            frame = Image.fromarray(np.uint8(frame))
            
            frame = np.array(yolo.detect_image(frame))
            
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        capture.release()
        if video_save_path!="":
            out.release()
        cv2.destroyAllWindows()


    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0) 
    else:
        raise AssertionError("Please specify a mode")