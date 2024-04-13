import streamlit as st
from PIL import Image
from yolo import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = YOLO()

st.title('Object Detector :bee:')

begin_remote_camera = st.button("Start Video Capture")

if begin_remote_camera:
    camera_reference = st.text_input('Insert Video Camera reference', '0')

    if camera_reference is not None:
        cap = cv2.VideoCapture(int(camera_reference))

        frame_placeholder = st.empty()

        stop_button_pressed = st.button("Stop Video Capture")

        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended.")
                break
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
           
            frame = Image.fromarray(np.uint8(frame))
            
            frame = np.array(model.detect_image(frame))

            frame_placeholder.image(frame, channels='RGB')

            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
    
        cap.release()
        cv2.destroyAllWindows()
    else:
        st.error('Must input the reference for the camera', icon="🚨")

        


uploaded_iamge = st.file_uploader(label="Upload your image here", type=["png","jpg","jpeg"])
if uploaded_iamge:
    img = Image.open(uploaded_iamge)
    
    image_prediction = model.detect_image(img)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(image_prediction)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)