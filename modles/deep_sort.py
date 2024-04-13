import os
import cv2
import time
import numpy as np

from yolo import YOLO
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
   
source_video_path   = "videos/"
save_video_path     = ""
video_fps           = 25
detection_threshold = 0.5     
    
def load_model():
    model = YOLO()
    return model


def object_tracker_object():
    object_tracker = DeepSort(max_age=5,
                    n_init=2,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3,
                    nn_budget=None,
                    override_track_class=None,
                    embedder="mobilenet",
                    half=True,
                    bgr=True,
                    embedder_gpu=True,
                    embedder_model_name=None,
                    embedder_wts=None,
                    polygon=False,
                    today=None)
    return object_tracker


def video_capture(source = 0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open video file.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def initialize_video_writer(save_path, frame_width, frame_height, fps):
    if save_path and save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (frame_width, frame_height)
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
        return out
    return None


def tranform_frame(model, frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    

    frame = Image.fromarray(np.uint8(frame))

    frame, results = model.detect_image(frame, supervision=True)    

    frame = np.array(frame)

    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    return frame, results


def detections_assigner(yolo_result):
    detections = []
    if len(yolo_result) != 0:
        x1, y1, x2, y2 = yolo_result[0]
        score = yolo_result[1]
        class_id = yolo_result[2]
        if score > detection_threshold:
            detections.append(([x1, y1, int(x2-x1), int(y2-y1)], score, class_id))
    
    return detections


def draw_tracked_objects(tracks, frame):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        
        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
        cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def video_detection():
    cap = video_capture()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_hight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = initialize_video_writer(save_video_path, frame_width, frame_hight, video_fps)

    object_tracker = object_tracker_object()
    model = load_model()

    while cap.isOpened():
        start = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            raise ValueError
        
        frame, results = tranform_frame(model, frame)

        detections = detections_assigner(results)
            
        tracks = object_tracker.update_tracks(detections, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        
        draw_tracked_objects(tracks, frame)
            
        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime


        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Video',frame)

        c= cv2.waitKey(1) & 0xff 
        if save_video_path != "":
            out.write(frame)

        if c==27:
            cap.release()
            break

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detection()