import os
import cv2
import time
import numpy as np
import argparse
from yolo import YOLO
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
   
source_video_path   = "videos/bees_into_hive.mp4"
save_video_path     = ""
video_fps           = 25
detection_threshold = 0.7

def load_model():
    model = YOLO()
    return model

#---------------------------------------------------#
# Arguments for the camera resolution
#---------------------------------------------------#
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def object_tracker_object():
    object_tracker = DeepSort(max_iou_distance=0.7,
                              max_age=30,
                              n_init=2,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.2,
                              nn_budget=None,
                              gating_only_position=False,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None
                              )
    return object_tracker

#---------------------------------------------------#
# Configuration of the video capture
#---------------------------------------------------#
def video_capture(source = 0):
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open video file.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    return cap

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
        result_tlwh = xyxy_to_tlwh(yolo_result[0])
        top, left, w, h = result_tlwh
        score = yolo_result[1]
        class_id = yolo_result[2]

        if score > detection_threshold:
            detections.append(([int(left), int(top), int(w), int(h)], score, class_id))
    
    return detections


def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    if len(boxes_xyxy.shape) == 1:  # Handling 1-dimensional array
        boxes_xywh[0] = (boxes_xyxy[0] + boxes_xyxy[2]) / 2.0  # center-x
        boxes_xywh[1] = (boxes_xyxy[1] + boxes_xyxy[3]) / 2.0  # center-y
        boxes_xywh[2] = boxes_xyxy[2] - boxes_xyxy[0]  # width
        boxes_xywh[3] = boxes_xyxy[3] - boxes_xyxy[1]  # height
    else:
        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0  # center-x
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0  # center-y
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height

    return boxes_xywh

def xyxy_to_tlwh(boxes):
  """Converts bounding boxes from xyxy (xmin, ymin, xmax, ymax) to tlwh format.

  Args:
      boxes: A NumPy array of bounding boxes in xyxy format. Each box is a 4-element list:
          [xmin, ymin, xmax, ymax]. Or, a single bounding box as a 4-element list.

  Returns:
      A NumPy array of bounding boxes in tlwh format. Each box is a 4-element list:
          [tl_x, tl_y, width, height]. Or, a single tlwh list if the input was a single box.
  """

  boxes = np.array(boxes)  # Ensure boxes is a NumPy array
  if boxes.ndim == 1:  # If it's a single bounding box (1 dimension)
    width = boxes[2] - boxes[0]
    height = boxes[3] - boxes[1]
    top_left_x = boxes[0]
    top_left_y = boxes[1]
    return np.array([top_left_x, top_left_y, width, height])
  else:  # If it's multiple bounding boxes (2 dimensions)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    top_left_x = boxes[:, 0]
    top_left_y = boxes[:, 1]
    return np.stack((top_left_x, top_left_y, width, height), axis=-1)

def draw_line(frame):
    p1 = (350, 50)
    p2 = (350, 600)
    frame = cv2.line(frame, p1, p2, (255, 0, 0), thickness=10)
    
    p3 = (1100, 50)
    p4 = (1100, 600)
    frame = cv2.line(frame, p3, p4, (255, 0, 0), thickness=10)

    return frame

def video_detection():
    count_line_in  = 0
    count_line_out = 0
    cache_id=[]

    cap = video_capture()
   
    object_tracker = object_tracker_object()
    model = load_model()

    while cap.isOpened():
        start = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            raise ValueError
        
        frame, results = tranform_frame(model, frame)

        detections = detections_assigner(results)

        frame = draw_line(frame)
        cv2.putText(frame, "Bee In: " + str(count_line_in), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 200), 6)
        cv2.putText(frame, "Bee Out: " + str(count_line_out), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 200), 6)
        tracks = object_tracker.update_tracks(detections, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            x = bbox[0]
            id = track_id

            if (x >= 400) and (id not in cache_id):  # lane 2
                cache_id.append(id)
                count_line_in = count_line_in + 1
                cv2.putText(frame, "Lane In: " + str(count_line_in), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 200), 6)


        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time


        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Video',frame)

        c= cv2.waitKey(1) & 0xff 
        if c==27:
            cap.release()
            break

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detection()