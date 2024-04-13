import supervision as sv
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
if __name__ == "__main__":
    yolo = YOLO()
    SOURCE_VIDEO_PATH = "videos/bees_into_hive.mp4"
    # 0: Varroa, 1: Bee
    selected_classes = [0]
    # create frame generator
    generator = sv.get_video_frames_generator(0)
    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    # acquire first video frame
    iterator = iter(generator)
    frame = next(iterator)
    frame = Image.fromarray(np.uint8(frame))

    
    #detections = sv.Detections.from_tensorflow(sv_detection_object)
    detections = yolo.detect_image(frame, supervision=True)
    detections = detections[np.isin(detections[2], selected_classes)]

    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]

    # annotate and display frame
    anotated_frame=box_annotator.annotate(scene=frame, detections=detections, labels=labels)


    # settings
    LINE_START = sv.Point(50, 1500)
    LINE_END = sv.Point(3840-50, 1500)

    TARGET_VIDEO_PATH = f"videos/vehicle-counting-result-with-counter.mp4"

    sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create BYTETracker instance
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    # create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # create LineZone instance, it is previously called LineCounter class
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # create instance of TraceAnnotator
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

    # create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # define call back function to be used in video processing
    def callback(frame: np.ndarray, index:int) -> np.ndarray:
        # model prediction on single frame and conversion to supervision Detections
        results = yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # only consider class id from selected_classes define above
        detections = detections[np.isin(detections.class_id, selected_classes)]
        # tracking detections
        detections = byte_tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} Bee {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        annotated_frame = trace_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame=box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

        # update line counter
        line_zone.trigger(detections)
        # return frame with box and line annotated result
        return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # process the whole video
    sv.process_video(
        source_path = SOURCE_VIDEO_PATH,
        target_path = TARGET_VIDEO_PATH,
        callback=callback
    )