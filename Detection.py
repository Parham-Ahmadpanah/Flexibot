import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import random
import cv2
import warnings
from threading import Thread
import time

warnings.simplefilter('ignore')
import ultralytics
from ultralytics import YOLO

########Change these
camera_port ='data/video2.avi'#'Data\Videos\output6.mp4' #'Data/videos/test.mp4' #'
realtime = True#False
model = YOLO("trains/train3/weights/best.pt")
# model = model.load("runs/detect/train/weights/best.pt")
########

# Function to perform object detection on an image
def detect_objects(image):
    yolo_outputs = model.predict(image, iou=0.1)
    output = yolo_outputs[0]
    box = output.boxes
    names = output.names
    detections = []
    for j in range(len(box)):
        labels = names[box.cls[j].item()]
        coordinates = box.xyxy[j].tolist()
        confidence = np.round(box.conf[j].item(), 2)
        detection = {
            'label': labels,
            'coordinates': coordinates,
            'confidence': confidence
        }
        detections.append(detection)
        print(detection)
    return detections

# def capture

# Function to read frames from webcam and perform object detection
def webcam_object_detection():
    cap = cv2.VideoCapture(camera_port)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('Data/Output/output_video.mp4', fourcc, fps, (width, height))
    
    # selected_points = gz.read_points_from_file('Weights/selected_points.txt')
    ii=1
    while True:
        ret, frame = cap.read()
        if not ret:
            print('end')
            break
        
        if ii % 2 != 0:
            pass
        ii = ii + 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        detections = detect_objects(frame_rgb)

        # nms_threshold = 0.4
        # indices = cv2.dnn.NMSBoxes(detection['coordinates'].tolist(), detection['confidence'].tolist(), score_threshold=0.5, nms_threshold=nms_threshold)

        image = frame_rgb
        heights = []
        for detection in detections:
            labels = detection['label']
            coordinates = detection['coordinates']
            if len(coordinates) >= 4:  # Ensure there are enough coordinates
                x1, y1, x2, y2 = coordinates
                height = y2 - y1  # Calculate height as the difference between Y-coordinates
                center_height = (y2 + y1) / 2 
                center_width = (x1+x2) /2 
                # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (250, 0, 0), 2)
                # cv2.putText(image, labels, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                ####
                if labels == 'cig butt': #and center_width > 168 and center_width < 501:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (250, 0, 0), 2)
                    cv2.putText(image, labels, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1)
                elif labels == 'end effector': #and center_width > 168 and center_width < 501:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 250), 2)
                    cv2.putText(image, labels, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
                            
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Real-time Object Detection', image )
        
        output_video.write(image)
        if realtime == False:
            key = cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

webcam_object_detection()

print('ok')