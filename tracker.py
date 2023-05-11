from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy
import pandas as pd

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model/FNF12102022_v7.pt', trust_repo= False)

def track(frame):
    detection = model(frame)
    tracker = DeepSort(max_age=30)
    num= detection.xyxy[0].numpy().shape[0]
    boxes=[]
    id=[]
    detections=[]
    for i in range (num):
        
        x= int(detection.xyxy[0].numpy()[i][0])
        y = int(detection.xyxy[0].numpy()[i][1])
        w= int(detection.xyxy[0].numpy()[i][2])
        h=int(detection.xyxy[0].numpy()[i][3]) 
        conf = detection.xyxy[0].numpy()[i][4]
        cl = int(detection.xyxy[0].numpy()[i][5])

        detection1= ([x, y, w, h], conf, cl)
        detections.append(detection1)
    tracks = tracker.update_tracks(detections, frame= frame)

    for track in tracks:
        boxes.append(track.to_tlwh())
        id.append(track.track_id)
    return boxes, id