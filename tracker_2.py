from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
import pandas as pd
import cv2
import os

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'FNF12102022_v7.pt', trust_repo= False)

def track(frame):
    detection = model(frame)
    tracker = DeepSort(max_age=30, embedder="torchreid")
    num= detection.xyxy[0].numpy().shape[0]
    boxes=[]
    id=[]
    detections=[]
    confidence = []
    if num!=0:
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
            print(track.track_id)
            print(track.to_tlwh())
            boxes.append(track.to_tlwh())
            id.append(track.track_id)
            confidence.append(track.det_conf)
    return boxes, id, confidence


cap1 = cv2.VideoCapture('0.mp4')

ret, frame=cap1.read()
size= frame.shape[:2]


out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 4, 
                    (frame.shape[1], frame.shape[0]))

color_code=[(235, 52, 52),
            (52, 64, 235),
            (3, 148, 252)]  
fr_num=1
cap = cv2.VideoCapture('0.mp4')
with open('tracking_deep.txt', 'w') as m_file:
    while True:
        ret, frame = cap.read()
            
        boxes, id, confi = track(frame)
        
        k = cv2.waitKey(1) & 0xff
        if ret==True:
            print(fr_num)
            print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for bbox1, id1, con1 in zip(boxes, id, confi):
                p11=  (int(bbox1[0]), int(bbox1[1]))
                p22=  (int(bbox1[2])), int(bbox1[3])
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(frame, p11, p22, color_code[int(float(id1))-1], 2, 1)
                
                cv2.putText(frame, id1, (int(bbox1[0]), int(bbox1[3]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code[int(float(id1))-1], 2)
                
                cv2.imshow('frame', frame)
                list1= [fr_num, int(float(id1)), bbox1[0], bbox1[1], bbox1[2]-bbox1[0], bbox1[3]-bbox1[1], con1, -1, -1, -1]
                text= ", ".join(str(num) for num in list1)
                pth= os.path.join('images_bb', str(fr_num)+'_2.jpg')
                print(pth)
                cv2.imwrite(pth, frame)
                print(text)
                m_file.write(str(text)+'\n')
            fr_num+=1
            out.write(frame)

        if k == 27 : break
m_file.close()

cap.release()
out.release()

