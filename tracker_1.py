import torch
import numpy as np
import pandas as pd
import cv2
from strongsort import StrongSORT

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'FNF12102022_v7.pt', trust_repo= False)

def track(frame):
    detection = model(frame)
    print(frame.shape)
    tracker = StrongSORT( device=torch.device("cpu"), fp16=False)
    num= detection.xyxy[0].numpy().shape[0]
    boxes=[]
    id=[]
    detections=[]
    confi=[]
    for i in range (num):
        
        x1= int(detection.xyxy[0].numpy()[i][0])
        y1 = int(detection.xyxy[0].numpy()[i][1])
        x2= int(detection.xyxy[0].numpy()[i][2])
        y2=int(detection.xyxy[0].numpy()[i][3]) 
        conf = detection.xyxy[0].numpy()[i][4]
        cl = int(detection.xyxy[0].numpy()[i][5])

        detection1= [x1, y1, x2-x1, y2-y1, conf, cl]
        detections.append(detection1)
        
    tracks = tracker.update(np.array(detections), ori_img=frame)

    for track in tracks:
        boxes.append(track[0:4])
        id.append(str(track[4]))
        confi.append(track[6])
    return boxes, id, confi

cap1 = cv2.VideoCapture('multitrack.mp4')

ret, frame=cap1.read()
size= frame.shape[:2]


out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 4, 
                    (frame.shape[1], frame.shape[0]))

color_code=[]  
for i in range(30):
    color = list(np.random.random(size=3) * 256)
    color_code.append(color)
fr_num= 1

cap = cv2.VideoCapture('multitrack.mp4')
with open('tracking_strong.txt', 'w') as m_file:
    while True:
        ret, frame = cap.read()
            
        boxes, id, confi = track(frame)
        
        k = cv2.waitKey(1) & 0xff
        if ret==True:
            print(fr_num)
            print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for bbox1, id1, con1 in zip(boxes, id, confi):
                p11=  (int(bbox1[0]), int(bbox1[1]))
                p22=  (int(bbox1[2])+int(bbox1[0]), int(bbox1[3])+ int(bbox1[1]))
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(frame, p11, p22, color_code[int(float(id1))], 2, 1)
                
                cv2.putText(frame, id1, (int(bbox1[0]), int(bbox1[3])+ int(bbox1[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code[int(float(id1))], 2)
                
                cv2.imshow('frame', frame)
                list1= [fr_num, int(float(id1)), bbox1[0], bbox1[1], bbox1[2], bbox1[3], con1, -1, -1, -1]
                text= ", ".join(str(num) for num in list1)
                
                
                m_file.write(str(text)+'\n')
                
            fr_num+=1
            out.write(frame)

        if k == 27 : break
m_file.close()

cap.release()
out.release()

