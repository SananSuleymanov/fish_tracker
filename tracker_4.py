from trackers.ocsort.ocsort import OCSort
import torch
import cv2
import numpy as np
import os

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'FNF12102022_v7.pt', trust_repo= False)

def track(frame):
    detection = model(frame)
    tracker = OCSort(det_thresh=0.5)
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

            detection1= [x, y, w, h, conf, cl]
            detections.append(detection1)

        tracks = tracker.update(np.array(detections), '')

        for track in tracks:
            
            boxes.append(track[0:4])
            id.append(str(track[4]))
            confidence.append(track[6])
    return boxes, id, confidence


cap1 = cv2.VideoCapture('0.mp4')

ret, frame=cap1.read()
size= frame.shape[:2]


color_code=[(235, 52, 52),
            (52, 64, 235),
            (3, 148, 252)]  
fr_num=1
cap = cv2.VideoCapture('0.mp4')
with open('tracking_oc_3.txt', 'w') as m_file:
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
                pth= os.path.join('images_bb', str(fr_num)+'.jpg')
                print(pth)
                cv2.imwrite(pth, frame)
                print(text)
                m_file.write(str(text)+'\n')
                
            fr_num+=1
           

        if k == 27 : break
m_file.close()

cap.release()


