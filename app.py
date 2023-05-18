from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify
import cv2
import tempfile
import time
import numpy as np
import os
import json
from tracker import track
import requests

app = Flask(__name__)


@app.route('/run', methods=['GET'])
def run_video():
    video_link= request.args.get('video_link')
    response= requests.get(video_link)

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(response.content)
    video_path = request.args.get(temp.name)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    data={}
    count=1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, id = track(frame)
        

        if ret:
            for bbox1, id1 in zip(bboxes, id):
                p11=  (int(bbox1[0]), int(bbox1[1]))
                p22=  (int(bbox1[2]), int(bbox1[3]))
                
                data[count] = [p11, p22, id1]
            count+=1
    downloads_folder = os.path.expanduser("~/Downloads")
    file_path = os.path.join(downloads_folder, "data.json")

    with open(file_path, "w") as f:
        json.dump(data, f)

    return jsonify(data)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded video file
        video_file = request.files['video_file']

        # Save the video file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(video_file.read())
            temp_file_path = temp.name

        return render_template('video.html', video_path=temp_file_path)

    return render_template('index.html')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    # Get the temporary file path
    video_path = request.args.get('video_path')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    

    # Check if video was opened successfully
    if not cap.isOpened():
        return redirect(url_for('index'))

    # Set the response header
    return Response(generate_frames(cap=cap, fps=fps), mimetype='multipart/x-mixed-replace; boundary=frame')
color_code=[]
for i in range(30):
    color = list(np.random.random(size=3) * 256)
    color_code.append(color)
fr_num= 1

def generate_frames(cap, fps):
    ret, frame = cap.read()

    frame_width = frame[0].shape[1]
    frame_height= frame[0].shape[0]
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, 
                      (frame_width,frame_height))


    frame_time= 1000/fps
    data= {}
    count=1
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            try:
                bboxes, id = track(frame)
            except:
                print("Problem in tracking")

            timer = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        

            for bbox1, id1 in zip(bboxes, id):
                p11=  (int(bbox1[0]), int(bbox1[1]))
                p22=  (int(bbox1[2]), int(bbox1[3]))
                cv2.rectangle(frame, p11, p22, color_code[int(float(id1))], 2, 1)
                label_size, baseline = cv2.getTextSize(id1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                data[count] = [p11, p22, id1]
                if bbox1[1] - label_size[1] < 0:
                   
                    cv2.putText(frame, id1, (int(bbox1[0]), int(bbox1[3]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code[int(float(id1))], 2)
                    
                else:
                    
                    cv2.putText(frame, id1, (int(bbox1[0]), int(bbox1[3]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code[int(float(id1))], 2)
            
            count+=1

            ret, jpeg = cv2.imencode('.jpg', frame)
        
            out.write(frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            
        else:
            file_path='data.json'
            with open(file_path, "w") as f:
                json.dump(data, f)
       
        
        #time.sleep(frame_time / 1000.0)


    #send_file(data, as_attachment=True, attachment_filename="data.json" )


if __name__ == '__main__':
    app.run(debug=True)
