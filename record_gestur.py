# ==============================
# FILE: record_gestur.py
# ==============================

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import mediapipe as mp
import csv
import pickle
import numpy as np

app = Flask(__name__)

# mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# auto detect camera
def find_camera(max_index=3):

    for i in range(max_index):

        cap = cv2.VideoCapture(i)

        if cap.isOpened():

            cap.release()

            return i

    return None

camera_index = find_camera()

if camera_index is None:
    raise RuntimeError("Camera tidak ditemukan")

cap = cv2.VideoCapture(camera_index)

CSV_FILE = "asl_data.csv"

record_label = ""
recording = False

# load model
try:

    with open("asl_model.pkl", "rb") as f:
        model = pickle.load(f)

except:
    model = None

# fungsi prediksi
def predict_gesture(landmarks):

    if model is None:
        return ""

    raw_data = []

    for lm in landmarks:
        raw_data.extend([lm.x, lm.y, lm.z])

    # normalisasi
    base_x = raw_data[0]
    base_y = raw_data[1]
    base_z = raw_data[2]

    data = []

    for i in range(0, len(raw_data), 3):

        data.append(raw_data[i] - base_x)
        data.append(raw_data[i+1] - base_y)
        data.append(raw_data[i+2] - base_z)

    data = np.array(data).reshape(1, -1)

    proba = model.predict_proba(data)[0]

    confidence = max(proba)

    prediction = model.classes_[proba.argmax()]

    if confidence < 0.80:
        return ""

    return str(prediction)

# video stream
def gen_frames():

    global record_label, recording

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        text_output = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # draw landmark
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
                )

                # prediksi realtime
                text_output = predict_gesture(hand_landmarks.landmark)

                # recording
                if recording and record_label:

                    raw_data = []

                    for lm in hand_landmarks.landmark:
                        raw_data.extend([lm.x, lm.y, lm.z])

                    # normalisasi
                    base_x = raw_data[0]
                    base_y = raw_data[1]
                    base_z = raw_data[2]

                    data_row = []

                    for i in range(0, len(raw_data), 3):

                        data_row.append(raw_data[i] - base_x)
                        data_row.append(raw_data[i+1] - base_y)
                        data_row.append(raw_data[i+2] - base_z)

                    data_row.append(record_label)

                    with open(CSV_FILE, "a", newline="") as f:

                        writer = csv.writer(f)

                        writer.writerow(data_row)

                # tampil prediksi
                if text_output:

                    cv2.putText(
                        frame,
                        "Detected: " + text_output,
                        (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        3
                    )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# route video
@app.route('/video_feed')
def video_feed():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# UI
@app.route('/')
def index():

    return render_template_string("""

<html>

<head>
<title>Gesture AI Recorder</title>
</head>

<body>

<h2>Gesture AI Recorder</h2>

<img src="/video_feed" width="700">

<br><br>

<input type="text" id="letter" placeholder="Masukkan huruf">

<button onclick="startRec()">Start Recording</button>

<button onclick="stopRec()">Stop Recording</button>

<p id="status">Status: idle</p>

<script>

function startRec(){

    const letter = document.getElementById("letter").value

    fetch("/start",{
        method:"POST",
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({letter:letter})
    })
    .then(r=>r.json())
    .then(data=>{
        document.getElementById("status").innerText=data.status
    })

}

function stopRec(){

    fetch("/stop",{method:"POST"})
    .then(r=>r.json())
    .then(data=>{
        document.getElementById("status").innerText=data.status
    })

}

</script>

</body>

</html>

""")

# start recording
@app.route('/start', methods=['POST'])
def start_recording():

    global record_label, recording

    data = request.get_json()

    record_label = data.get("letter","").upper()

    if record_label:

        recording = True

        return jsonify(status=f"Recording {record_label}")

    return jsonify(status="Masukkan huruf dulu")

# stop recording
@app.route('/stop', methods=['POST'])
def stop_recording():

    global recording

    recording = False

    return jsonify(status="Recording stopped")

# run
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
