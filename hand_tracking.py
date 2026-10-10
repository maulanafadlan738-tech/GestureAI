# ==============================
# FILE: hand_tracking.py
# ==============================

from flask import Flask, Response
import cv2
import mediapipe as mp
import pickle
import time
import pyttsx3

app = Flask(__name__)

# load model
with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

# text to speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# kamera
cap = cv2.VideoCapture(0)

# variabel kata
word = ""
last_letter = ""
last_time = 0

# stabilizer
pred_buffer = []
BUFFER_SIZE = 10

def gen_frames():

    global word, last_letter, last_time, pred_buffer

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        letter = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # draw hand
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
                )

                raw_data = []

                # ambil landmark
                for lm in hand_landmarks.landmark:
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

                # prediksi confidence
                proba = model.predict_proba([data])[0]

                confidence = max(proba)

                prediction = model.classes_[proba.argmax()]

                # filter confidence
                if confidence >= 0.80:

                    pred_buffer.append(prediction)

                    if len(pred_buffer) > BUFFER_SIZE:
                        pred_buffer.pop(0)

                    # ambil prediksi stabil
                    letter = max(set(pred_buffer), key=pred_buffer.count)

                    current_time = time.time()

                    # input huruf
                    if letter != last_letter and current_time - last_time > 1.5:

                        word += letter

                        last_letter = letter

                        last_time = current_time

                    # auto spasi
                    if current_time - last_time > 3:

                        if not word.endswith(" "):
                            word += " "

        # tampil detected
        cv2.putText(
            frame,
            "Detected: " + letter,
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        # tampil word
        cv2.putText(
            frame,
            "Word: " + word,
            (10,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/video_feed')
def video_feed():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/reset')
def reset_word():

    global word, last_letter

    word = ""
    last_letter = ""

    return "OK"

@app.route('/speak')
def speak_word():

    global word

    if word != "":
        engine.say(word)
        engine.runAndWait()

    return "OK"

@app.route('/')
def index():

    return """

<html>

<head>

<title>Sign Language Translator</title>

<style>

body{
font-family:Arial;
background:#f4f4f4;
margin:0;
}

.header{
background:#333;
color:white;
padding:15px;
text-align:center;
font-size:22px;
}

.container{
display:flex;
justify-content:center;
margin-top:30px;
}

.camera{
margin-right:30px;
}

.camera img{
border:2px solid #ddd;
border-radius:6px;
}

.panel{
width:300px;
}

.box{
background:white;
padding:20px;
border:1px solid #ddd;
border-radius:6px;
margin-bottom:20px;
font-size:22px;
text-align:center;
}

button{
width:100%;
padding:12px;
margin-bottom:10px;
font-size:16px;
border:none;
border-radius:4px;
cursor:pointer;
}

.reset{
background:#e74c3c;
color:white;
}

.speak{
background:#2ecc71;
color:white;
}

button:hover{
opacity:0.9;
}

</style>

</head>

<body>

<div class="header">
Sign Language Translator
</div>

<div class="container">

<div class="camera">
<img src="/video_feed" width="640">
</div>

<div class="panel">

<div class="box">
Gesture result appears on camera
</div>

<button class="reset" onclick="fetch('/reset')">
Reset Word
</button>

<button class="speak" onclick="fetch('/speak')">
Speak
</button>

</div>

</div>

</body>

</html>

"""

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
