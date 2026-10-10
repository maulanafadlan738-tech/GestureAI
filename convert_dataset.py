import os
import cv2
import mediapipe as mp
import csv

DATASET_PATH = "/home/fadlan/dataset"
OUTPUT_CSV = "asl_data.csv"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

with open(OUTPUT_CSV, mode="w", newline="") as f:

    writer = csv.writer(f)

    for label in os.listdir(DATASET_PATH):

        folder = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder):
            continue

        print("Processing:", label)

        for img_name in os.listdir(folder):

            if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    data = []

                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    # NORMALISASI
                    base_x = data[0]
                    base_y = data[1]
                    base_z = data[2]

                    normalized = []

                    for i in range(0, len(data), 3):

                        normalized.append(data[i] - base_x)
                        normalized.append(data[i+1] - base_y)
                        normalized.append(data[i+2] - base_z)

                    normalized.append(label)

                    writer.writerow(normalized)

print("SELESAI")
