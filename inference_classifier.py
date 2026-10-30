import pickle

import cv2
import mediapipe as mp
import numpy as np

from features import extract_features, LABELS, UNKNOWN_CLASS_IDS

CONFIDENCE_THRESHOLD = 0.85 # tune: raise if wrong signs still get labeled, lower if real signs show '?'
CAMERA_INDEX = 1            # 0 = default camera; try 1 or 2 if it's the wrong one

model = pickle.load(open('./model.p', 'rb'))['model']

cap = cv2.VideoCapture(CAMERA_INDEX)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        features, (min_x, min_y, max_x, max_y) = extract_features(results.multi_hand_landmarks)

        proba = model.predict_proba([np.asarray(features)])[0]
        best = int(np.argmax(proba))
        confidence = float(proba[best])
        class_id = int(model.classes_[best])

        if confidence >= CONFIDENCE_THRESHOLD and class_id not in UNKNOWN_CLASS_IDS:
            text = f'{LABELS.get(class_id, str(class_id))} {confidence:.0%}'
            color = (0, 200, 0)
        else:
            text = f'? {confidence:.0%}'
            color = (0, 0, 255)

        x1, y1 = int(min_x * W) - 10, int(min_y * H) - 10
        x2, y2 = int(max_x * W) + 10, int(max_y * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, text, (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()