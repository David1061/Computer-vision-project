import os
import pickle
import sys

import cv2
import mediapipe as mp

from features import extract_features

# Usage:  python create_dataset.py                      -> ./data  -> data.pickle
#         python create_dataset.py data_test test.pickle
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else './data'
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else 'data.pickle'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data, labels, sessions = [], [], []

# Layout: data/<class>/<session timestamp>/<image>.jpg
# Images placed directly in data/<class>/ (old layout) are treated as session "legacy".
for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    kept = skipped = 0
    class_sessions = set()

    for root, _, files in sorted(os.walk(class_path)):
        rel = os.path.relpath(root, class_path)
        session = 'legacy' if rel == '.' else rel

        for img_name in sorted(files):
            if not img_name.lower().endswith(IMAGE_EXTS):
                continue
            img = cv2.imread(os.path.join(root, img_name))
            if img is None:
                continue

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            features, _ = extract_features(results.multi_hand_landmarks)
            if features is None:
                skipped += 1
                continue

            data.append(features)
            labels.append(class_name)
            sessions.append(session)
            class_sessions.add(session)
            kept += 1

    print(f'  class {class_name}: {kept} kept, {skipped} skipped (no hand), '
          f'{len(class_sessions)} session(s)')

if not data:
    raise SystemExit('No samples found. Record some data with collect_imgs.py first.')

with open(OUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'sessions': sessions}, f)

print(f'Saved {len(data)} samples to {OUT_FILE}')