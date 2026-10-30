import os
import sys
import time

import cv2
import mediapipe as mp

from features import LABELS

# Usage:  python collect_imgs.py            -> saves into ./data
#         python collect_imgs.py data_test  -> saves into ./data_test (separate test set)
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else './data'
CAMERA_INDEX = 1             # 0 = default camera; try 1 or 2 if it's the wrong one
IMAGES_PER_CLASS = 200       # per class, per session
SAVE_EVERY_N_FRAMES = 4      # spacing frames out gives more variety than back-to-back shots
COUNTDOWN_SECONDS = 3
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prompts rotate during recording so you naturally vary the pose.
VARIATION_PROMPTS = [
    'Centered, normal distance',
    'Move CLOSER to the camera',
    'Move FARTHER from the camera',
    'Tilt / rotate your hand slightly',
    'Move to the LEFT side of the frame',
    'Move to the RIGHT side of the frame',
    'Slowly wiggle your fingers a little',
    'Switch hands / change wrist angle',
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAMERA_INDEX)


def draw_text(frame, lines, color=(0, 255, 0), scale=0.8, dy=34):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + dy * i), FONT, scale, color, 2, cv2.LINE_AA)


def count_images(class_id):
    """Count every image under data/<class>/ including all session folders."""
    total = 0
    for _, _, files in os.walk(os.path.join(DATA_DIR, str(class_id))):
        total += sum(f.lower().endswith(IMAGE_EXTS) for f in files)
    return total


def remove_if_empty(path):
    try:
        os.rmdir(path)  # only succeeds if the folder is empty (or raises if missing)
    except OSError:
        pass


def record_class(class_id):
    """Record one session for a class. Returns the number of images saved."""
    name = LABELS[class_id]
    class_dir = os.path.join(DATA_DIR, str(class_id))

    # Countdown (ESC cancels)
    end = time.time() + COUNTDOWN_SECONDS
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            return 0
        draw_text(frame, [f'Get ready: {name}', f'Starting in {int(end - time.time()) + 1}',
                          'ESC = cancel'], (0, 200, 255))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return 0

    # Each session gets its own time-named folder: data/<class>/<timestamp>/
    session_dir = os.path.join(class_dir, time.strftime('%Y-%m-%d_%H-%M-%S'))
    per_prompt = max(1, IMAGES_PER_CLASS // len(VARIATION_PROMPTS))
    saved = frame_idx = 0

    try:
        while saved < IMAGES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detected = bool(results.multi_hand_landmarks)

            # Only save frames with a detected hand; create the folder on the first save
            if detected and frame_idx % SAVE_EVERY_N_FRAMES == 0:
                os.makedirs(session_dir, exist_ok=True)
                if cv2.imwrite(os.path.join(session_dir, f'{saved}.jpg'), frame):
                    saved += 1

            display = frame.copy()
            if detected:
                for h in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display, h, mp_hands.HAND_CONNECTIONS)
            prompt = VARIATION_PROMPTS[min(saved // per_prompt, len(VARIATION_PROMPTS) - 1)]
            draw_text(display,
                      [f'{name}: {saved}/{IMAGES_PER_CLASS}', prompt,
                       '' if detected else 'NO HAND DETECTED - not saving',
                       'Q / ESC = stop this session'],
                      (0, 255, 0) if detected else (0, 0, 255))
            cv2.imshow('frame', display)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        # No data -> no folder. rmdir only removes folders that are empty.
        remove_if_empty(session_dir)
        remove_if_empty(class_dir)

    return saved


class_ids = sorted(LABELS)
selected = class_ids[0]
counts = {cid: count_images(cid) for cid in class_ids}
status = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lines = ['SELECT CLASS: number key, or A / D to cycle', '']
    for cid in class_ids:
        marker = '>' if cid == selected else ' '
        lines.append(f'{marker} [{cid}] {LABELS[cid]}   ({counts[cid]} images)')
    lines += ['', 'S / ENTER = start recording    ESC = quit']
    if status:
        lines += ['', status]

    panel_h = 40 + 34 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    draw_text(frame, lines, (255, 255, 255), scale=0.7)
    # highlight the selected class line
    row = 2 + class_ids.index(selected)
    cv2.putText(frame, lines[row], (20, 40 + 34 * row), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(25) & 0xFF
    if key == 27:
        break
    elif ord('0') <= key <= ord('9') and (key - ord('0')) in LABELS:
        selected = key - ord('0')
    elif key == ord('d'):
        selected = class_ids[(class_ids.index(selected) + 1) % len(class_ids)]
    elif key == ord('a'):
        selected = class_ids[(class_ids.index(selected) - 1) % len(class_ids)]
    elif key in (ord('s'), 13, 10):
        n = record_class(selected)
        counts[selected] = count_images(selected)
        status = (f'Saved {n} images for {LABELS[selected]}' if n
                  else 'No data saved, so no session folder was created')
        print(status)

cap.release()
cv2.destroyAllWindows()