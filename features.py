"""Shared feature extraction so training and inference can never drift apart.

Per hand we build:
  - 21 landmarks (x, y, z) relative to the wrist, scaled by palm size   -> 63
  - finger extension ratios (tip distance / knuckle distance from wrist) -> 5
  - finger bend (cosine of the angle at the middle joint)                -> 5
  - fingertip-to-neighbouring-fingertip distances (finger spread)        -> 4
Then for the whole frame: wrist-to-wrist offset (2) + two-hands flag (1).
"""
import numpy as np

NUM_HANDS = 2
HAND_FEATS = 77
NUM_FEATURES = NUM_HANDS * HAND_FEATS + 3  # 157

# Folder name (class id) -> text shown on screen.
LABELS = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'Unknown',   # open palm, relaxed hand, random gestures
}
UNKNOWN_CLASS_IDS = {3}

TIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]
# joint triples (base, middle, tip) used for the bend angle
JOINTS = [(2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


def _cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))


def _hand_features(h):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in h.landmark])
    pts = pts - pts[0]                       # wrist at origin
    palm = np.linalg.norm(pts[9]) or 1e-6    # wrist -> middle knuckle
    pts = pts / palm

    feats = list(pts.flatten())                                            # 63
    for tip, mcp in zip(TIPS, MCPS):                                       # 5
        feats.append(float(np.linalg.norm(pts[tip]) / (np.linalg.norm(pts[mcp]) + 1e-6)))
    for a, b, c in JOINTS:                                                 # 5
        feats.append(_cos_angle(pts[a], pts[b], pts[c]))
    for t1, t2 in zip(TIPS[:-1], TIPS[1:]):                                # 4
        feats.append(float(np.linalg.norm(pts[t1] - pts[t2])))
    return feats


def extract_features(multi_hand_landmarks):
    """Return (features, (x_min, y_min, x_max, y_max)) or (None, None) if no hands."""
    if not multi_hand_landmarks:
        return None, None

    # Consistent hand order: left-most wrist first, max 2 hands
    hands_lms = sorted(multi_hand_landmarks, key=lambda h: h.landmark[0].x)[:NUM_HANDS]

    features = []
    for h in hands_lms:
        features += _hand_features(h)
    features += [0.0] * (NUM_HANDS * HAND_FEATS - len(features))  # pad missing hand

    if len(hands_lms) == 2:
        w0, w1 = hands_lms[0].landmark[0], hands_lms[1].landmark[0]
        size = max(
            np.hypot(h.landmark[9].x - h.landmark[0].x, h.landmark[9].y - h.landmark[0].y)
            for h in hands_lms) or 1e-6
        features += [(w1.x - w0.x) / size, (w1.y - w0.y) / size, 1.0]
    else:
        features += [0.0, 0.0, 0.0]

    xs = [lm.x for h in hands_lms for lm in h.landmark]
    ys = [lm.y for h in hands_lms for lm in h.landmark]
    return features, (min(xs), min(ys), max(xs), max(ys))
