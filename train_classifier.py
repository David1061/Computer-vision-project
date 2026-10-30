import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
sessions = np.asarray(data_dict.get('sessions', ['legacy'] * len(labels)))

# Hold out whole recording sessions for testing (frames from the same session are
# near-duplicates, so a random split gives an overly optimistic score).
rng = random.Random(42)
np_rng = np.random.RandomState(42)
test_mask = np.zeros(len(labels), dtype=bool)
weak_classes = []

for cls in np.unique(labels):
    idx = np.where(labels == cls)[0]
    cls_sessions = sorted(set(sessions[idx]))

    if len(cls_sessions) >= 2:
        n_hold = max(1, round(0.2 * len(cls_sessions)))
        held = rng.sample(cls_sessions, n_hold)
        test_mask[idx[np.isin(sessions[idx], held)]] = True
        print(f'class {cls}: holding out {n_hold}/{len(cls_sessions)} session(s) for testing')
    else:
        weak_classes.append(str(cls))
        n_test = max(1, int(0.2 * len(idx)))
        test_mask[np_rng.choice(idx, n_test, replace=False)] = True
        print(f'class {cls}: only 1 session, using a random frame split (optimistic score)')

x_train, y_train = data[~test_mask], labels[~test_mask]
x_test, y_test = data[test_mask], labels[test_mask]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print('\n{:.1f}% of held-out samples were classified correctly!'.format(
    accuracy_score(y_test, y_predict) * 100))
print(classification_report(y_test, y_predict, zero_division=0))
if weak_classes:
    print(f'NOTE: classes {weak_classes} have only one session, so their score is '
          'optimistic. Record more sessions for them.')

# Final model: retrain on ALL data (more variety) before saving
model.fit(data, labels)
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print('Saved model.p (trained on all data)')
