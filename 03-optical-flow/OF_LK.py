import cv2
import numpy as np

# Paramètres Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture("personnes_en_mouvement.mp4")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Détecter des points à suivre (Shi-Tomasi)
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Masque pour dessiner les trajectoires
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul du flot optique Lucas-Kanade
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

    # Sélection des points valides
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Dessiner les trajectoires
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
        frame = cv2.circle(frame, (int(a), int(b)), 3, color=(0, 0, 255), thickness=-1)

    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow - Lucas-Kanade', img)

    # Mise à jour pour la prochaine itération
    prev_gray = next_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Sortie si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
