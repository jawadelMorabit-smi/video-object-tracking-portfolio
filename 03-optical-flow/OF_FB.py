import cv2
import numpy as np

def create_hsv_color_wheel(diameter=100):
    """Crée un disque de couleur HSV pour visualiser les directions du flot."""
    wheel = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    cx, cy = diameter // 2, diameter // 2
    for y in range(diameter):
        for x in range(diameter):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx*dx + dy*dy)
            if r <= cx:
                angle = np.arctan2(dy, dx)
                hue = int((angle + np.pi) / (2*np.pi) * 179)
                sat = 255
                val = int(r / cx * 255)
                wheel[y, x] = (hue, sat, val)
    return cv2.cvtColor(wheel, cv2.COLOR_HSV2BGR)

cap = cv2.VideoCapture("personnes_en_mouvement.mp4")

# Créer le disque de couleur
color_wheel = create_hsv_color_wheel(100)

# Lire la première image
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul du flot optique dense
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convertir en HSV pour affichage
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Placer le disque de couleur en haut à gauche
    h, w, _ = color_wheel.shape
    rgb_flow[10:10+h, 10:10+w] = color_wheel

    cv2.imshow('Dense Optical Flow - Farnebäck', rgb_flow)

    prev_gray = next_gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
