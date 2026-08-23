"""
TP3 - Partie 1, Question 1.3 : effet des parametres de Farneback
===============================================================

cv2.calcOpticalFlowFarneback(prev, next, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

Ce script affiche cote a cote plusieurs jeux de parametres pour observer
visuellement leur influence sur le flot optique dense.

Resume de l'effet de chaque parametre :
  - pyr_scale (0.5)  : ratio de reduction d'une echelle a l'autre dans la
                       pyramide. < 1. Proche de 1 = pyramide fine (plus lente).
  - levels    (3)    : nombre de niveaux de la pyramide. Plus eleve = capte
                       les GRANDS deplacements ; 1 = pas de pyramide.
  - winsize   (15)   : taille de la fenetre de moyennage. Grand = robuste au
                       bruit et aux grands mouvements MAIS resultat plus flou/lisse,
                       petits details perdus. Petit = net mais bruite.
  - iterations(3)    : nombre d'iterations de l'algorithme a chaque niveau.
                       Plus = plus precis mais plus lent.
  - poly_n    (5)    : taille du voisinage pour l'expansion polynomiale (5 ou 7).
                       Grand = surface plus lisse, mouvement plus "robuste" mais flou.
  - poly_sigma(1.2)  : ecart-type du lissage gaussien de l'expansion.
                       ~1.1 pour poly_n=5, ~1.5 pour poly_n=7.
  - flags     (0)    : 0 ou cv2.OPTFLOW_FARNEBACK_GAUSSIAN (fenetre gaussienne =
                       plus precis mais plus lent).

Appuyez sur 'q' pour quitter.
"""

import cv2
import numpy as np

# Jeux de parametres a comparer : (nom, dict de parametres Farneback)
PARAM_SETS = [
    ("Defaut",          dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)),
    ("winsize=5 (net)", dict(pyr_scale=0.5, levels=3, winsize=5,  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)),
    ("winsize=35 (flou)",dict(pyr_scale=0.5, levels=3, winsize=35, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)),
    ("levels=5+gauss",  dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=7, poly_sigma=1.5,
                             flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)),
]


def flow_to_bgr(flow, frame_shape):
    """Convertit un champ de flot (H,W,2) en image couleur HSV->BGR."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label(img, text):
    """Ecrit un titre en haut a gauche de l'image."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


cap = cv2.VideoCapture("personnes_en_mouvement.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Pour une mosaique 2x2 on redimensionne chaque vignette de moitie.
scale = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vignettes = []
    for name, params in PARAM_SETS:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, **params)
        vis = flow_to_bgr(flow, frame.shape)
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
        vignettes.append(label(vis, name))

    # Mosaique 2x2
    top = np.hstack((vignettes[0], vignettes[1]))
    bottom = np.hstack((vignettes[2], vignettes[3]))
    mosaic = np.vstack((top, bottom))

    cv2.imshow("Farneback - effet des parametres (Q1.3)", mosaic)

    prev_gray = next_gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
