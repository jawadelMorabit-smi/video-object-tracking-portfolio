"""
TP3 - Partie 2 : Flot optique avec RAFT (version CPU / GPU-auto)
================================================================

Ce script teste un modele RAFT pre-entraine sur une video reelle.
Il repond aux points 2.1 a 2.5 de l'enonce :

  2.1  raft_root        -> chemin vers le depot RAFT clone (a adapter)
  2.2  MODEL            -> modele pre-entraine a charger (au choix dans models/)
  2.3  source d'entree  -> un fichier video au lieu de la webcam
  2.4  visualisation    -> representation HSV (voir variantes plus bas)
  2.5  GPU              -> bascule automatique sur GPU si une carte NVIDIA est dispo,
                          sinon execution sur CPU (plus lent, c'est normal).

Pre-requis (a faire une seule fois) :
    git clone https://github.com/princeton-vl/RAFT
    # telecharger models.zip puis le decompresser dans RAFT/  -> RAFT/models/*.pth
    #   https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
"""

import sys
import os
import argparse
from collections import OrderedDict

import torch
import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration (a adapter a votre machine)
# --------------------------------------------------------------------------
# 2.1 : chemin vers le depot RAFT. Par defaut on cherche un dossier "RAFT"
#       place a cote de ce script. Remplacez par un chemin absolu si besoin.
raft_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAFT")

# 2.2 : modele pre-entraine. Choix possibles dans RAFT/models/ :
#       raft-things.pth, raft-kitti.pth, raft-sintel.pth, raft-small.pth, ...
MODEL = "models/raft-things.pth"

# 2.3 : source video (fichier au lieu de la webcam cv2.VideoCapture(0)).
VIDEO = "personnes_en_mouvement.mp4"

# Nombre d'iterations de raffinement RAFT (12 = compromis vitesse/qualite sur CPU).
ITERS = 12

# 2.5 : device automatique -> GPU si dispo, sinon CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Imports RAFT (depuis le depot clone)
# --------------------------------------------------------------------------
sys.path.append(raft_root)
sys.path.append(os.path.join(raft_root, "core"))   # raft.py et ses dependances
sys.path.append(os.path.join(raft_root, "utils"))  # utilitaires

if not os.path.isdir(raft_root):
    raise FileNotFoundError(
        f"Depot RAFT introuvable : {raft_root}\n"
        "Clonez-le d'abord : git clone https://github.com/princeton-vl/RAFT"
    )

from raft import RAFT
from utils.utils import InputPadder

print(f"PyTorch version : {torch.__version__}")
print(f"Device utilise  : {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (pas de GPU NVIDIA detecte)"))

# --------------------------------------------------------------------------
# Arguments attendus par le constructeur RAFT
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='utiliser le modele leger')
parser.add_argument('--mixed_precision', action='store_true', help='precision mixte')
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args([])     # liste vide : on ne lit pas sys.argv
args.small = False               # True si vous chargez raft-small.pth

# --------------------------------------------------------------------------
# Chargement du modele
# --------------------------------------------------------------------------
model = RAFT(args)

model_path = os.path.join(raft_root, MODEL)
# map_location=device : charge correctement un poids GPU sur une machine CPU.
# weights_only=False : compatibilite avec les versions recentes de PyTorch.
state_dict = torch.load(model_path, map_location=device, weights_only=False)

# Les poids sont sauvegardes avec un prefixe "module." (DataParallel) : on le retire.
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device).eval()   # 2.5 : .to(device) marche en CPU comme en GPU

# --------------------------------------------------------------------------
# Lecture de la video (2.3)
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Impossible d'ouvrir la video : {VIDEO}")

ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Video vide ou illisible.")

prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
prev_tensor = torch.from_numpy(prev_rgb).permute(2, 0, 1).float()[None].to(device)

# InputPadder : ajuste la taille aux multiples requis par RAFT (cree une fois).
padder = InputPadder(prev_tensor.shape)
prev_pad = padder.pad(prev_tensor)[0]

# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()[None].to(device)
    frame_pad = padder.pad(frame_tensor)[0]

    with torch.no_grad():
        _, flow_up = model(prev_pad, frame_pad, iters=ITERS, test_mode=True)

    # On enleve le padding pour revenir a la taille d'origine.
    flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()

    # ----------------------------------------------------------------------
    # 2.4 : visualisation HSV
    #   - Hue        (canal 0) -> ANGLE / direction du mouvement
    #   - Saturation (canal 1) -> 255 (couleur pleine)
    #   - Value      (canal 2) -> MAGNITUDE (intensite du mouvement)
    # ----------------------------------------------------------------------
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2                       # teinte = direction
    hsv[..., 1] = 255                                         # saturation pleine
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # valeur = vitesse
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow(f"RAFT Optical Flow ({device})", bgr)

    prev_pad = frame_pad
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
