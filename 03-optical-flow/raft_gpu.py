import sys
import os
import argparse
import torch
import cv2
import numpy as np
from collections import OrderedDict

# --------------------------
# Ajouter les chemins RAFT
# --------------------------
raft_root = r"D:\enseignement\Developpement\visual analytics\2025\RAFT"
sys.path.append(raft_root)
sys.path.append(os.path.join(raft_root, "core"))   # pour raft.py et ses dépendances
sys.path.append(os.path.join(raft_root, "utils"))  # pour utils

from raft import RAFT
from utils.utils import InputPadder

# --------------------------
# Créer les arguments pour RAFT
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')  # True pour modèle léger
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args([])  # vide pour ne pas lire sys.argv
args.small = False  # mettre True si vous voulez le petit modèle

# --------------------------
# Charger le modèle RAFT
# --------------------------
model = RAFT(args)

state_dict = torch.load(os.path.join(raft_root, "models/raft-kitti.pth"))
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.eval().cuda()  # GPU si disponible

# --------------------------
# Initialiser la webcam
# --------------------------
cap = cv2.VideoCapture("personnes_en_mouvement.mp4")
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
prev_frame = torch.from_numpy(prev_frame).permute(2,0,1).float()[None].cuda()

padder = InputPadder(prev_frame.shape)

# --------------------------
# Boucle principale
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float()[None].cuda()

    prev_frame_pad, frame_tensor_pad = padder.pad(prev_frame, frame_tensor)

    with torch.no_grad():
        _, flow_up = model(prev_frame_pad, frame_tensor_pad, iters=20, test_mode=True)

    flow = flow_up[0].permute(1,2,0).cpu().numpy()

    # Visualiser le flux optique
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("RAFT Optical Flow", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = frame_tensor

cap.release()
cv2.destroyAllWindows()
