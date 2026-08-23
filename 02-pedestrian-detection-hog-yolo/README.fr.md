# 02 — Détection de piétons : HOG+SVM vs YOLOv5

Deux générations de détection d'objets appliquées à des images de piétons :

![HOG+SVM vs YOLOv5](../assets/hog_vs_yolo.png)

1. **HOG + SVM** (classique) : descripteurs Histogram of Oriented Gradients + SVM linéaire à fenêtre glissante ; OpenCV fournit un détecteur de piétons pré-entraîné (`cv2.HOGDescriptor_getDefaultPeopleDetector`).
2. **YOLOv5** (apprentissage profond) : détecteur CNN one-stage, exécuté via le dépôt officiel [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Contenu

- `hog.py` — script de détection HOG *(placeholder — voir statut ci-dessous)*
- L'expérience complète a été menée dans un clone local de **YOLOv5** avec son propre environnement virtuel. Le framework n'est pas réhébergé ici — clonez-le en amont :

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
python detect.py --weights yolov5s.pt --source video.mp4   # poids COCO pré-entraînés
```

## Enseignements clés

- HOG+SVM fonctionne sur des piétons nets et de face, mais peine face aux occlusions, aux petites échelles et au flou de mouvement.
- YOLOv5 détecte toutes les personnes en une seule passe, gère les scènes denses et tourne en temps réel sur GPU.
- Leçon pratique : l'isolation d'environnement (`venv`) est essentielle — la pile de dépendances de YOLOv5 (version de torch, conflits opencv-python-headless) entre facilement en collision avec d'autres projets.

## Séquence test
Clip de passage piéton (`personnes.mp4`, non redistribuée).
