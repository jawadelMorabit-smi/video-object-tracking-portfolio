# 02 — Détection d'objets : HOG+SVM vs YOLO

Deux générations de détection d'objets, appliquées à deux problèmes :

![HOG+SVM vs YOLO](../assets/hog_vs_yolo.png)

## 🌟 Projet phare : Détection de la cataracte (mini-projet)

Une comparaison appliquée complète sur une tâche d'imagerie médicale : détecter la **cataracte** sur des photographies d'yeux ([dataset Roboflow](https://universe.roboflow.com/jaouads-workspace/cataract-eye-detection), 2 classes).

| Métrique | HOG + SVM | YOLOv5 |
|---|---|---|
| Précision | 93,53 % | **94,50 %** |
| Rappel | 93,43 % | 90,89 % |
| mAP@0.5 / F1 | — | **96,55 %** / 80,3 % mAP@0.5:0.95 |
| Entraînement | 52 s (CPU) | 52 min (GPU T4) |
| Inférence/image | 0,03 ms | ~6,5 ms (~150 FPS) |

> **Provenance des données** : construit à partir du [Eye Detection Dataset brut (Kaggle)](https://www.kaggle.com/datasets/icebearogo/eye-detection-dataset) (~2000 images d'yeux annotées YOLO), puis curatisé dans Roboflow — nettoyage des labels, normalisation des IDs de classes (`0=sain`, `1=cataracte`, vérifiée visuellement) et augmentation (1385 → 4135 images de train).

**Projet complet avec notebooks exécutés, figures et analyse :**

👉 **[jawadelMorabit-smi/eye-cataract-detection](https://github.com/jawadelMorabit-smi/eye-cataract-detection)**

Aussi sur Kaggle : [hog_svm_cataract_roboflow](https://www.kaggle.com/code/jaouadelmorabit/hog-svm-cataract-roboflow) · [yolo_cataract_detection](https://www.kaggle.com/code/jaouadelmorabit/yolo-cataract-detection)

## Détection de piétons (exercice de TP)

Les mêmes techniques appliquées à des images de rue via le dépôt officiel YOLOv5 dans un environnement virtuel dédié. Le framework n'est pas réhébergé ici — clonez-le en amont :

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
python detect.py --weights yolov5s.pt --source video.mp4   # poids COCO pré-entraînés
```

## Enseignements clés

- HOG+SVM s'entraîne en quelques secondes sans GPU et reste interprétable, mais peine face aux occlusions, variations d'échelle et flou de mouvement.
- YOLOv5 détecte tous les objets en une seule passe, gère les scènes denses, tourne en temps réel sur GPU et fournit nativement un score de confiance par boîte.
- L'isolation d'environnement (`venv`) est essentielle : la pile de dépendances de YOLOv5 entre facilement en collision avec d'autres projets.
- Leçon appliquée du projet cataracte : les méthodes classiques et profondes sont plus proches qu'on ne le croit sur la précision ; les vrais différenciateurs sont le coût d'entraînement, la vitesse d'inférence et les scores de confiance.
