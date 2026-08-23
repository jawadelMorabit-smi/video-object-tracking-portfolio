# Portefeuille d'Analyse Vidéo & Suivi d'Objets

**De la soustraction de fond au suivi siaméen — un parcours progressif en analyse vidéo**
*Master BIAM 2025–2026 · Vidéo Biomédicale · FSDM, Université Sidi Mohamed Ben Abdellah*

> 🇬🇧 [Readme in English](README.md)

---

Ce dépôt rassemble cinq travaux pratiques couvrant les techniques fondamentales de l'analyse vidéo, ordonnées des méthodes classiques vers l'apprentissage profond. Chaque module est autonome : son propre README, ses scripts exécutables et son énoncé.

## Le Parcours

![Le parcours — cinq modules, du classique au suivi profond](assets/journey_roadmap.png)

| # | Module | Famille de techniques | Outils clés |
|---|--------|----------------------|-------------|
| 01 | [Soustraction de fond](01-background-subtraction/README.fr.md) | Différence d'images, GMM (MOG2), KNN, GMG | OpenCV |
| 02 | [Détection de piétons — HOG & YOLOv5](02-pedestrian-detection-hog-yolo/README.fr.md) | Descripteurs classiques vs détection profonde one-stage | OpenCV HOG, YOLOv5 |
| 03 | [Flot optique](03-optical-flow/README.fr.md) | Flot épars (Lucas–Kanade) & dense (Farnebäck, CLG), estimation profonde | OpenCV, PyTorch, RAFT |
| 04 | [Suivi multi-objets — DeepSORT](04-multi-object-tracking-deepsort/README.fr.md) | Tracking-by-detection, filtre de Kalman + ré-identification | YOLOv8, DeepSORT, Mask R-CNN |
| 05 | [Suivi mono-objet — PySOT](05-single-object-tracking-pysot/README.fr.md) | Réseaux siamés : appariement de gabarit SiamRPN++ | PySOT |

## Galerie de Résultats

Suivi multi-objets YOLOv8 + DeepSORT sur une scène urbaine — identités stables maintenues d'une image à l'autre :

<p>
  <img src="assets/deepsort_frame_1.jpg" width="32%" alt="Suivi DeepSORT image 1"/>
  <img src="assets/deepsort_frame_2.jpg" width="32%" alt="Suivi DeepSORT image 2"/>
  <img src="assets/deepsort_frame_3.jpg" width="32%" alt="Suivi DeepSORT image 3"/>
</p>

Suivi mono-objet SiamRPN++ (PySOT) d'un sac malgré occlusions et changements d'échelle :

![Démo de suivi SiamRPN++](05-single-object-tracking-pysot/bag_demo.gif)

## Contenu de chaque module

### 01 — Soustraction de fond
La base de la détection d'objets en mouvement : séparer le premier plan d'un modèle de fond appris.
- **Différence d'images** : approche la plus simple, sans mémoire du passé
- **MOG2 / KNN / GMG** : modèles statistiques de fond comparés côte à côte
- Script de benchmark mesurant FPS et qualité de masque pour chaque méthode, avec extraction de contours et boîtes englobantes

### 02 — Détection de piétons
Détection classique par HOG+SVM (fenêtre glissante) face au détecteur one-stage YOLOv5 sur des images de piétons.

### 03 — Flot optique
Le mouvement comme champ de vecteurs par pixel :
- **Lucas–Kanade** : suivi épars de coins Shi–Tomasi avec trajectoires
- **Farnebäck** : flot dense avec visualisation HSV (roue chromatique)
- **RAFT**, modèle profond récurrent pré-entraîné, bascule automatique CPU/GPU
- Compte rendu écrit complet (`COMPTE_RENDU.md`, en français)

### 04 — Suivi multi-objets
Le tracking-by-detection bien fait :
- Détection YOLOv8 → association DeepSORT (prédiction par filtre de Kalman + appariement d'apparence par Re-ID)
- Estimation de vitesse par piste
- Mask R-CNN (segmentation d'instances) combiné à DeepSORT, plus tableau comparatif des deux pipelines

### 05 — Suivi mono-objet
Traceurs siamés : apprendre un gabarit sur la première image, puis régresser sa position dans chacune des suivantes.
- SiamRPN++ (backbone ResNet-50) via le framework PySOT
- Pipeline de démo sur `bag.avi` produisant le GIF ci-dessus

## Structure du dépôt

```
├── 01-background-subtraction/        Scripts MOG2 / KNN / GMG + benchmark
├── 02-pedestrian-detection-hog-yolo/ Notes HOG + mise en place YOLOv5
├── 03-optical-flow/                  LK, Farnebäck, CLG, RAFT + compte rendu
├── 04-multi-object-tracking-deepsort/ Notebooks YOLOv8+DeepSORT + cœur DeepSORT
├── 05-single-object-tracking-pysot/  Script de démo SiamRPN++ + GIF résultat
└── assets/                           Images des README
```

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

> Les poids pré-entraînés volumineux (modèles `.pth` de RAFT, `ckpt.t7` de DeepSORT, `.pt` de YOLOv8) ne sont **pas stockés** dans ce dépôt. Chaque README de module indique précisément où les télécharger.

La plupart des modules ont été développés autour d'une séquence test commune (`personnes_en_mouvement.mp4`) ; toute vidéo de piétons peut la remplacer directement.

## Index des modules

| Module | Pour commencer | Énoncé |
|--------|---------------|--------|
| Soustraction de fond | [`mog2knnV2.py`](01-background-subtraction/mog2knnV2.py) | [Tp01.pdf](01-background-subtraction/Tp01.pdf) |
| Détection de piétons | [`hog.py`](02-pedestrian-detection-hog-yolo/hog.py) | — |
| Flot optique | [`raft_cpu.py`](03-optical-flow/raft_cpu.py) | [TP3.pdf](03-optical-flow/TP3.pdf) |
| Suivi multi-objets | [`TP_DeepSORT.ipynb`](04-multi-object-tracking-deepsort/TP_DeepSORT.ipynb) | [TP4.pdf](04-multi-object-tracking-deepsort/TP4.pdf) |
| Suivi mono-objet | [`demo_pysot.py`](05-single-object-tracking-pysot/demo_pysot.py) | [TP5.pdf](05-single-object-tracking-pysot/TP5.pdf) |

## Auteur

**Jaouad El Morabit** — Master BIAM 2025–2026, Imagerie Biomédicale

Voir aussi mon autre projet : [radiogenomics-analytics-framework](https://github.com/jawadelMorabit-smi/radiogenomics-analytics-framework) — prédire la méthylation de MGMT à partir d'IRM par radiomique et CNNs 3D.
