# TP3 — Flot Optique
**Master BIAM — Vidéo Biomédicale — Université Sidi Mohamed Ben Abdellah, FSDM**
Année universitaire 2025/2026

---

## 0. Introduction — qu'est-ce que le flot optique ?

Le **flot optique** (*optical flow*) est le champ de vecteurs décrivant le **déplacement
apparent** de chaque pixel entre deux images successives d'une vidéo. Pour chaque pixel
`(x, y)` on estime un vecteur `(u, v)` indiquant de combien il s'est déplacé entre
l'instant `t` et `t+1`.

On distingue :

| Type | Principe | Exemple |
|------|----------|---------|
| **Flot épars** (*sparse*) | On suit seulement quelques points d'intérêt | Lucas-Kanade (`OF_LK.py`) |
| **Flot dense** (*dense*) | On calcule un vecteur pour **chaque** pixel | Farneback (`OF_FB.py`), RAFT |

Hypothèse fondamentale : la **conservation de l'intensité** d'un pixel suivi le long de
son mouvement → `I(x, y, t) = I(x+u, y+v, t+1)`.

Dans ce TP, deux méthodes de flot **dense** sont mises en œuvre :
- **Farneback** : méthode classique basée sur les pyramides gaussiennes et l'expansion polynomiale.
- **RAFT** : modèle d'apprentissage profond (réseau récurrent).

---

## Partie 1 — Méthode de Farneback (OpenCV)

### 1.1 Principe
La méthode de Gunnar **Farneback** estime le flot **dense**. Elle approxime le voisinage
de chaque pixel par un **polynôme quadratique**, puis déduit le déplacement à partir de la
variation de ces polynômes entre les deux images. L'usage d'une **pyramide gaussienne**
(images de résolutions décroissantes) permet de capter les **grands déplacements**.

### 1.2 Implémentation
Fichier : **`OF_FB.py`** (déjà fonctionnel). Points clés :
- lecture de la vidéo `personnes_en_mouvement.mp4` (au lieu de la webcam `VideoCapture(0)`) ;
- appel `cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)` ;
- visualisation **HSV** : teinte = direction, valeur = magnitude ;
- bonus : une **roue de couleurs** (`create_hsv_color_wheel`) sert de légende des directions.

Lancement :
```
python OF_FB.py
```

### 1.3 Question — effet des paramètres de `calcOpticalFlowFarneback()`

Signature :
```python
cv2.calcOpticalFlowFarneback(prev, next, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
```

| Paramètre | Valeur déf. | Rôle | Effet quand on l'augmente |
|-----------|-------------|------|----------------------------|
| `pyr_scale` | 0.5 | ratio de réduction entre niveaux de pyramide (< 1) | proche de 1 → pyramide plus fine, plus précis mais plus lent |
| `levels` | 3 | nombre de niveaux de pyramide | capte de **plus grands déplacements** ; 1 = pas de pyramide |
| `winsize` | 15 | taille de la fenêtre de moyennage | plus **robuste au bruit** et aux grands mouvements, **mais flou**, détails fins perdus |
| `iterations` | 3 | itérations par niveau | **plus précis** mais plus lent |
| `poly_n` | 5 | taille du voisinage de l'expansion polynomiale (5 ou 7) | champ plus **lisse/robuste** mais plus flou |
| `poly_sigma` | 1.2 | écart-type du lissage gaussien (≈1.1 si poly_n=5 ; ≈1.5 si poly_n=7) | lissage plus fort |
| `flags` | 0 | options | `cv2.OPTFLOW_FARNEBACK_GAUSSIAN` → fenêtre gaussienne, plus précis mais plus lent |

**Observations attendues** (script de démonstration : **`OF_FB_params.py`**, mosaïque 2×2) :
- `winsize` **petit (5)** → flot net mais **bruité**, beaucoup de petits vecteurs parasites.
- `winsize` **grand (35)** → flot très **lissé**, les contours des personnes sont « étalés ».
- `levels=5` + `OPTFLOW_FARNEBACK_GAUSSIAN` → meilleure capture des **mouvements rapides**,
  résultat plus stable, au prix du **temps de calcul**.

Lancement de la comparaison :
```
python OF_FB_params.py
```

---

## Partie 2 — Méthode RAFT (réseau de neurones)

### 2.0 Principe
**RAFT** (*Recurrent All-Pairs Field Transforms*, Teed & Deng, ECCV 2020) calcule le flot
dense par apprentissage profond. Il :
1. extrait des caractéristiques des deux images (encodeur CNN) ;
2. construit un **volume de corrélation « toutes paires »** entre les pixels ;
3. **raffine itérativement** le champ de flot via une unité **récurrente (GRU)**.

Il est nettement plus **précis** que Farneback, surtout sur les bords et les mouvements
complexes, mais plus **coûteux** (très lent en CPU).

### Installation (pré-requis)
```bash
git clone https://github.com/princeton-vl/RAFT
cd RAFT
# Télécharger models.zip puis le décompresser dans RAFT/  -> RAFT/models/*.pth
#   https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
pip install torch torchvision opencv-python
```
Test rapide fourni par le dépôt :
```bash
python demo.py --model=models/raft-things.pth --path=demo-frames
```

### Réponses aux questions 2.1 → 2.5
Implémentées dans le fichier **`raft_cpu.py`** (version robuste CPU/GPU).

- **2.1 — `raft_root`** : remplacé le chemin du professeur
  (`D:\enseignement\...\RAFT`) par un chemin relatif au script :
  ```python
  raft_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAFT")
  ```
  (à remplacer par un chemin absolu si le dépôt est ailleurs).

- **2.2 — modèle pré-entraîné** : paramétrable en tête de fichier, ex.
  `MODEL = "models/raft-things.pth"` (autres choix : `raft-kitti.pth`,
  `raft-sintel.pth`, `raft-small.pth`). Chargé avec `map_location=device`.

- **2.3 — source vidéo** : `cv2.VideoCapture("personnes_en_mouvement.mp4")` au lieu de la
  webcam `VideoCapture(0)`.

- **2.4 — visualisation HSV** : le champ de flot `(u, v)` est converti en couleur :
  | Canal | Contenu | Signification |
  |-------|---------|---------------|
  | `hsv[...,0]` **Hue** | angle de `(u,v)` | **direction** du mouvement |
  | `hsv[...,1]` **Saturation** | 255 | couleur pleine |
  | `hsv[...,2]` **Value** | magnitude normalisée | **intensité** (vitesse) du mouvement |

  *Variante possible* : mettre la **magnitude sur la saturation** (`hsv[...,1] = mag`,
  `hsv[...,2] = 255`) → fond blanc là où il n'y a pas de mouvement, parfois plus lisible.

- **2.5 — GPU** : au lieu du `.cuda()` en dur (qui plante sans carte NVIDIA), bascule
  automatique :
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device).eval()
  ```
  → le code tourne en **CPU** sur cette machine (PyTorch `+cpu`, pas de GPU détecté) et
  passera **automatiquement sur GPU** sur une machine équipée d'une NVIDIA GeForce.

Lancement :
```
python raft_cpu.py
```

---

## 3. Comparaison Farneback vs RAFT

| Critère | Farneback | RAFT |
|---------|-----------|------|
| Type | classique (pyramides + polynômes) | apprentissage profond (récurrent) |
| Précision | correcte, bords flous | élevée, bords nets |
| Vitesse | temps réel (CPU) | lent en CPU, rapide en GPU |
| Dépendances | OpenCV seul | PyTorch + dépôt + modèles |
| Réglage | nombreux paramètres manuels | modèle pré-entraîné |

---

## 4. Bonus (hors énoncé, fournis dans le dossier)
- **`OF_LK.py`** : flot optique **épars** de Lucas-Kanade (suivi de points Shi-Tomasi,
  trajectoires).
- **`OF_CLG_GPU.py`** : méthode **variationnelle** (Horn–Schunck / CLG-Bruhn) implémentée
  « à la main » en PyTorch, pyramide *coarse-to-fine* avec *warping*.

---

## 5. Fichiers du projet
| Fichier | Rôle |
|---------|------|
| `OF_FB.py` | Farneback dense (Partie 1) |
| `OF_FB_params.py` | comparaison des paramètres Farneback (Q1.3) |
| `raft_cpu.py` | RAFT CPU/GPU-auto (Partie 2, points 2.1–2.5) |
| `raft_gpu.py` | version d'origine (GPU obligatoire) |
| `OF_LK.py`, `OF_CLG_GPU.py` | bonus |
| `personnes_en_mouvement.mp4` | vidéo de test |

> **Note** : insérer ici des **captures d'écran** de chaque fenêtre (Farneback, mosaïque
> de paramètres, RAFT) pour illustrer le rendu.
