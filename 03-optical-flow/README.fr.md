# 03 — Flot Optique

Estimation du champ de mouvement par pixel entre images consécutives : pour chaque pixel, un vecteur `(u, v)` décrivant son déplacement entre l'instant `t` et `t+1`, sous l'hypothèse de constance de luminance `I(x, y, t) = I(x+u, y+v, t+1)`.

Le flot **épars** suit quelques points d'intérêt ; le flot **dense** calcule un vecteur pour *chaque* pixel.

![Principe du flot optique](../assets/optical_flow_concept.png)

## Le flot dense en pratique

Sortie Farnebäck sur la séquence test — la teinte code la direction du mouvement, la luminosité sa vitesse. Les deux piétons se détachent nettement du fond statique :

![Flot optique dense Farnebäck — image vs visualisation HSV](../assets/optical_flow_farneback.png)

## Scripts

| Script | Méthode | Type |
|--------|--------|------|
| `OF_LK.py` | **Lucas–Kanade** (`cv2.calcOpticalFlowPyrLK`) sur coins Shi–Tomasi, avec trajectoires dessinées sur un masque persistant | épars |
| `OF_FB.py` | **Farnebäck** (`cv2.calcOpticalFlowFarneback`) avec visualisation HSV par roue chromatique : teinte = direction, saturation = magnitude | dense |
| `OF_FB_params.py` | Farnebäck avec paramètres pyramide/itérations ajustables pour étudier leur effet | dense |
| `OF_CLG_GPU.py` | Méthode variationnelle **CLG** implémentée en PyTorch avec pyramide coarse-to-fine | dense |
| `raft_cpu.py` | **RAFT** (Recurrent All-Pairs Field Transforms), modèle profond pré-entraîné — bascule CPU/GPU automatique | dense (profond) |
| `raft_gpu.py` | Variante RAFT forcée sur GPU | dense (profond) |

## Mise en place de RAFT

Le code RAFT et les poids pré-entraînés ne sont pas redistribués :

```bash
git clone https://github.com/princeton-vl/RAFT
# télécharger les poids et décompresser dans RAFT/models/
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
```

`raft_cpu.py` pointe vers le clone via une variable `raft_root`, charge le checkpoint choisi (`raft-sintel.pth` par défaut), lit un fichier vidéo plutôt que la webcam, et visualise le flot en HSV. Il repasse automatiquement sur CPU sans GPU NVIDIA (plus lent, mais fonctionnel).

## Enseignements clés

- Lucas–Kanade est peu coûteux et précis sur les coins texturés mais abandonne les zones sans texture.
- Farnebäck couvre toute l'image mais lisse les contours d'objets.
- RAFT produit des frontières de mouvement bien plus nettes que les méthodes classiques et généralise entre scènes — au prix de calcul GPU.
- La roue chromatique HSV est la façon standard de lire un flot dense : la teinte encode la direction, la valeur encode la vitesse.

## Documentation

- 📄 Compte rendu complet : [`COMPTE_RENDU.md`](COMPTE_RENDU.md)
- Énoncé du TP : [`TP3.pdf`](TP3.pdf) / [`TP3.docx`](TP3.docx)
