# 05 — Suivi Multi-Objets : YOLOv8 + DeepSORT

Le tracking-by-detection : un détecteur propose des objets à chaque image, et un tracker les relie en identités cohérentes dans le temps.

![Pipeline DeepSORT](../assets/deepsort_pipeline.png)

## Pipeline

```
vidéo ──► détection YOLOv8 ──► association DeepSORT ──► boîtes suivies + IDs stables
                │                        │
                │                  filtre de Kalman (prédiction de mouvement,
                │                  gère les occlusions courtes)
                │                  + plongements d'apparence Re-ID
                │                  (appariement par distance cosinus)
                └── variante Mask R-CNN : masques de segmentation d'instances
                    alimentant la même logique d'association
```

## Contenu

| Fichier | Description |
|---------|-------------|
| [`TP_DeepSORT.ipynb`](TP_DeepSORT.ipynb) | ⭐ **Notebook principal** (prêt pour Colab, en français) : installation → suivi YOLOv8+DeepSORT avec estimation de vitesse par piste → segmentation Mask R-CNN+DeepSORT → tableau et graphique comparatifs |
| [`YOLOv8_DeepSORT_TRACKING_SCRIPT.ipynb`](YOLOv8_DeepSORT_TRACKING_SCRIPT.ipynb) | Version de travail antérieure du script de suivi |
| [`deep_sort_pytorch/`](deep_sort_pytorch/) | Le cœur du [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch) classique (filtre de Kalman, affectation hongroise, réseau Re-ID), utilisé pour l'étude |
| [`TP4.pdf`](TP4.pdf) | Énoncé du TP |

## Résultats

YOLOv8 + DeepSORT sur une scène urbaine — les identités restent stables malgré croisements et occlusions partielles :

<p align="center">
  <img src="../assets/deepsort_frame_1.jpg" width="30%"/>
  <img src="../assets/deepsort_frame_2.jpg" width="30%"/>
  <img src="../assets/deepsort_frame_3.jpg" width="30%"/>
</p>

## Exécution

Ouvrir `TP_DeepSORT.ipynb` dans Google Colab (runtime GPU recommandé) ou en local :

```bash
pip install ultralytics deep-sort-realtime
jupyter notebook TP_DeepSORT.ipynb
```

Le notebook accepte une vidéo téléversée ou un échantillon téléchargé ; les sorties annotées sont écrites dans `output/`.

## Poids

Non redistribués — téléchargés automatiquement par les bibliothèques :
- `yolov8n.pt` / `yolov8l.pt` : téléchargés automatiquement par `ultralytics`
- Encodeur Re-ID DeepSORT `ckpt.t7` : depuis [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)

## Enseignements clés

- La qualité de détection borne la qualité de suivi — des détections manquées brisent les chaînes d'identité.
- Le filtre de Kalman comble les occlusions courtes ; les caractéristiques d'apparence résolvent les permutations d'ID après croisement de piétons.
- Mask R-CNN ajoute des masques pixel-perfects à environ la moitié des FPS de la détection par boîtes (mesuré dans la section comparative du notebook).
