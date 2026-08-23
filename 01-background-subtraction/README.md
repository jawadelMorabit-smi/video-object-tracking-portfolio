# 01 — Background Subtraction

> 🇫🇷 [Readme en français](README.fr.md)


Moving-object detection by modeling the scene background and marking everything that deviates from it as foreground.

![Background subtraction concept](../assets/bg_subtraction_concept.png)

## How it works

Every method here follows the same skeleton — only the *background model* changes:

1. **Model the background** — MOG2 fits a Gaussian mixture per pixel; KNN keeps recent samples per pixel; GMG does Bayesian first-frames estimation
2. **Classify each new pixel** — foreground if it doesn't fit the learned model
3. **Clean up** — morphological opening removes speckle, contour extraction + area filter yields bounding boxes

## Scripts

| Script | What it does |
|--------|-------------|
| `diffirence.py` | Naive **frame differencing**: absolute difference of two consecutive grayscale frames + threshold. No background memory — fast but leaves holes in slow-moving objects. |
| `MOG2.py` | **MOG2** (adaptive Gaussian Mixture Model): each pixel modeled by a mixture of Gaussians; detects shadows (marked gray in the mask). |
| `Knn.py` | **KNN** background model: classifies a pixel as background if it is close enough to at least K of the recent samples. |
| `knnavance.py` | KNN mask → contour detection → bounding boxes around moving people, with an area filter to drop noise. |
| `mog2&knn.py` | Side-by-side display of MOG2 vs KNN masks on the same frames for direct comparison. |
| `mogmog2gmg.py` | Adds **GMG** to the comparison — three statistical models side by side. |
| `mog2knnAvance.py` | Benchmark class: FPS measurement, morphological cleanup, per-method stats on a video file. |
| `mog2knnV2.py` | Final consolidated version of the benchmark pipeline. |

## Key takeaways

- **Shadow handling matters**: MOG2 marks shadows gray instead of white, which prevents them from merging nearby detections.
- **Morphological opening** (`cv2.morphologyEx`) removes speckle noise before contour extraction.
- **Area filtering** bounding boxes (> ~1000 px) removes residual noise blobs.
- Statistical models (MOG2/KNN) adapt to gradual illumination changes; plain differencing does not.

## Run

```bash
python mog2knnV2.py          # benchmark on personnes_en_mouvement.mp4
python knnavance.py          # live webcam with bounding boxes
```

> Video note: scripts expect a pedestrian video named `personnes_en_mouvement.mp4` in the working directory (not redistributed). Any similar clip works.

## Lab statement
See [Tp01.pdf](Tp01.pdf).
