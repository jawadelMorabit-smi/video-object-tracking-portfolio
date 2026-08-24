# 02 — Object Detection: HOG+SVM vs YOLO

Two generations of object detection, applied to two problems:

![HOG+SVM vs YOLO](../assets/hog_vs_yolo.png)

## 🌟 Flagship: Eye Cataract Detection (mini-project)

A complete applied comparison on a medical imaging task: detecting **cataracts** from eye photographs ([Roboflow dataset](https://universe.roboflow.com/jaouads-workspace/cataract-eye-detection), 2 classes).

| Metric | HOG + SVM | YOLOv5 |
|---|---|---|
| Precision | 93.53% | **94.50%** |
| Recall | 93.43% | 90.89% |
| mAP@0.5 / F1 | — | **96.55%** / 80.3% mAP@0.5:0.95 |
| Training | 52 s (CPU) | 52 min (GPU T4) |
| Inference/image | 0.03 ms | ~6.5 ms (~150 FPS) |

> **Data provenance**: built on the raw [Eye Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/icebearogo/eye-detection-dataset) (~2000 YOLO-annotated eye images), then curated in Roboflow — label cleaning, class-ID normalization (`0=sain`, `1=cataracte`, verified visually) and augmentation (1385 → 4135 train images).

**Full project with executed notebooks, figures and analysis:**

👉 **[jawadelMorabit-smi/eye-cataract-detection](https://github.com/jawadelMorabit-smi/eye-cataract-detection)**

Also on Kaggle: [hog_svm_cataract_roboflow](https://www.kaggle.com/code/jaouadelmorabit/hog-svm-cataract-roboflow) · [yolo_cataract_detection](https://www.kaggle.com/code/jaouadelmorabit/yolo-cataract-detection)

## Pedestrian detection (lab exercise)

The same techniques applied to street footage using the official YOLOv5 repository in a dedicated virtual environment. The framework is not re-hosted here — clone it upstream:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
python detect.py --weights yolov5s.pt --source video.mp4   # pretrained COCO weights
```

## Key takeaways

- HOG+SVM trains in seconds without GPU and stays interpretable, but struggles with occlusion, scale variation and motion blur.
- YOLOv5 detects all objects in one forward pass, handles crowded scenes, runs real-time on GPU, and provides native per-box confidence.
- Environment isolation (`venv`) matters: YOLOv5's dependency stack clashes easily with other projects.
- Applied lesson from the cataract project: classical and deep methods are closer than expected on precision; the practical differentiators are training cost, inference speed, and confidence scores.
