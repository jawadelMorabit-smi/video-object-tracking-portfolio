# -*- coding: utf-8 -*-
"""
Export a short preview of the MOG2 vs KNN comparison (mog2knnV2.py view)
as MP4 + GIF, for embedding in the module README.

Usage:
    python export_preview.py [seconds]      # default: 5
"""

import cv2
import numpy as np
import sys

CONFIG = {
    "video_path": "personnes_en_mouvement.mp4",
    "history": 500,
    "mog2_var_threshold": 16,
    "knn_dist_threshold": 500,
    "detect_shadows": True,
    "morph_kernel_size": (3, 3),
}

GIF_SCALE = 0.5     # GIF width factor (combo is 3 panels wide)
GIF_FPS = 12        # GIF frame rate (half of source -> smaller file)


def init_algorithms():
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=CONFIG["history"],
        varThreshold=CONFIG["mog2_var_threshold"],
        detectShadows=CONFIG["detect_shadows"])
    knn = cv2.createBackgroundSubtractorKNN(
        history=CONFIG["history"],
        dist2Threshold=CONFIG["knn_dist_threshold"],
        detectShadows=CONFIG["detect_shadows"])
    return mog2, knn


def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG["morph_kernel_size"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def make_panel(frame, mog2, knn):
    """Reproduces exactly the layout shown by mog2knnV2.run_visualization()."""
    w = frame.shape[1]
    m_mog2 = apply_morphology(mog2.apply(frame))
    m_knn = apply_morphology(knn.apply(frame))

    vis_mog2 = cv2.cvtColor(m_mog2, cv2.COLOR_GRAY2BGR)
    vis_knn = cv2.cvtColor(m_knn, cv2.COLOR_GRAY2BGR)
    vis_mog2[m_mog2 == 127] = [255, 0, 0]   # shadows in blue
    vis_knn[m_knn == 127] = [255, 0, 0]

    combo = np.hstack((frame, vis_mog2, vis_knn))
    cv2.putText(combo, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combo, "MOG2", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combo, "KNN", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return combo


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    cap = cv2.VideoCapture(CONFIG["video_path"])
    assert cap.isOpened(), f"video not found: {CONFIG['video_path']}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    max_frames = int(fps * seconds)

    mog2, knn = init_algorithms()

    ret, frame = cap.read()
    assert ret, "empty video"
    combo = make_panel(frame, mog2, knn)
    h, w = combo.shape[:2]

    out_mp4 = "demo_mog2_knn.mp4"
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
    print(f"[1/2] writing {out_mp4}  ({w}x{h} @ {fps:.1f} fps, {max_frames} frames)")

    gif_frames = []
    n = 0
    while True:
        writer.write(combo)
        # keep every other frame for the GIF (GIF_FPS = source/2)
        if n % 2 == 0:
            small = cv2.resize(combo, (0, 0), fx=GIF_SCALE, fy=GIF_SCALE)
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        n += 1
        if n >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        combo = make_panel(frame, mog2, knn)

    writer.release()
    cap.release()
    print(f"      wrote {len(gif_frames)*2} frames")

    import imageio.v2 as imageio
    out_gif = "../assets/bg_subtraction_demo.gif"
    print(f"[2/2] writing {out_gif}  ({gif_frames[0].shape[1]}x{gif_frames[0].shape[0]} @ {GIF_FPS} fps)")
    imageio.mimsave(out_gif, gif_frames, fps=GIF_FPS, loop=0)
    print("done.")


if __name__ == "__main__":
    main()
