# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:53:52 2026

@author: jawad_el_motabit
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
bg_substractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)


while True:
    ret, frame =cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480,270))
    
    mask_mog2 = mog2.apply(frame)
    mask_knn = bg_substractor.apply(frame)
    
    cv2.imshow("MOG2", mask_mog2)
    cv2.imshow("KNN", mask_knn)
    

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'): 
        cv2.imwrite("ground_truth_frame.jpg", frame)
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()

img = cv2.imread("ground_truth_frame.jpg", 0)
_, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("ground_truth_mask.png", binary_mask)


f1_score_mog2 = calculate_f1(mog2_mask, ground_truth_mask)

f1_score_knn = calculate_f1(knn_mask, ground_truth_mask)

print(f"MOG2 F1: {f1_score_mog2:.2f} | KNN F1: {f1_score_knn:.2f}")q

