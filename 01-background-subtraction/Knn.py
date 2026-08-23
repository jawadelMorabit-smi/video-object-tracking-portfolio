# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:34:40 2026

@author: jawad_el_motabit
"""

import cv2

bg_substractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame =cap.read()
    if not ret:
        break
    fg_mask = bg_substractor.apply(frame)
    cv2.imshow("knn foeground mask", fg_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()