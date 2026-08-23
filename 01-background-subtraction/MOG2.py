# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:01:40 2026

@author: jawad_el_motabit
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


while True:
    ret, frame =cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480,270))
    
    mask_mog2 = mog2.apply(frame)
    
    cv2.imshow("MOG2", mask_mog2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()