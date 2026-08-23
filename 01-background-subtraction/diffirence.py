# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:19:55 2026

@author: jawad_el_motabit
"""

import cv2
import numpy as np
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("personnes_en_mouvement.mp4")

kernel = np.ones((5,5), np.uint8)
ret, frame1 =cap.read()
gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while True:
    ret, frame2 =cap.read()
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_frame1, gray_frame2)
    _, thresh = cv2.threshold(frame_diff,25,255, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Foregrounde", thresh)
    gray_frame1 = gray_frame2
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()
